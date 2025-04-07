from torch import nn, optim
from data_classes.architecture import NeuralNetworkArchitecture
from data_classes.training_config import BaseTrainingConfig, OptimizationMethod
from data_classes.training_data import TrainingData
import torch

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class SequentialNeuralNetwork:
    def __init__(self, net_arch: NeuralNetworkArchitecture):
        self.net_arch = net_arch
        if net_arch.BATCH_NORMALIZATION:
            model = nn.Sequential(
                nn.Linear(net_arch.INPUT_DIM, net_arch.DEPTH),
                nn.BatchNorm1d(num_features=net_arch.DEPTH),
                net_arch.ACTIVATION_FUNCTION())
            for i in range(1, net_arch.NUM_HIDDEN_LAYERS):
                model = nn.Sequential(
                    model, nn.Linear(net_arch.DEPTH, net_arch.DEPTH),
                    nn.BatchNorm1d(num_features=net_arch.DEPTH),
                    net_arch.ACTIVATION_FUNCTION())
        else:
            model = nn.Sequential(
                nn.Linear(net_arch.INPUT_DIM, net_arch.DEPTH),
                net_arch.ACTIVATION_FUNCTION())
            for i in range(1, net_arch.NUM_HIDDEN_LAYERS):
                model = nn.Sequential(
                    model, nn.Linear(net_arch.DEPTH, net_arch.DEPTH),
                    net_arch.ACTIVATION_FUNCTION())
        self.model = nn.Sequential(
            model, nn.Linear(net_arch.DEPTH, net_arch.OUTPUT_DIM))
        
        self.model.apply(init_weights)
        
        # if net_arch.Xavier_init:
        #     model.apply(init_weights)

    
    def create_optimizer(self, config: BaseTrainingConfig):
        match config.OPTIMIZER:
            case OptimizationMethod.SGD:
                return torch.optim.SGD(
                    params=self.model.parameters(),
                    lr=config.LEARNING_RATE,
                    momentum=config.MOMENTUM,
                    weight_decay=config.REG_PARAM,
                    nesterov=config.NESTEROV,
                )
            case OptimizationMethod.ADAM:
                return torch.optim.Adam(
                    params=self.model.parameters(),
                    lr=config.LEARNING_RATE,
                    betas=config.BETAS,
                    eps=config.EPS,
                    weight_decay=config.REG_PARAM,
                )
            case OptimizationMethod.RMSPROP:
                return torch.optim.RMSprop(
                    params=self.model.parameters(),
                    lr=config.LEARNING_RATE,
                    momentum=config.MOMENTUM,
                    alpha=config.ALPHA,
                    eps=config.EPS,
                    centered=config.CENTERED,
                    weight_decay=config.REG_PARAM,
                )
            case _:
                raise NotImplementedError(f"Optimizer for {config.OPTIMIZER} is not implemented")


    def evaluate(self, data: TrainingData):
        test_objective = nn.L1Loss()
        output_train = self.model(data.train_x.float())
        train_error = test_objective(output_train, data.train_y.float()).item()

        output_test = self.model(data.test_x.float())
        generalization_error = test_objective(output_test, data.test_y.float()).item()

        self.train_error = train_error
        self.generalization_error = generalization_error


    def train(self, settings: BaseTrainingConfig, data: TrainingData):
        optimizer = self.create_optimizer(settings)
        train_x = data.train_x
        train_y = data.train_y

        train_objective = nn.MSELoss()

        for e in range(settings.NUM_EPOCHS):
            optimizer.zero_grad()
            output = self.model(train_x.float())
            loss = train_objective(output, train_y.float())
            loss.backward()
            optimizer.step()

        self.evaluate(data)

        self.training_results = {
            "learning_rate": settings.LEARNING_RATE,
            "reg_param": settings.REG_PARAM,
            "batch_norm": self.net_arch.BATCH_NORMALIZATION,
            "depth": self.net_arch.DEPTH,
            "num_hidden_layers": self.net_arch.NUM_HIDDEN_LAYERS,
            "activation": self.net_arch.ACTIVATION_FUNCTION.__name__,
            "train_error": self.train_error,
            "test_error": self.generalization_error,
            "train_size": len(data.train_x),
        }
