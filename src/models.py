from torch import nn, optim
from data_classes import NeuralNetworkArchitecture, BaseTrainingConfig, TrainingData, OptimizationMethod
import torch

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class NeuralNetwork:
    model: nn.Sequential
    train_objective = nn.MSELoss()
    test_objective = nn.L1Loss()

    def __init__(self, input_dimension: int, width: int, num_hidden_layers: int, learning_rate: float, regression_param: float):
        self.model = nn.Sequential(
            nn.Linear(input_dimension, width),
            nn.BatchNorm1d(num_features=width),
            nn.Sigmoid()
        )
        for _ in range(num_hidden_layers):
            self.model = nn.Sequential(
                self.model,
                nn.Linear(width, width),
                nn.BatchNorm1d(num_features=width),
                nn.Sigmoid()
            )
        self.model = nn.Sequential(
            self.model,
            nn.Linear(width, 1)
        )

        self.model.apply(init_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=regression_param)


class SequentialNeuralNetwork:
    def __init__(self, net_arch: NeuralNetworkArchitecture):
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
        model = nn.Sequential(
            model, nn.Linear(net_arch.DEPTH, net_arch.OUTPUT_DIM))
        
        # if net_arch.Xavier_init:
        #     model.apply(init_weights)

        self.model = model

    
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



    def train(self, settings: BaseTrainingConfig, data: TrainingData):
        optimizer = self.create_optimizer(settings)
        train_x = data.train_x
        train_y = data.train_y
        test_x = data.test_x
        test_y = data.test_y

        train_objective = nn.MSELoss()

        for e in range(settings.NUM_EPOCHS):
            optimizer.zero_grad()
            output = self.model(train_x.float())
            loss = train_objective(output, train_y.float())
            loss.backward()
            optimizer.step()

        test_objective = nn.L1Loss()
        output_train = self.model(train_x.float())
        train_error = test_objective(output_train, train_y.float()).item()

        output_test = self.model(test_x.float())
        generalization_error = test_objective(output_test, test_y.float()).item()

        self.train_error = train_error
        self.test_error = generalization_error

#     data_handling.writer(model_params,set_size,train_type,exp_type,sampling_method,train_error,generalization_error,dim)