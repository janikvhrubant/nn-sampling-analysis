# src/models.py
from torch import nn, optim, utils
from data_classes.architecture import NeuralNetworkArchitecture
from data_classes.training_config import BaseTrainingConfig, OptimizationMethod
from data_classes.training_data import TrainingData
import torch
from torch.utils.data import DataLoader, TensorDataset
from lion_pytorch import Lion
from datetime import datetime

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class SequentialNeuralNetwork:
    def __init__(self, net_arch: NeuralNetworkArchitecture):
        self.net_arch = net_arch

        # Prefer CUDA, then MPS, else CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

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
        self.model = self.model.to(self.device)

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
            case OptimizationMethod.LION:
                return Lion(
                    params=self.model.parameters(),
                    lr=config.LEARNING_RATE,
                    betas=config.BETAS,
                    weight_decay=config.REG_PARAM,
                    # use_triton=True
                )
            case _:
                raise NotImplementedError(f"Optimizer for {config.OPTIMIZER} is not implemented")

    def evaluate(self, data: TrainingData):
        test_objective = nn.L1Loss()

        train_x = data.train_x.to(self.device).float()
        train_y = data.train_y.to(self.device).float()
        test_x = data.test_x.to(self.device).float()
        test_y = data.test_y.to(self.device).float()

        output_train = self.model(train_x)
        train_error = test_objective(output_train, train_y).item()

        output_test = self.model(test_x)
        generalization_error = test_objective(output_test, test_y).item()

        self.train_error = train_error
        self.generalization_error = generalization_error

    def train(self, settings: BaseTrainingConfig, data: TrainingData):
        optimizer = self.create_optimizer(settings)
        criterion = nn.MSELoss()

        num_epochs = getattr(settings, "NUM_EPOCHS", getattr(settings, "num_epochs", 1))
        batch_size = getattr(settings, "BATCH_SIZE", getattr(settings, "batch_size", None))

        # Keep on CPU; only move inside the loop
        train_x_cpu = data.train_x.float()
        train_y_cpu = data.train_y.float()

        use_full_batch = (batch_size is None) or (batch_size >= len(train_x_cpu))

        self.model.train()
        t = datetime.now()

        if use_full_batch:
            tx = train_x_cpu.to(self.device, non_blocking=(self.device.type == "cuda"))
            ty = train_y_cpu.to(self.device, non_blocking=(self.device.type == "cuda"))
            for _ in range(num_epochs):
                optimizer.zero_grad(set_to_none=True)
                output = self.model(tx)
                loss = criterion(output, ty)
                loss.backward()
                optimizer.step()
            effective_bs = None
        else:
            dataset = TensorDataset(train_x_cpu, train_y_cpu)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,  # avoid nested multiprocessing on Windows
                pin_memory=(self.device.type == "cuda"),
                # pin_memory_device="cuda",  # optional for PyTorch >= 2.0
            )

            for _ in range(num_epochs):
                for bx_cpu, by_cpu in loader:
                    bx = bx_cpu.to(self.device, non_blocking=(self.device.type == "cuda")).float()
                    by = by_cpu.to(self.device, non_blocking=(self.device.type == "cuda")).float()
                    optimizer.zero_grad(set_to_none=True)
                    output = self.model(bx)
                    loss = criterion(output, by)
                    loss.backward()
                    optimizer.step()
            effective_bs = batch_size

        optim_time = datetime.now() - t

        self.evaluate(data)

        self.training_results = {
            "optimizer": settings.OPTIMIZER.value,
            "learning_rate": settings.LEARNING_RATE,
            "weight_decay": settings.REG_PARAM,
            "beta_1": getattr(settings, "BETAS", (None, None))[0],
            "beta_2": getattr(settings, "BETAS", (None, None))[1],
            "eps": getattr(settings, "EPS", None),
            "num_epochs": num_epochs,
            "batch_size": effective_bs,
            "layer_depth": self.net_arch.DEPTH,
            "n_layers": self.net_arch.NUM_HIDDEN_LAYERS,
            "activation_function": self.net_arch.ACTIVATION_FUNCTION.__name__,
            "train_error": self.train_error,
            "test_error": self.generalization_error,
            "training_set_size": len(data.train_x),
            "train_time": optim_time.total_seconds()
        }
