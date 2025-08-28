# src/models.py
from torch import nn, optim, utils
from data_classes.architecture import NeuralNetworkArchitecture
from data_classes.training_config import BaseTrainingConfig, OptimizationMethod
from data_classes.training_data import TrainingData
import torch
from torch.utils.data import DataLoader, TensorDataset
from lion_pytorch import Lion
import time
from torch.utils.tensorboard import SummaryWriter
import optuna


activation_function_map = {
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
}

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
                activation_function_map[net_arch.ACTIVATION_FUNCTION]())
            for i in range(1, net_arch.NUM_HIDDEN_LAYERS):
                model = nn.Sequential(
                    model, nn.Linear(net_arch.DEPTH, net_arch.DEPTH),
                    nn.BatchNorm1d(num_features=net_arch.DEPTH),
                    activation_function_map[net_arch.ACTIVATION_FUNCTION]())
        else:
            model = nn.Sequential(
                nn.Linear(net_arch.INPUT_DIM, net_arch.DEPTH),
                activation_function_map[net_arch.ACTIVATION_FUNCTION]())
            for i in range(1, net_arch.NUM_HIDDEN_LAYERS):
                model = nn.Sequential(
                    model, nn.Linear(net_arch.DEPTH, net_arch.DEPTH),
                    activation_function_map[net_arch.ACTIVATION_FUNCTION]())

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
                    weight_decay=config.REG_PARAM
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

    def train(self, settings: BaseTrainingConfig, data: TrainingData, trial=None, report_every: int = 25, log_dir: str = None):
        optimizer = self.create_optimizer(settings)
        criterion = nn.MSELoss()

        num_epochs = getattr(settings, "NUM_EPOCHS", getattr(settings, "num_epochs", 1))
        batch_size = getattr(settings, "BATCH_SIZE", getattr(settings, "batch_size", None))

        train_x = data.train_x.float()
        train_y = data.train_y.float()
        test_x  = data.test_x.to(self.device).float()
        test_y  = data.test_y.to(self.device).float()

        batch_size = len(train_x) if batch_size is None or batch_size > len(train_x) else batch_size

        self.model.train()
        start_time = time.time()

        dataset = TensorDataset(train_x, train_y)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=(self.device.type == "cuda")
        )

        if log_dir is None:
            log_dir = f"data/{settings.OPTIMIZER.value}_logs_{time.strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(log_dir=log_dir)

        for epoch in range(num_epochs):
            epoch_start = time.time()
            running_loss = 0.0
            n_samples = 0

            for bx, by in loader:
                bx = bx.to(self.device, non_blocking=(self.device.type == "cuda")).float()
                by = by.to(self.device, non_blocking=(self.device.type == "cuda")).float()

                optimizer.zero_grad(set_to_none=True)
                output = self.model(bx)
                loss = criterion(output, by)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * bx.size(0)
                n_samples += bx.size(0)

            train_loss = running_loss / max(1, n_samples)

            self.model.eval()
            with torch.no_grad():
                test_output = self.model(test_x)
                test_loss = criterion(test_output, test_y).item()
            self.model.train()

            epoch_time = time.time() - epoch_start
            cumulative_time = time.time() - start_time

            writer.add_scalar("Training-Loss", train_loss, epoch)
            writer.add_scalar("Test-Data-Loss", test_loss, epoch)
            writer.add_scalar("Time per Epoch", epoch_time, epoch)
            writer.add_scalar("Total Time", cumulative_time, epoch)

            if trial is not None and ((epoch + 1) % report_every == 0 or epoch == num_epochs - 1):
                trial.report(test_loss, step=epoch)
                try:
                    if trial.should_prune():
                        writer.close()
                        raise optuna.TrialPruned()
                except ModuleNotFoundError:
                    pass

        optim_time = time.time() - start_time

        writer.close()
        self.evaluate(data)

        self.training_results = {
            "optimizer": settings.OPTIMIZER.value,
            "learning_rate": settings.LEARNING_RATE,
            "weight_decay": settings.REG_PARAM,
            "beta_1": getattr(settings, "BETAS", (None, None))[0],
            "beta_2": getattr(settings, "BETAS", (None, None))[1],
            "eps": getattr(settings, "EPS", None),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "layer_depth": self.net_arch.DEPTH,
            "n_layers": self.net_arch.NUM_HIDDEN_LAYERS,
            "activation_function": self.net_arch.ACTIVATION_FUNCTION,
            "train_error": self.train_error,
            "test_error": self.generalization_error,
            "training_set_size": len(data.train_x),
            "train_time": optim_time
        }
