import numpy as np
import matplotlib.pyplot as plt
from data_classes.scenario import Scenario

G_FORCE = 9.81
EPSILON: float = 0.1
STEP_SIZE: float = 0.00125

class ProjectileMethod:
    def __init__(self, y: np.ndarray):
        self.y_init = y
        self.calc_params()
        
        self.curr_x = np.array([0.0])
        self.curr_height = np.array([self.init_height])
        self.curr_x_velocity = np.array([self.init_x_velocity])
        self.curr_h_velocity = np.array([self.init_h_velocity])

        while self.curr_height[-1] > 0:
            self.ode_step()

    def G_transform(self):
        return 2 * self.y_init - 1

    def calc_params(self):
        G_y = self.G_transform()

        self.density = 1.225 * (1 + EPSILON * G_y[0])
        self.radius = 0.23 * (1 + EPSILON * G_y[1])
        self.drag_coeff = 0.1 * (1 + EPSILON * G_y[2])
        self.mass = 0.145 * (1 + EPSILON * G_y[3])
        self.init_height = 1 + EPSILON * G_y[4]
        self.alpha = 30 * (1 + EPSILON * G_y[5])
        
        init_velocity = 25 * (1 + EPSILON * G_y[6])
        self.init_x_velocity = init_velocity * np.cos(np.radians(self.alpha))  # Use radians for angle
        self.init_h_velocity = init_velocity * np.sin(np.radians(self.alpha))

    def calc_drag_force(self) -> float:
        velocity_norm_squared = np.linalg.norm(self.curr_velocity_vec())**2
        area = np.pi * self.radius**2
        drag_force = (0.5 * self.density * self.drag_coeff * area * velocity_norm_squared) / self.mass
        return drag_force

    def ode_step(self):
        drag_force = self.calc_drag_force()

        self.curr_x = np.append(self.curr_x, self.curr_x[-1] + self.curr_x_velocity[-1] * STEP_SIZE)
        self.curr_height = np.append(self.curr_height, self.curr_height[-1] + self.curr_h_velocity[-1] * STEP_SIZE)

        self.curr_x_velocity = np.append(self.curr_x_velocity, self.curr_x_velocity[-1] - STEP_SIZE * drag_force)
        self.curr_h_velocity = np.append(self.curr_h_velocity, self.curr_h_velocity[-1] - STEP_SIZE * G_FORCE)

    def plot(self):
        plt.plot(self.curr_x, self.curr_height, label="Projectile Path")
        plt.xlabel("Horizontal Distance (m)")
        plt.ylabel("Vertical Height (m)")
        plt.title("Projectile Motion")
        plt.legend()
        plt.grid()
        plt.show()

    def curr_velocity_vec(self):
        return np.array([self.curr_x_velocity[-1], self.curr_h_velocity[-1]])

    def get_total_distance(self):
        return float(self.curr_x[-1])


def projectile_motion_fun(y: np.ndarray):
    if isinstance(y, list) and all(isinstance(i, float) for i in y):
        y = np.array(y, dtype=float)

    assert y.shape[1] == 7, f"Expected 7 dimensions, but got {y.shape[0]} dimensions"
    if y.shape[0] == 7 and y.shape[1] != 7:
        y = y.T
    solutions = []
    for i, row in enumerate(y):
        method = ProjectileMethod(row)
        solutions.append(method.get_total_distance())
    return np.array(solutions)


def sum_sines_fun(x: np.ndarray):
    return np.sum(np.sin(x), axis=1)


def get_experiment_function(scenario: Scenario):
    match scenario:
        case Scenario.SUM_SINES:
            return sum_sines_fun
        case Scenario.PROJECTILE:
            return projectile_motion_fun
        case _:
         raise NotImplementedError(f"Scenario function for {scenario} is not implemented")