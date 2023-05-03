import numpy as np

class EarlyStop:
    def __init__(self, patience, loss=False):
        self.patience = patience
        self.best_value = np.inf if loss else -1
        self.best_epoch = 0
        self.loss = loss

    def step(self, current_value, current_epoch):
        print("Current:{} Best:{}".format(current_value, self.best_value))
        if self.loss:
            if current_value < self.best_value:
                self.best_value = current_value
                self.best_epoch = current_epoch
        else:
            if current_value > self.best_value:
                self.best_value = current_value
                self.best_epoch = current_epoch

    def stop_training(self, current_epoch) -> bool:
        return current_epoch - self.best_epoch > self.patience