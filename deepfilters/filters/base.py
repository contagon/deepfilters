from abc import ABC, abstractmethod

class BayesianFilter(ABC):

    @abstractmethod
    def predict(self, u):
        pass

    @abstractmethod
    def update(self, z):
        pass