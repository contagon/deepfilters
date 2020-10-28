from abc import ABC, abstractmethod

class BayesianFilter(ABC):

    @abstractmethod
    def update(self, u):
        pass

    @abstractmethod
    def predict(self, z):
        pass