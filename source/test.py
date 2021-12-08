import torch
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class Tester:
    def __init__(self, test_loader, ml_model, device):
        self.__test_loader = test_loader
        self.__ml_model = ml_model
        self.__device = device

    def evaluate(self, model):
        normal_vectors = []
        abnormal_vectors = []

        model.double()
        model.eval()
        with torch.no_grad():
            for X, y in self.__test_loader:
                feature_vector = model.encoder_forward(X.to(self.__device))
                if y == 1:
                    normal_vectors.append((feature_vector.cpu().numpy(), y.cpu().numpy()))
                else:
                    abnormal_vectors.append((feature_vector.cpu().numpy(), y.cpu().numpy()))

        if self.__ml_model == "oc-svm":
            ml_model = OneClassSVM()
