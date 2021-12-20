import torch
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class Tester:
    """
    Attributes
    ----------
    _Tester.__test_loader : list
                            Organized test data.

    _Tester.__ml_model : str
                         Chosen ML-Model to use.

    _Tester.__device : str
                       Device to run the DL-Model.

    Methods
    _______
    _Tester.evaluate : Runs the inference on the model.

    """

    def __init__(self, test_loader, ml_model, device):
        self.__test_loader = test_loader
        self.__ml_model = ml_model
        self.__device = device

    def evaluate(self, model):
        """Extracts the embedding feature vectors of the pre-trained encoder after running the inference with the
        scalograms and feeds them to the chosen ML model in order to compute the metrics.

        Parameters
        ----------
        model : object
                Pre-trained encoder.
        """
        normal_vectors = []
        abnormal_vectors = []

        ground_truth = []
        predictions = []

        TP = TN = FP = FN = 0

        if self.__ml_model == "oc-svm":
            ml_model = OneClassSVM(gamma=1e-3, nu=0.05)
        elif self.__ml_model == "isolation-forest":
            ml_model = IsolationForest(n_estimators=50, contamination=0.05)
        elif self.__ml_model == "local-outlier-factor":
            ml_model = LocalOutlierFactor(n_neighbors=10, contamination=0.05)

        model.double()
        model.eval()
        with torch.no_grad():
            for X, y in self.__test_loader:
                feature_vector = model.encoder_forward(X.to(self.__device))
                if y == 1:
                    normal_vectors.append(feature_vector.cpu().numpy())
                else:
                    abnormal_vectors.append(feature_vector.cpu().numpy())

                y_pred = ml_model.fit_predict(feature_vector.cpu().numpy())
                predictions.append(y_pred)
                ground_truth.append(y.numpy())

                if y_pred == -1 and y == -1:
                    TP += 1
                elif y_pred == 1 and y == 1:
                    TN += 1
                elif y_pred == -1 and y == 1:
                    FP += 1
                elif y_pred == 1 and y == -1:
                    FN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (precision + recall)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}", )
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
