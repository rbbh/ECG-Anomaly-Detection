import torch
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from source.metrics import Metrics
from source.preprocess import Preprocess


class Tester:
    """This class runs the inference on the chosen models using the test data.

    Attributes
    ----------
    _Tester.__test_loader : list
                            Organized test data.

    _Tester.__ml_model : str
                         Chosen ML-Model to use.

    _Tester.__feature_type : str
                             Type of feature used.

    _Tester.__device : str
                       Device to run the DL-Model.

    Methods
    -------
    _Tester.evaluate : Public
                       Runs the inference on the model.

    """

    def __init__(self, test_loader, ml_model, preprocess_obj: Preprocess, device):
        self.__test_loader = test_loader
        self.__ml_model = ml_model
        self.__feature_type = preprocess_obj.get_feature_type
        self.__device = device

    def evaluate(self, model):
        """Extracts the embedding feature vectors of the pre-trained encoder after running the inference with the
        features and feeds them to the chosen ML model in order to compute the metrics.

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
            for X, y in tqdm(self.__test_loader, desc="Testing"):
                feature_vector = model.encoder_forward(X.to(self.__device))
                if y == 1:
                    normal_vectors.append(X.cpu().numpy().flatten())
                else:
                    abnormal_vectors.append(X.cpu().numpy().flatten())

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

        metrics_obj = Metrics(TP, TN, FP, FN, normal_vectors, abnormal_vectors, self.__feature_type)

        print(f"Accuracy: {metrics_obj.compute_accuracy():.4f}")
        print(f"Precision: {metrics_obj.compute_precision():.4f}", )
        print(f"Recall: {metrics_obj.compute_recall():.4f}")
        print(f"F1-Score: {metrics_obj.compute_f1_score():.4f}")
