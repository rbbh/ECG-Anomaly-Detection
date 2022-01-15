import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Metrics:

    def __init__(self, tp, tn, fp, fn, normal_vectors, abnormal_vectors, feature_type):
        self.__tp = tp
        self.__tn = tn
        self.__fp = fp
        self.__fn = fn
        self.__normal_vectors = normal_vectors
        self.__abnormal_vectors = abnormal_vectors
        self.__feature_type = feature_type

        self.__precision = self.compute_recall()
        self.__recall = self.compute_recall()
        self.__plot_feature_vectors()

    def compute_accuracy(self):
        return (self.__tp + self.__tn) / (self.__tp + self.__tn + self.__fp + self.__fn)

    def compute_precision(self):
        return self.__tp / (self.__tp + self.__fp)

    def compute_recall(self):
        return self.__tp / (self.__tp + self.__fn)

    def compute_f1_score(self):
        return 2 * self.__precision * self.__recall / (self.__precision + self.__recall)

    def __plot_feature_vectors(self):
        base_path = Path(f"outputs/plots/{self.__feature_type}")
        concatenated_vectors = np.concatenate((self.__normal_vectors, self.__abnormal_vectors), axis=0)
        tsne = TSNE(n_components=2, verbose=0, n_jobs=-1)
        if concatenated_vectors[0].shape[0] > 2000:
            pca = PCA(n_components=50)
            reduced_features_aux = pca.fit_transform(concatenated_vectors)
            reduced_features = tsne.fit_transform(reduced_features_aux)
        else:
            reduced_features = tsne.fit_transform(concatenated_vectors)

        plt.scatter(reduced_features[:len(self.__normal_vectors), 0], reduced_features[:len(self.__normal_vectors), 1])
        plt.scatter(reduced_features[len(self.__normal_vectors):, 0], reduced_features[len(self.__normal_vectors):, 1])
        plt.legend(['Normal Features', 'Abnormal Features'])
        plt.title('t-SNE')
        plt.savefig((base_path / "scatter_plot_dir_100").with_suffix(".png"))
        plt.show()
