# Standard modules
import os
import sys

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Advanced python modules
import numpy as np
import torch
import torchvision
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import sklearn
from sklearn.decomposition import PCA

class Extractor():
    def __init__(self):
        self.model = torchvision.models.mobilenet_v3_large(
            weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            width_mult=1.0,
            reduced_tail=False,
            dilated=False,
        )
        self.model.eval()

        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self, image: np.ndarray) -> np.ndarray:
        image = Image.fromarray(image)
        image = self.preprocess(image)
        with torch.inference_mode():
            image = image[None].to(self.device)
            output = self.model(image)

        return output[0].to('cpu').numpy()

class XMeans():
    def __init__(self, n_clusters_max: int, n_clusters_init: int=2, random_seed: int=83):
        self.n_clusters_max = n_clusters_max
        self.n_clusters_init = n_clusters_init
        self.random_seed = random_seed

    def predict(self, Xs: np.ndarray):
        xm_c = kmeans_plusplus_initializer(
            data=Xs,
            amount_centers=n_clusters_init,
            random_state=self.random_seed,
        ).initialize()

        xm_i = xmeans(
            data=Xs,
            initial_centers=xm_c,
            kmax=self.n_clusters_max,
            ccore=True,
            random_state=self.random_seed,
        )
        xm_i.process()

        classes = len(xm_i._xmeans__centers)
        predict = xm_i.predict(Xs)

        indices = []
        for i in range(classes):
            cluster_indices = np.where(predict == i)[0].tolist()
            indices.append(cluster_indices)

        return indices

class PrincipalComponentAnalysis():
    def __init__(self, n_components: int):
        self.pca = PCA(n_components=n_components)

    def fit(self, Xs: np.ndarray):
        Xs = self.pca.fit_transform(Xs)
        return Xs
