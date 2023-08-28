# -*- coding: utf-8 -*- 

# Standard modules
import os
import glob
from tqdm import tqdm

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
import cv2
import dlib
import numpy as np

# Handmade modules
from face_manipulation import FaceRecognition
from blur_detection import BlurDetection
from models import Extractor, XMeans, PrincipalComponentAnalysis

class StillImageSelectr():
    def __init__(self, title: str='Sample', n_components: int=2, verbose: bool=True) -> None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.input_image_path = u'%s/../samples/%s/' % (base_path, title)
        self.portfolio_path = u'%s/../Portfolio/%s/' % (base_path, title)
        
        # Face recognitor
        self.face_recognitor = FaceRecognition(ML_type='cnn' if dlib.DLIB_USE_CUDA else 'hog', verbose=verbose)
        self._preparation_for_face_recognitor()

        # Blur detector
        self.blur_detector = BlurDetection(verbose=verbose)

        # Image feature extractor
        self.extractor = Extractor()

        # PCA runner
        self.pca = PrincipalComponentAnalysis(n_components=n_components)

        # X-means runner
        self.x_means = XMeans()

    def _preparation_for_face_recognitor(self):
        for name in os.listdir(self.portfolio_path):
            self.face_recognitor.append_known_person(
                images=[f'{self.portfolio_path}/{name}/{image}' for image in os.listdir(f'{self.portfolio_path}/{name}')],
                name = name,
            )

    def main(self):
        logger.info('画像の特徴、画像のブレを検出')
        image_paths = glob.glob(os.path.join(self.input_image_path, '*.png'))
        for image_path in tqdm(image_paths):
            features = self.face_recognitor.face_recognition(image_name=image_path)
            features = self.blur_detector.blur_detection_each_face(image_name=image_path, features=features)

        logger.info('画像のクラスタリング')
        images = [cv2.imread(image_path) for image_path in image_paths]
        Xs = np.array([self.extractor.run(image) for image in images])
        Xs = self.pca.fit(Xs)

        indices = self.x_means.predict(Xs)

        print(indices)

    def dump_results(self):
        pass

if __name__ == '__main__':
    still_image_selector = StillImageSelectr()
    still_image_selector.main()
