# Standard modules
import os
import sys
import glob
import math
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

# Advanced modules
import cv2
import numpy as np

class BlurDetection(object):
    def __init__(self, verbose=True) -> None:
        if verbose: self._print_information()

    def _print_information(self) -> None:
        logger.info('-' * 50)
        logger.info('-' * 50)

    def variance_of_laplacian(self, image: np.ndarray) -> float:
        var = cv2.Laplacian(image, cv2.CV_64F).var()
        return var

    def blur_detection(self, image_name: str) -> float:
        image = cv2.imread(image_name)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        var = self.variance_of_laplacian(image = gray_image)
        logger.debug('Variance of image convoluted with the laplacian matrix: %.5f' % (var))

        return var

    def blur_detection_each_face(self, image_name: str, features: dict) -> dict:
        image = cv2.imread(image_name)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for name in features:
            top, bottom, left, right = features[name]['bbox']['tblr']
            gray_image_without_background = gray_image[top:bottom, left:right]
            var = self.variance_of_laplacian(image = gray_image_without_background)
            features[name]['laplacian'] = var
            logger.debug('Variance of image convoluted with the laplacian matrix for %s: %.5f' % (name, var))

        return features

if __name__ == '__main__':
    from face_manipulation import FaceRecognition

    face_recognitor = FaceRecognition(ML_type='hog', verbose=True)
    known_person_path = '/home/oishi/still_image_selector/Portfolio/初恋ざらり/'
    for name in os.listdir(known_person_path):
        face_recognitor.append_known_person(
            images = [known_person_path + '/' + name + '/' + image for image in os.listdir(known_person_path + name)],
            name = name,
        )

    directory_path = '/home/oishi/still_image_selector/samples/初恋ざらり/'
    image_paths = glob.glob(os.path.join(directory_path, '14*.png'))
    image_paths = sorted(image_paths)
    for image_path in tqdm(image_paths):
        features = face_recognitor.face_recognition(image_name=image_path)

        image = cv2.imread(image_path)
        blur_detector = BlurDetection(image=image, verbose=True)
        features = blur_detector.blur_detection_each_face(features=features)

        logger.info(features)