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

    def blur_detection(self, image: np.ndarray) -> float:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurriness = self.variance_of_laplacian(image=gray_image)
        return blurriness

    def blur_detection_each_face(self, image: np.ndarray, face_locations: list) -> list[float]:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurrinesses = []
        for (top, right, bottom, left) in face_locations:
            clipped_gray_image = gray_image[top:bottom, left:right]
            blurriness = self.variance_of_laplacian(image=clipped_gray_image)
            blurrinesses.append(blurriness)
        return blurrinesses

if __name__ == '__main__':
    from face_manipulation import FaceManipulationWithFaceRecognition

    # face_recognition
    face_recognitor = FaceManipulationWithFaceRecognition(ML_type='hog', verbose=True)
    known_person_path = '/home/oishi/still_image_selector/Portfolio/Sample/'
    for name in os.listdir(known_person_path):
        face_recognitor.append_known_person(
            images = [known_person_path + '/' + name + '/' + image for image in os.listdir(known_person_path + name)],
            name = name,
        )

    # blur detection
    blur_detector = BlurDetection(verbose=True)

    # images to be handled
    directory_path = '/home/oishi/still_image_selector/samples/Sample/'
    image_paths = glob.glob(os.path.join(directory_path, '14*.png'))
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        face_locations, _, _ = face_recognitor.get_face_information(image=image)
        face_blurrinesses = blur_detector.blur_detection_each_face(image=image, face_locations=face_locations)
        logger.info(face_blurrinesses)
