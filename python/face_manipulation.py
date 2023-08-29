# Standard python modules
import os
import sys
import time
import math
import pprint

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
import numpy as np
import cv2
import face_recognition
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Handmade modules
import colors

class FaceManipulationWithFaceRecognition(object):
    '''
    face_recognitionモジュールを用いた顔情報の抽出。

    face_recognitor = FaceManipulationWithFaceRecognition()
    face_recognitor.append_known_person(reference_images, name)
    landmarks_list = face_recognitor.get_face_information(target_image)
    target_image = face_recognitor.decorate_landmarks_image(target_image, landmarks_list)
    '''
    def __init__(self, ML_type: str='cnn', verbose: bool=True) -> None:
        # 顔検出の際に使用する機械学習アルゴリズム
        self.ML_type = ML_type

        # 既知の人物と画像を保存する
        self.known_face_encodings = []
        self.known_face_names = []

        self.verbose = verbose

    def append_known_person(self, images: list[str], name: str) -> None:
        for image in images:
            if not os.path.isfile(image):
                logger.critical('%sとして入力された画像ファイル\'%s\'は見当たりません。' % (name, image))
                sys.exit(1)

            logger.info('\'%s\'は%sの画像として入力されます。' % (image, name))
            _image = face_recognition.load_image_file(image)
            _image_encoding = face_recognition.face_encodings(_image)[0]

            self.known_face_encodings.append(_image_encoding)
            self.known_face_names.append(name)

    def get_face_locations_and_landmarks(self, image):
        face_locations = face_recognition.face_locations(image, model=self.ML_type)
        face_landmarks = face_recognition.face_landmarks(image, face_locations)

        return face_locations, face_landmarks

    def get_face_information(self, image):
        face_locations = face_recognition.face_locations(image, model=self.ML_type)
        face_landmarks = face_recognition.face_landmarks(image, face_locations)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        face_matches = []
        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

            best_match_index = np.argmin(distances)
            match_name = self.known_face_names[best_match_index]
            if not matches[best_match_index]:
                match_name = 'Unknown'
            face_matches.append(match_name)

        return face_locations, face_landmarks, face_matches

    def decorate_landmarks_image(self, image, landmarks_list):
        for landmarks in landmarks_list:
            cv2.drawMarker(image, landmarks['chin'][0], colors.COLOR_BLUE, cv2.MARKER_SQUARE)
            cv2.drawMarker(image, landmarks['chin'][8], colors.COLOR_BLUE, cv2.MARKER_SQUARE)
            cv2.drawMarker(image, landmarks['chin'][16], colors.COLOR_BLUE, cv2.MARKER_SQUARE)

            cv2.drawMarker(image, landmarks['nose_bridge'][0], colors.COLOR_GREEN, cv2.MARKER_SQUARE)
            cv2.drawMarker(image, landmarks['nose_bridge'][3], colors.COLOR_GREEN, cv2.MARKER_SQUARE)

            cv2.drawMarker(image, landmarks['top_lip'][0], colors.COLOR_RED, cv2.MARKER_SQUARE)
            cv2.drawMarker(image, landmarks['bottom_lip'][0], colors.COLOR_RED, cv2.MARKER_SQUARE)

            cv2.drawMarker(image, landmarks['left_eye'][0], colors.COLOR_YELLOW, cv2.MARKER_SQUARE)
            cv2.drawMarker(image, landmarks['right_eye'][3], colors.COLOR_YELLOW, cv2.MARKER_SQUARE)

        return image

class FaceManipulationWithMediaPipe(object):
    '''
    media_pipモジュールを用いた顔情報の抽出。

    face_recognitor = FaceManipulationWithMediaPipe()
    results = face_recognitor.get_face_information(target_image)
    landmarks_list = face_recognitor.get_face_landmarks(results)
    target_image = face_recognitor.decorate_landmarks_image(target_image, landmarks_list)
    '''
    def __init__(self, output_face_blendshapes: bool=False, output_facial_transformation_matrixes: bool=False) -> None:
        base_options = mp.tasks.BaseOptions(model_asset_path='trained_models/face_landmarker.task')
        face_landmarker = mp.tasks.vision.FaceLandmarker
        face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
        vision_running_mode = mp.tasks.vision.RunningMode

        options = face_landmarker_options(
            base_options=base_options,
            running_mode=vision_running_mode.IMAGE,
            output_face_blendshapes=output_face_blendshapes,
            output_facial_transformation_matrixes=output_facial_transformation_matrixes,
            num_faces=1,
        )

        self.landmarker = face_landmarker.create_from_options(options)

    def get_face_information(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = self.landmarker.detect(mp_image)

        return results

    def get_face_landmarks(self, results):
        face_landmarks_list = []
        for ls in results.face_landmarks:
            face_landmarks = []
            for l in ls:
                face_landmarks.append((l.x, l.y))
            face_landmarks_list.append(face_landmarks)

        return face_landmarks_list
    
    def decorate_landmarks_image(self, image, landmarks_list):
        height, width, _ = image.shape
        for landmarks in landmarks_list:
            for landmark in landmarks:
                landmark = (int(landmark[0] * width), int(landmark[1] * height))
                cv2.drawMarker(image, landmark, colors.COLOR_WHITE)

        return image

class FaceLandmarksCalibration(object):
    def __init__(self):
        self.face_recognitor_with_fr = FaceManipulationWithFaceRecognition()
        self.face_recognitor_with_mp = FaceManipulationWithMediaPipe()

    def get_face_information_with_fr(self, image):
        _, landmarks_list = self.face_recognitor_with_fr.get_face_locations_and_landmarks(image=image)

        return landmarks_list

    def get_face_information_with_mp(self, image):
        results = self.face_recognitor_with_mp.get_face_information(image=image)
        landmarks_list = self.face_recognitor_with_mp.get_face_landmarks(results=results)

        return landmarks_list

    def compare(self, image, output_name=None):
        landmarks_with_fr = self.get_face_information_with_fr(image)[0]
        landmarks_with_mp = self.get_face_information_with_mp(image)[0]

        correspondances = {}
        
        height, width, _ = image.shape
        logger.info('入力画像のサイズ(h, w) = (%d, %d)' % (height, width))
        for key in landmarks_with_fr:
            correspondances[key] = {}
            for i_landmark_with_fr, landmark_with_fr in enumerate(landmarks_with_fr[key]):
                x_fr, y_fr = landmark_with_fr
                minimum_euclidian_distance = 100
                minimum_euclidian_distance_index = 0
                for i_landmark_with_mp, landmark_with_mp in enumerate(landmarks_with_mp):
                    x_mp, y_mp = landmark_with_mp
                    euclidian_distance = math.sqrt(math.pow(x_fr - x_mp * width, 2) + math.pow(y_fr - y_mp * height, 2))
                    if minimum_euclidian_distance > euclidian_distance:
                        minimum_euclidian_distance = euclidian_distance
                        minimum_euclidian_distance_index = i_landmark_with_mp
                correspondances[key][i_landmark_with_fr] = minimum_euclidian_distance_index
                logger.info('face_recognitionで検出したランドマーク: %s - %d -> media-pipeで検出したランドマーク: %d (%f)' % (key, i_landmark_with_fr, minimum_euclidian_distance_index, minimum_euclidian_distance))

        if not output_name is None:
            image = self.face_recognitor_with_fr.decorate_landmarks_image(image=image, landmarks_list=[landmarks_with_fr])
            image = self.face_recognitor_with_mp.decorate_landmarks_image(image=image, landmarks_list=[landmarks_with_mp])
            cv2.imwrite(output_name, image)

        return correspondances

if __name__ == '__main__':
    image = cv2.imread('../images/model10211041_TP_V4.jpg')

    # face_recognitionを用いた人物検出
    face_recognitor = FaceManipulationWithFaceRecognition()
    face_recognitor.append_known_person(
        images=['../images/model10211041_TP_V4.jpg'], name='Suke',
    )
    face_recognitor.get_face_information(image=image)

    sys.exit(1)

    # face_recognitionとmedi-pipの橋渡し
    calibrator = FaceLandmarksCalibration()
    correspondances = calibrator.compare(image=image, output_name='hoge.jpg')

    pprint.pprint(correspondances)
