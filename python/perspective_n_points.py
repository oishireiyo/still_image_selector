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
import numpy
import cv2
import numpy as np

# Handmade modules
from face_manipulation import FaceManipulationWithFaceRecognition, FaceManipulationWithMediaPipe, FaceLandmarksCalibration
import colors

class PerspectiveNPoints(object):
    def __init__(self, width: int=800, height: int=600,
                 calibrated_correspondences=None,
                 landmark_indices: list=[
                     #('chin', 0),
                     ('chin', 8),
                     #('chin', 16), # 輪郭
                     ('nose_bridge', 0), ('nose_bridge', 3), # 鼻筋
                     ('top_lip', 0), ('bottom_lip', 0),      # 唇の端
                     ('left_eye', 0), ('right_eye', 3),      # 目の端
                 ],
                 facial_point_file_path='canonical_face_model/canonical_face_model.obj',
                 calibration_image_path='../images/model10211041_TP_V4.jpg',
                 verbose: bool=True
    ):
        # 入力画像の特徴
        self.width = width
        self.height = height

        # 顔の情報を記録するオブジェクト
        self.facial_point_file = facial_point_file_path
        self.facial_points_3d = []
        self.facial_points_2d = []

        # perspective-n-points実行のためのパラメータ
        self.landmark_indices = landmark_indices
        self.camera_matrix = np.array([
            (width,     0,  width / 2),
            (    0, width, height / 2),
            (    0,      0,         1),
        ], dtype=np.float32)
        self.distortion_matrix = np.zeros((4, 1))

        # media-pipeとface_recognitionの関係性
        self.correspondences = calibrated_correspondences
        if self.correspondences is None:
            calibration_image = cv2.imread(calibration_image_path)
            self.correspondences = FaceLandmarksCalibration().compare(image=calibration_image, output_name='calibration.png')

        if verbose: self._print_information()

    def _print_information(self) -> None:
        logger.info('-' * 50)
        logger.info('-' * 50)

    def parse_canonical_facial_points_3d(self):
        logger.info('一般的な顔の3次元情報を与える、ただし西欧人のモデルなので東洋人に適応できるかは不明。多分できる。')
        with open(self.facial_point_file, mode='r') as f:
            lines = f.readlines()
            for key, index in self.landmark_indices:
                line_number = self.correspondences[key][index] # landmarkはindex=0からスタートしていることに注意
                elements = lines[line_number].split()
                self.facial_points_3d.append(
                    (float(elements[1]), float(elements[2]), float(elements[3].replace('\n', ''))))
        self.facial_points_3d = np.array(self.facial_points_3d, dtype=np.float32)

    def get_landmark_indices(self):
        return self.landmark_indices

    def get_correspondences(self):
        return self.correspondences

    def set_camera_matrix(self, width: int, height: int):
        self.camera_matrix = np.array([
            (width,     0,  width / 2),
            (    0, width, height / 2),
            (    0,      0,         1),
        ], dtype=np.float32)

    def parse_detected_facial_points_2d(self, landmarks):
        self.facial_points_2d = []
        for key, index in self.landmark_indices:
            self.facial_points_2d.append(landmarks[key][index])
        self.facial_points_2d = np.array(self.facial_points_2d, dtype=np.float32)

    def perspective_n_points(self):
        success, rotation, translation = cv2.solvePnP(self.facial_points_3d, self.facial_points_2d,
                                                      self.camera_matrix, self.distortion_matrix,
                                                      flags = cv2.SOLVEPNP_ITERATIVE) # SOLVEPNP_EPNP or SOLVEPNP_ITERATIVE
        return (success, rotation, translation)

    def project_facial_points_3d_to_2d(self):
        _, rotation, translation = self.perspective_n_points()
        projected_facial_points_2d, _ = cv2.projectPoints(self.facial_points_3d, rotation, translation,
                                                          self.camera_matrix, self.distortion_matrix)
        return (self.facial_points_3d, projected_facial_points_2d)

    def project_given_points_3d_to_2d(self, points_3d):
        _, rotation, translation = self.perspective_n_points()
        projected_facial_points_2d, _ = cv2.projectPoints(points_3d, rotation, translation,
                                                          self.camera_matrix, self.distortion_matrix)
        return (points_3d, projected_facial_points_2d)

    def get_roll_pitch_yaw(self):
        _, rotation, translation = self.perspective_n_points()
        rodrigues = cv2.Rodrigues(rotation)[0]
        proj_matrix = np.hstack((rodrigues, translation))
        eular_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eular_angles]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        roll = -math.degrees(math.asin(math.sin(roll)))

        return (pitch, yaw, roll)

    def get_roll_pitch_yaw_each_face(self, image, face_locations, face_landmarks):
        height, width, _ = image.shape
        self.set_camera_matrix(width=width, height=height)

        pitch_yaw_rolls = []
        for (top, right, bottom, left), landmarks in zip(face_locations, face_landmarks):
            clipped_image = image[top:bottom, left:right]
            self.parse_detected_facial_points_2d(landmarks=landmarks)
            pitch_yaw_roll = self.get_roll_pitch_yaw()
            pitch_yaw_rolls.append(pitch_yaw_roll)

        return pitch_yaw_rolls

if __name__ == '__main__':
    input_image = cv2.imread('../images/model10211041_TP_V4.jpg')
    height, width, _ = input_image.shape

    # face_recognitionを用いた顔検出
    face_recognitor = FaceManipulationWithFaceRecognition()
    _, landmarks_list, _ = face_recognitor.get_face_information(image=input_image)

    pnp = PerspectiveNPoints(width=width, height=height)
    pnp.parse_canonical_facial_points_3d()
    pnp.parse_detected_facial_points_2d(landmarks_list[0])

    _, points_2d = pnp.project_facial_points_3d_to_2d()
    for p in points_2d:
        cv2.drawMarker(input_image, position=(int(p[0][0]), int(p[0][1])), color=colors.COLOR_BLUE, markerType=cv2.MARKER_SQUARE)

    given_points_3d = np.array([
        (0, 0, 0),
        (10, 0, 0),
        (0, 10, 0),
        (0, 0, 10),
    ], dtype=np.float32)
    _, points_2d = pnp.project_given_points_3d_to_2d(points_3d=given_points_3d)

    cv2.arrowedLine(input_image,
                    pt1=(int(points_2d[0][0][0]), int(points_2d[0][0][1])),
                    pt2=(int(points_2d[1][0][0]), int(points_2d[1][0][1])),
                    color=colors.COLOR_RED, thickness=2)
    cv2.arrowedLine(input_image,
                    pt1=(int(points_2d[0][0][0]), int(points_2d[0][0][1])),
                    pt2=(int(points_2d[2][0][0]), int(points_2d[2][0][1])),
                    color=colors.COLOR_BLUE, thickness=2)
    cv2.arrowedLine(input_image,
                    pt1=(int(points_2d[0][0][0]), int(points_2d[0][0][1])),
                    pt2=(int(points_2d[3][0][0]), int(points_2d[3][0][1])),
                    color=colors.COLOR_GREEN, thickness=2)

    # media-pipeを用いた顔検出
    face_recognitor = FaceManipulationWithMediaPipe()
    results_with_mp = face_recognitor.get_face_information(input_image)
    landmarks_list = face_recognitor.get_face_landmarks(results=results_with_mp)

    landmark_indices = pnp.get_landmark_indices()
    correspondences = pnp.get_correspondences()

    for key, index in landmark_indices:
        id = correspondences[key][index]
        x, y = landmarks_list[0][id]
        cv2.drawMarker(input_image, position=(int(x * width), int(y * height)), color=colors.COLOR_BLACK, markerType=cv2.MARKER_STAR)

    cv2.imwrite('aho.jpg', input_image)

    pitch, yaw, roll = pnp.get_roll_pitch_yaw()
    cv2.putText(input_image,
                text='pitch, yaw, roll = %f, %f, %f' % (pitch, yaw, roll), org=(10, height-10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=colors.COLOR_BLACK)
    print(pitch, yaw, roll)
