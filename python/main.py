# Standard modules
import os
import time
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
import colors
from face_manipulation import FaceManipulationWithFaceRecognition
from blur_detection import BlurDetection
from perspective_n_points import PerspectiveNPoints
from models import Extractor, XMeans, PrincipalComponentAnalysis

class StillImageSelectr():
    def __init__(self, title: str='Sample',
                 facial_point_file_path='canonical_face_model/canonical_face_model.obj',
                 calibration_image_path='../images/model10211041_TP_V4.jpg',
                 n_PCA_components: int=10, n_clusters_init=8, n_clusters_max: int=20,
                 output_file_name='information.csv',
                 verbose: bool=True,
                 ) -> None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.input_image_path = u'%s/../samples/%s/' % (base_path, title)
        self.portfolio_path = u'%s/../Portfolio/%s/' % (base_path, title)

        # Image feature extractor
        self.extractor = Extractor(verbose=verbose)

        # PCA executor
        self.n_PCA_components = n_PCA_components
        self.pca = PrincipalComponentAnalysis(n_components=n_PCA_components, verbose=verbose)

        # X-means executor
        self.n_clusters_init = n_clusters_init
        self.n_clusters_max = n_clusters_max
        self.x_means = XMeans(n_clusters_max=n_clusters_max, n_clusters_init=n_clusters_init, verbose=verbose)

        # Face recognitor
        self.face_recognitor = FaceManipulationWithFaceRecognition(ML_type='cnn' if dlib.DLIB_USE_CUDA else 'hog',
                                                                   verbose=verbose)
        self._preparation_for_face_recognitor()

        # Blur detector
        self.blur_detector = BlurDetection(verbose=verbose)

        # Perspective-N-Points executor
        self.pnp = PerspectiveNPoints(facial_point_file_path=facial_point_file_path,
                                      calibration_image_path=calibration_image_path,
                                      verbose=verbose)
        self.pnp.parse_canonical_facial_points_3d()

        # Output file name
        os.makedirs('../outputs/%s' % (title), exist_ok=True)
        self.output_file_path='../outputs/%s' % (title)
        self.output_file_name='%s/%s' % (self.output_file_path, output_file_name)

    def _preparation_for_face_recognitor(self):
        for name in os.listdir(self.portfolio_path):
            self.face_recognitor.append_known_person(
                images=[f'{self.portfolio_path}/{name}/{image}' for image in os.listdir(f'{self.portfolio_path}/{name}')],
                name = name,
            )

    def main(self):
        logger.info('%sに存在する画像ファイルを取得' % (self.input_image_path))
        image_paths = glob.glob(os.path.join(self.input_image_path, '*.png'))
        image_paths = sorted(image_paths)
        images = [cv2.imread(image_path) for image_path in image_paths]

        logger.info('画像のクラスタリング')
        Xs = np.array([self.extractor.run(image) for image in images])
        Xs = self.pca.fit(Xs)
        cluster_ids = self.x_means.predict(Xs)

        logger.info('画像の特徴、画像のブレを検出')
        dump_str  = '# このファイルは%sにより自動生成されたものです。\n' % (__file__)
        dump_str += '# 入力されたパラメータは以下の通りです。\n'
        dump_str += '# n_PCA_components: %d\n' % (self.n_PCA_components)
        dump_str += '# n_clusters_init: %d, n_clusters_max: %d\n' % (self.n_clusters_init, self.n_clusters_max)
        dump_str += 'image_path,image_width,image_height,cluster_id,parson,' \
                    'bbox_top,bbox_right,bbox_bottom,bbox_left,blurriness,' \
                    'pitch,yaw,roll,area_chin,area_nose,area_top_lip,area_bottom_lip,' \
                    'area_oral_cavity,area_left_eye,area_right_eye\n'
        for i, (image_path, image, cluster_id) in enumerate(zip(image_paths, images, cluster_ids)):
            logger.info('画像\'%s\' (%d / %d) を処理中。。。' % (image_path, i+1, len(image_paths)))
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            face_locations, face_landmarks, face_matches = \
                self.face_recognitor.get_face_information(image=image)
            face_blurrinesses = \
                self.blur_detector.blur_detection_each_face(image=image, face_locations=face_locations)
            face_pitch_yaw_rolls = \
                self.pnp.get_roll_pitch_yaw_each_face(image=image, face_locations=face_locations, face_landmarks=face_landmarks)
            for face_location, face_landmark, face_match, face_blurriness, (pitch, yaw, roll) in \
                zip(face_locations, face_landmarks, face_matches, face_blurrinesses, face_pitch_yaw_rolls):
                top, right, bottom, left = face_location
                area_chin, area_nose, area_top_lip, area_bottom_lip, area_oral_cavity, area_left_eye, area_right_eye = \
                    self.face_recognitor.get_facial_expression(landmarks=face_landmark)

                dump_str += f'{image_path},{height},{width},{cluster_id},{face_match},' \
                            f'{top},{right},{bottom},{left},{face_blurriness},'\
                            f'{pitch},{yaw},{roll},{area_chin},{area_nose},{area_top_lip},{area_bottom_lip},'\
                            f'{area_oral_cavity},{area_left_eye},{area_right_eye}\n'

                self.decorate_image(image=image, location=face_location, landmarks=face_landmark,
                                    name=face_match, blurriness=face_blurriness,
                                    angles=(pitch, yaw, roll), cluster_id=cluster_id)
                cv2.imwrite('%s/%s' % (self.output_file_path, os.path.basename(image_path)), image)

        with open(self.output_file_name, mode='w') as f:
            f.write(dump_str)

    def decorate_image(self, image, location, landmarks, name, blurriness, angles, cluster_id):
        self.decorate_image_bbox(image=image, location=location)
        self.decorate_image_landmarks(image=image, landmarks=landmarks)
        self.decorate_image_name_blurriness(image=image, location=location, name=name, blurriness=blurriness)
        self.decorate_image_angles(image=image, location=location, angles=angles)
        self.decorate_image_cluster_id(image=image, cluster_id=cluster_id)

    def decorate_image_bbox(self, image, location):
        top, right, bottom, left = location
        cv2.rectangle(image, pt1=(left, top), pt2=(right, bottom), color=colors.COLOR_RED, thickness=2, lineType=cv2.LINE_8)

    def decorate_image_landmarks(self, image, landmarks):
        for key in landmarks:
            for p in landmarks[key]:
                cv2.drawMarker(image, p, colors.COLOR_BLUE, cv2.MARKER_SQUARE)

    def decorate_image_name_blurriness(self, image, location, name, blurriness):
        top, right, bottom, left = location
        cv2.putText(image, text='Name: %s, Blurriness: %.3f' % (name, blurriness), org=(left, top-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0,
                    color=colors.COLOR_YELLOW, thickness=2, lineType=cv2.LINE_8)

    def decorate_image_angles(self, image, location, angles):
        top, right, bottom, left = location
        pitch, yaw, roll = angles
        cv2.putText(image, text='Pitch: %.3f, Yaw: %.3f, Roll: %.3f' % (pitch, yaw, roll), org=(left, bottom),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0,
                    color=colors.COLOR_GREEN, thickness=2, lineType=cv2.LINE_8)

    def decorate_image_cluster_id(self, image, cluster_id):
        height, width, _ = image.shape
        cv2.putText(image, text='Cluster ID: %d' % (cluster_id), org=(10, height-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0,
                    color=colors.COLOR_BLACK, thickness=2, lineType=cv2.LINE_8)

if __name__ == '__main__':
    start_time = time.time()

    still_image_selector = StillImageSelectr()
    still_image_selector.main()

    end_time = time.time()
    logger.info('処理時間: %.4f 秒' % (end_time - start_time))