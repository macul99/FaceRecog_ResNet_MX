from os import path

#BASE_PATH = '/media/macul/hdd/MsCelebV1-Faces-Cropped_112x112'
#BASE_PATH = '/media/macul/black/face_database_raw_data/mscelb_from_insightface'
BASE_PATH = '/media/macul/black/face_database_raw_data/deepglint_112x112'

#IMAGES_PATH = path.sep.join([BASE_PATH, "Data"])
IMAGES_PATH = '/media/macul/hdd/deepglint_112x112'
LANDMARK_PATH = path.sep.join([BASE_PATH, "landmark.txt"])
LABEL_PATH = path.sep.join([BASE_PATH, "label.txt"])
DATASET_MEAN = path.sep.join([BASE_PATH, "mean.json"])

#NUM_CLASSES = 85717
NUM_CLASSES = 180844
NUM_TEST_IMAGES = 2

#MX_OUT_PATH = '/media/macul/black/face_database_raw_data/mscelb_from_insightface'
MX_OUT_PATH = '/media/macul/black/face_database_raw_data/deepglint_112x112'
TRAIN_MX_LIST = path.sep.join([MX_OUT_PATH, "lists/train.lst"])
VAL_MX_LIST = path.sep.join([MX_OUT_PATH, "lists/val.lst"])
TEST_MX_LIST = path.sep.join([MX_OUT_PATH, "lists/test.lst"])

TRAIN_MX_REC = path.sep.join([MX_OUT_PATH, "rec/train.rec"])
VAL_MX_REC = path.sep.join([MX_OUT_PATH, "rec/val.rec"])
TEST_MX_REC = path.sep.join([MX_OUT_PATH, "rec/test.rec"])

LANDMARK_TH = 0.0 # only landmark score higher than this one is considered
LANDMARK_FLAG = True


DELIMITER = ' '

