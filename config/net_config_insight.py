from os import path

BASE_PATH = './dataset' # soft link to '/media/macul/black/face_database_raw_data/ms1m_crop_112x112'

DATASET_MEAN = path.sep.join([BASE_PATH, "mean.json"])
TRAIN_MX_LIST = path.sep.join([BASE_PATH, "lists/train.lst"])
VAL_MX_LIST = path.sep.join([BASE_PATH, "lists/val.lst"])
TEST_MX_LIST = path.sep.join([BASE_PATH, "lists/test.lst"])

TRAIN_MX_REC = path.sep.join([BASE_PATH, "rec/train_insight.rec"])
VAL_MX_REC = path.sep.join([BASE_PATH, "rec/val.rec"])
TEST_MX_REC = path.sep.join([BASE_PATH, "rec/test.rec"])

NUM_CLASSES = 85742 #85742 insightface dataset
NUM_TEST_IMAGES = 2

MX_OUT_PATH = './output'


LANDMARK_TH = 0.9 # only landmark score higher than this one is considered

# model parameter
Embedding_Size = 512 # embedding size
Label_Name = ['softmax_label']
Label_Width = 1
Landmark_Num_Ponits = 5
Data_Shape = (3, 112, 112)

# argumentation parameter
Aug_rand_mirror = True

# optimizer parameter
Opt_name = 'SGD' # SGD, Adam
Opt_lr = 0.1
Opt_momentum = 0.9
Opt_weight_decay = 0.0005
Opt_rescale_grad = 1.0

# initializer parameter
Init_name = 'Xavier' # Xavier, He
Init_factor_type = 'out' # avg, in, out
Init_rnd_type = 'gaussian' # gaussian, uniform;  for 'Xavier' only
Init_magnitude = 2 # for 'Xavier' only
Init_slope = 0.25 # for 'He' only

# loss parameter
Arc_margin_angle = 0.5 # arc face margin_m
Arc_margin_scale = 64.0 # arc face margin_s
Arc_grad_scale = 1.0
Landmark_grad_scale = 0.0 # for landmark regression loss

# resnet parameter
Resnet_stages = (3, 4, 14, 3)
Resnet_filters = (64, 64, 128, 256, 512)
Resnet_input_layer = 'v2' # specify input layer version number
Resnet_residue_module = 'v3' # specify residue module version number
Resnet_use_se = False # use squeeze_excite network
Resnet_bottle_neck = False # use bottle_neck structure


# training parameters
BATCH_SIZE = 64
NUM_DEVICES = 2
NUM_EPOCH = 100
