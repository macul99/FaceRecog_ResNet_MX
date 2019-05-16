run generate_dataset.sh

to verify the rec file:

import mxnet as mx

data_iter = mx.io.ImageRecordIter(
    path_imgrec='/media/macul/black/face_database_raw_data/deepglint_112x112/rec/test.rec',
    data_shape=(3, 112, 112), # output data shape. An 227x227 region will be cropped from the original image.
    batch_size=4, # number of samples per batch
    label_width=11 # resize the shorter edge to 256 before cropping
    # ... you can add more augmentation options as defined in ImageRecordIter.
    )

data_iter.reset()
batch = data_iter.next()
data = batch.data[0]
label = batch.label[0]


#########################################
to verify the trained model accuracy:

import sys
import pickle
sys.path.append('/home/macul/libraries/mk_utils/mklib/utils/')
from buildFaceDB import buildFaceDB
from mxFeatureExtract import mxFeatureExtract
buildDB = buildFaceDB()
extractor=mxFeatureExtract('/media/macul/black/mxnet_training/r50/server_train20','train_20',58, outputs_name={'embedding':'embedding_output'},mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json')
buildDB.get_embedding_to_txt(extractor.getEmbedding, dir_path='/media/macul/black/face_database_raw_data/template_ali_112x112',dst_dir='/media/macul/black/face_database_raw_data/template_ali_112x112_server_train20_58')

# modify feature_dir_rootpath = "/media/macul/black/face_database_raw_data/template_ali_112x112_dgx_train6" in verification.py and identification_rank.py
cd ~/Projects/face-recognition-benchmarks-master/IJPB/code
source ~/tf_venv/bin/activate
python3 -m verification
python3 -m identification_rank


