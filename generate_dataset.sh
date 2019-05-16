LIB=~/my_libs
SRC=/media/macul/black/face_database_raw_data/faces_emore/train.rec
DST=./mscelb_from_insightface
MX_ENV=~/mx_venv

# the following command need mxnet env
source $MX_ENV/bin/activate
python -m $LIB/mklib/nn/mxhelper/mxrec2img -s $SRC -d $DST/Data -f rec2img_insightface

python -m ./helper/process_label -s $DST/Data -d $DST/label.txt

# the following command need caffe support
deactivate
python -m ./helper/process_landmark -s $DST/Data -d $DST/landmark.txt

# config dataset_config.py
source $MX_ENV/bin/activate
python -m ./helper/process_lists

# generate rec files
python -m $MX_ENV/lib/python2.7/site-packages/mxnet/tools/im2rec $DST/lists/ $DST/Data --quality 100 --num-thread 16 --pack-label

# mv idx and rec files
mkdir $DST/rec
mv $DST/lists/*.idx $DST/rec
mv $DST/lists/*.rec $DST/rec
