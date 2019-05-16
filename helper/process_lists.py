import dataset_config as config
import numpy as np
import progressbar
import json
import cv2
from os import listdir, makedirs
from os.path import join, exists
from prepare_dataset import PrepareDataset

print("[INFO] loading image paths...")

(R, G, B) = ([], [], [])

mch = PrepareDataset(config)

list_folder = config.TRAIN_MX_LIST[0:config.TRAIN_MX_LIST.rfind('/')]
rec_folder = config.TRAIN_MX_REC[0:config.TRAIN_MX_REC.rfind('/')]
if not exists(list_folder): makedirs(list_folder)
if not exists(rec_folder): makedirs(rec_folder)

if config.LANDMARK_FLAG:
	(trainPaths, trainLabels, trainLandmark, testPaths, testLabels, testLandmark, valPaths, valLabels, valLandmark) = mch.buildDataSetLandmark(delimiter=config.DELIMITER)

	datasets = [("train", trainPaths, trainLabels, trainLandmark, config.TRAIN_MX_LIST),
				("val", valPaths, valLabels, valLandmark, config.VAL_MX_LIST),
				("test", testPaths, testLabels, testLandmark, config.TEST_MX_LIST)]	

	for (dType, paths, labels, landmarks, outputPath) in datasets:
		print("[INFO] building {}...".format(outputPath))		
		f = open(outputPath, "w")

		# initialize the progress bar
		widgets = ["Building List: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
		pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

		for (i, (path, label, landmark)) in enumerate(zip(paths, labels, landmarks)):
			# write the image index, label, and output path to file
			row = "\t".join([str(i), str(label), str(landmark[0]), str(landmark[1]), str(landmark[2]), str(landmark[3]), str(landmark[4]), 
							 str(landmark[5]), str(landmark[6]), str(landmark[7]), str(landmark[8]), str(landmark[9]), path])
			f.write("{}\n".format(row))
			# if we are building the training dataset, then compute the
			# mean of each channel in the image, then update the
			# respective lists
			if dType == "train":
				image = cv2.imread(path)
				(b, g, r) = cv2.mean(image)[:3]
				R.append(r)
				G.append(g)
				B.append(b)
			# update the progress bar
			pbar.update(i)

		pbar.finish()
		f.close()
else:
	(trainPaths, trainLabels, testPaths, testLabels, valPaths, valLabels) = mch.buildDataSet()

	datasets = [("train", trainPaths, trainLabels, config.TRAIN_MX_LIST),
				("val", valPaths, valLabels, config.VAL_MX_LIST),
				("test", testPaths, testLabels, config.TEST_MX_LIST)]

	for (dType, paths, labels, outputPath) in datasets:
		print("[INFO] building {}...".format(outputPath))
		f = open(outputPath, "w")

		# initialize the progress bar
		widgets = ["Building List: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
		pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

		for (i, (path, label)) in enumerate(zip(paths, labels)):
			# write the image index, label, and output path to file
			row = "\t".join([str(i), str(label), path])
			f.write("{}\n".format(row))
			# if we are building the training dataset, then compute the
			# mean of each channel in the image, then update the
			# respective lists
			if dType == "train":
				image = cv2.imread(path)
				(b, g, r) = cv2.mean(image)[:3]
				R.append(r)
				G.append(g)
				B.append(b)
			# update the progress bar
			pbar.update(i)

		pbar.finish()
		f.close()

print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()

#cd ~/mx_venv/lib/python2.7/site-packages/mxnet/tools
#python -m im2rec /media/macul/black/face_database_raw_data/ms1m_crop_112x112/lists0/ /media/macul/hdd/MsCelebV1-Faces-Cropped_112x112/Data --quality 100 --num-thread 16 --pack-label


'''
import mxnet as mx

data_iter = mx.io.ImageRecordIter(
    path_imgrec='/media/macul/black/face_database_raw_data/ms1m_crop_112x112/lists0/val.rec',
    data_shape=(3, 112, 112), # output data shape. An 227x227 region will be cropped from the original image.
    batch_size=4, # number of samples per batch
    label_width=11 # resize the shorter edge to 256 before cropping
    # ... you can add more augmentation options as defined in ImageRecordIter.
    )

data_iter.reset()
batch = data_iter.next()
data = batch.data[0]
label = batch.label[0]
'''