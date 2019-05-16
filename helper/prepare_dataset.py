from PIL import Image
import numpy as np
import os
from os import listdir, makedirs
from os.path import join, exists, isfile
import random
from imutils import paths

class PrepareDataset:
	def __init__(self, config):
		self.config = config
		self.labelMappings = self.buildClassLabels(delimiter=self.config.DELIMITER)
		self.landmark_ref = np.array([	[30.2946 + 8, 51.6963],
										[65.5318 + 8, 51.5014],
										[48.0252 + 8, 71.7366],
										[33.5493 + 8, 92.3655],
										[62.7299 + 8, 92.2041] ])
		self.img_w_h = 112.0

	def buildClassLabels(self, delimiter=','):
		rows = open(self.config.LABEL_PATH).read().strip().split("\n")
		labelMappings = {}
		for i, row in enumerate(rows):
			if i==0:
				continue
			(person, label) = row.split(delimiter)
			labelMappings[person] = int(label)-1
		return labelMappings

	def buildDataSet(self):
		train_paths = []
		train_labels = []
		test_paths = []
		test_labels = []
		val_paths = []
		val_labels = []

		folder_list = listdir(self.config.IMAGES_PATH)

		random.seed(33)
		for i, person in enumerate(folder_list):
			img_list = list(paths.list_images(join(self.config.IMAGES_PATH,person)))
			label = self.labelMappings[person]
			if label < self.config.NUM_CLASSES:
				random.shuffle(img_list)
				for j, img in enumerate(img_list):
					if j<self.config.NUM_TEST_IMAGES:
						test_labels.append(label)
						test_paths.append(img)
					else:
						train_labels.append(label)
						train_paths.append(img)
			else:
				for j, img in enumerate(img_list):
					val_labels.append(label)
					val_paths.append(img)
		return np.array(train_paths), np.array(train_labels), np.array(test_paths), np.array(test_labels), np.array(val_paths), np.array(val_labels)

	def buildDataSetLandmark(self, delimiter=','):
		landmark_flag = isfile(self.config.LANDMARK_PATH)
		assert landmark_flag, 'landmark file not found!!'

		train_paths = []
		train_labels = []
		train_landmark = []
		test_paths = []
		test_labels = []
		test_landmark = []
		val_paths = []
		val_labels = []
		val_landmark = []

		landmarks = {}
		if landmark_flag:
			with open(self.config.LANDMARK_PATH, 'r') as f:
				f.readline()
				for l in f.readlines():
					l = l.split(delimiter)
					landmarks[l[0]] = { 'score':float(l[1]),
										'x1': 	(float(l[2])-self.landmark_ref[0,0])/self.img_w_h,
										'y1': 	(float(l[3])-self.landmark_ref[0,1])/self.img_w_h,
										'x2': 	(float(l[4])-self.landmark_ref[1,0])/self.img_w_h,
										'y2': 	(float(l[5])-self.landmark_ref[1,1])/self.img_w_h,
										'x3': 	(float(l[6])-self.landmark_ref[2,0])/self.img_w_h,
										'y3': 	(float(l[7])-self.landmark_ref[2,1])/self.img_w_h,
										'x4': 	(float(l[8])-self.landmark_ref[3,0])/self.img_w_h,
										'y4': 	(float(l[9])-self.landmark_ref[3,1])/self.img_w_h,
										'x5': 	(float(l[10])-self.landmark_ref[4,0])/self.img_w_h,
										'y5': 	(float(l[11])-self.landmark_ref[4,1])/self.img_w_h,
										}
			

		folder_list = listdir(self.config.IMAGES_PATH)

		random.seed(33)
		for i, person in enumerate(folder_list):
			img_list = list(paths.list_images(join(self.config.IMAGES_PATH,person)))
			label = self.labelMappings[person]			
			if label < self.config.NUM_CLASSES:
				random.shuffle(img_list)
				for j, img in enumerate(img_list):
					key = join(*img.split('/')[-2:])
					if landmarks[key]['score'] < self.config.LANDMARK_TH:
						continue
					if j<self.config.NUM_TEST_IMAGES:
						test_labels.append(label)
						test_paths.append(img)
						test_landmark.append([	landmarks[key]['x1'],
												landmarks[key]['y1'],
												landmarks[key]['x2'],
												landmarks[key]['y2'],
												landmarks[key]['x3'],
												landmarks[key]['y3'],
												landmarks[key]['x4'],
												landmarks[key]['y4'],
												landmarks[key]['x5'],
												landmarks[key]['y5'],
												])
					else:
						train_labels.append(label)
						train_paths.append(img)
						train_landmark.append([	landmarks[key]['x1'],
												landmarks[key]['y1'],
												landmarks[key]['x2'],
												landmarks[key]['y2'],
												landmarks[key]['x3'],
												landmarks[key]['y3'],
												landmarks[key]['x4'],
												landmarks[key]['y4'],
												landmarks[key]['x5'],
												landmarks[key]['y5'],
												])
			else:
				for j, img in enumerate(img_list):
					key = join(*img.split('/')[-2:])
					if landmarks[key]['score'] < self.config.LANDMARK_TH:
						continue
					val_labels.append(label)					
					val_paths.append(img)
					val_landmark.append([	landmarks[key]['x1'],
											landmarks[key]['y1'],
											landmarks[key]['x2'],
											landmarks[key]['y2'],
											landmarks[key]['x3'],
											landmarks[key]['y3'],
											landmarks[key]['x4'],
											landmarks[key]['y4'],
											landmarks[key]['x5'],
											landmarks[key]['y5'],
											])
		return np.array(train_paths), np.array(train_labels), np.array(train_landmark), np.array(test_paths), np.array(test_labels), np.array(test_landmark), np.array(val_paths), np.array(val_labels), np.array(val_landmark)