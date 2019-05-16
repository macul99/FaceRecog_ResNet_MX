# generate landmark files of insightface dataset, using dan to get landmarks
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from PIL import Image
import numpy as np
import cv2
from mklib.dan_caffe import dan
import time
import os
from os import listdir, makedirs
from os.path import join, exists
from imutils import paths
import progressbar




class ProcessLandmark():
    @staticmethod
    def parse_args(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-s", "--src", required=True, help="path to source folder")
        ap.add_argument("-d", "--dest", required=True, help="path to landmark.txt")
        ap.add_argument("-dlm", "--delimiter", type=int, default=' ', help="delimiter")
        args = vars(ap.parse_args())
        args.src = os.path.abspath(args.src)
        args.dest = os.path.abspath(args.dest)
        return args

    @staticmethod
    def genLandmark(self, src, dest, delimiter=' '):
    	if not isdir(src):
            assert False, 'source folder not exist!!!'

        dest_folder = dest[0:dest.rfind('/')]
        if not isdir(dest_folder):
            mkdir(dest_folder)

        dan_det = dan()

        widgets = ["Building List: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(listdir(src)), widgets=widgets).start()

		with open(dest,'w') as f:
			f.write('image_path'+delimiter+'score'+delimiter+'x1'+delimiter+'y1'+delimiter+'x2'+delimiter+'y2'+delimiter+'x3'+delimiter+'y3'+delimiter+'x4'+delimiter+'y4'+delimiter+'x5'+delimiter+'y5\n')
			for i, person in enumerate(listdir(src)):
				#print(person, i)
				pbar.update(i)
				for j, img in enumerate(list(paths.list_images(join(src,person)))):
					#print(img, j)
					points, score = dan_det.detectLandmark(cv2.imread(img), num_points=5)
					#print(points)
					#print(score)
					f.write(join(person,img.split('/')[-1])
							+delimiter+str(score)
							+delimiter+str(points[0,0])
							+delimiter+str(points[0,1])
							+delimiter+str(points[1,0])
							+delimiter+str(points[1,1])
							+delimiter+str(points[2,0])
							+delimiter+str(points[2,1])
							+delimiter+str(points[3,0])
							+delimiter+str(points[3,1])
							+delimiter+str(points[4,0])
							+delimiter+str(points[4,1])
							+'\n')

		pbar.finish()


if __name__ == '__main__':
    args = ProcessLandmark.parse_args()

    # python -m process_landmark -s '/media/macul/black/face_database_raw_data/mscelb_from_insightface/Data' -d '/media/macul/black/face_database_raw_data/mscelb_from_insightface/landmark.txt'
    ProcessLandmark.genLandmark(args.src, args.dest, args.delimiter)
