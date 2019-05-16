from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from PIL import Image
import numpy as np
import os
from os import listdir, makedirs
from os.path import join, exists
import random
import progressbar

class ProcessLabel():
    @staticmethod
    def parse_args(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-s", "--src", required=True, help="path to source folder")
        ap.add_argument("-d", "--dest", required=True, help="path to label.txt")
        ap.add_argument("-dlm", "--delimiter", type=int, default=' ', help="delimiter")
        args = vars(ap.parse_args())
        args.src = os.path.abspath(args.src)
        args.dest = os.path.abspath(args.dest)
        return args

    @staticmethod
    def genLabel(self, src, dest, delimiter=' '):
    	if not isdir(src):
            assert False, 'source folder not exist!!!'

        dest_folder = dest[0:dest.rfind('/')]
        if not isdir(dest_folder):
            mkdir(dest_folder)

    	folder_list = listdir(src)
		#folder_list.sort()
		random.seed(42)
		random.shuffle(folder_list)		

		widgets = ["Building List: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(folder_list), widgets=widgets).start()

		with open(dest,'w') as f:
			f.write('folder_name'+delimiter+'label\n')
			for i, person in enumerate(folder_list):
				#print(person, i)
				pbar.update(i)
				f.write(person + delimiter + str(i+1) + '\n')
		pbar.finish()


if __name__ == '__main__':
    args = ProcessLabel.parse_args()

    # python -m process_label -s '/media/macul/black/face_database_raw_data/mscelb_from_insightface/Data' -d '/media/macul/black/face_database_raw_data/mscelb_from_insightface/label.txt'
    ProcessLabel.genLabel(args.src, args.dest, args.delimiter)
 