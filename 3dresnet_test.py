import os
import sys
import time
import warnings
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import pathlib
import cv2
import user_set
import CNN as cnn
from FacePose_pytorch.dectect import AntiSpoofPredict
from FacePose_pytorch.pfld.pfld import PFLDInference, AuxiliaryNet	
from FacePose_pytorch.compute import find_pose, get_num

from resnet_3d.opts import parse_opts
from resnet_3d.model import generate_model
from resnet_3d.mean import get_mean
from resnet_3d.classify import classify_video


warnings.filterwarnings('ignore')

class Qt(QWidget):
    def mv_Chooser(self):    
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "./","Mp4 (*.mp4)", options=opt)
	
        return fileUrl[0]

classes = cnn.read_classes('classes.txt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

	print(classes)
	qt_env = QApplication(sys.argv)
	process = Qt()
	fileUrl = process.mv_Chooser()
	print(fileUrl)
	if(fileUrl == ""):
		print("Without input file!!")
		sys.exit(0)

	full_clip = []
	# model = torch.load('./3drenet_model.pth')

#-----------------------------------------------------------------------------------------
	opt = parse_opts()
	opt.mean = get_mean()
	opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
	opt.sample_size = 112
	opt.sample_duration = 10
	opt.n_classes = 6

	model = generate_model(opt)
	print('loading model {}'.format(opt.model))
	model_data = torch.load(opt.model)
	assert opt.arch == model_data['arch']
	print("------------------------------------------------")
	model.load_state_dict(model_data['state_dict'],strict=False)
	model.eval()

#-----------------------------------------------------------------------------------------
	font = cv2.FONT_HERSHEY_SIMPLEX

	cap = cv2.VideoCapture(fileUrl)
	ret, frame = cap.read()
	frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
	height, width = frame.shape[:2]

	fps = cap.get(cv2.CAP_PROP_FPS)

	while(ret):
		ret, frame = cap.read()
		if(not ret):
			break
		frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

		full_clip.append(frame)
		if len(full_clip)>10:
			del full_clip[0]
			
		cv2.imshow("frame", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			break

	cap.release()