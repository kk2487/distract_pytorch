import os
import sys
import cv2 
import random
import shutil
import numpy as np
from tqdm import tqdm
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
from pathlib import Path   

class Qt(QWidget):
    def mv_Chooser(self):    
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "C:/Users/hongze/Desktop/rgb","Mp4 (*.mp4)", options=opt)
	
        return fileUrl[0]

mode = 0
if __name__ == '__main__':

	if(len(sys.argv) < 2):
		print("Error : Enter mode --gray or --rgb")
		exit(0)
	elif(sys.argv[1] == '--gray'):
		mode = 1
	elif(sys.argv[1] == '--rgb'):
		mode = 2

	qt_env = QApplication(sys.argv)
	process = Qt()
	fileUrl = process.mv_Chooser()
	print(fileUrl)


	filename = Path(fileUrl).stem
	print(filename)

	cap = cv2.VideoCapture(fileUrl)

	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	ret, frame = cap.read()
	#順時鐘轉90度
	frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
	height, width = frame.shape[:2]
	if(os.path.exists('./dataset')):
		shutil.rmtree('./dataset/')
	os.makedirs('./dataset/train')
	os.makedirs('./dataset/validation')
	i = 0
	if(mode == 1):
		for i in tqdm(range(length)):
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			if(random.randint(0,10) == 1):
				savefile = "./dataset/validation/"+filename + "_" + str(i) + ".jpg"
			else:
				savefile = "./dataset/train/"+filename + "_" + str(i) + ".jpg"
			gray = cv2.resize(gray, (256, 256) )
			cv2.imwrite(savefile, gray)
			ret, frame = cap.read()
			frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
			i = i+1
	elif(mode == 2):
		for i in tqdm(range(length)):
			if(random.randint(0,10) == 1):
				savefile = "./dataset/validation/"+filename + str(i) + ".jpg"
			else:
				savefile = "./dataset/train/"+filename + str(i) + ".jpg"
			frame = cv2.resize(frame, (256, 256) )
			cv2.imwrite(savefile, frame)
			ret, frame = cap.read()
			frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
			i = i+1
