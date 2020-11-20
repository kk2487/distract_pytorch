import os
import sys
import time
import numpy as np
from PIL import Image
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import pathlib
import cv2
import utils
import CNN as cnn
from FacePose_pytorch.dectect import AntiSpoofPredict

def get_num(point_dict, name, axis):
	num = point_dict.get(f'{name}')[axis]
	num = float(num)
	return num

# for distract model
def get_num(point_dict, name, axis):
	num = point_dict.get(f'{name}')[axis]
	num = float(num)
	return num
	
transformer=transforms.Compose([
	transforms.Resize((256,256)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
	transforms.Normalize([0.485,0.456,0.406], # 0-1 to [-1,1] , formula (x-mean)/std
						[0.229,0.224,0.255])
])
# for headpose model
transform = transforms.Compose([transforms.ToTensor()])
classes = cnn.read_classes('classes.txt')

def prediction(img, transformer, model):
	
	image_tensor=transformer(img).float()
	image_tensor=image_tensor.unsqueeze_(0)
	
	if torch.cuda.is_available():
		image_tensor.cuda()
		
	input=Variable(image_tensor)
	
	
	output=model(input)
	
	index=output.data.numpy().argmax()
	
	pred=classes[index]
	
	return pred

def crop_range(x1, x2, y1, y2, w, h):
	size = int(max([w, h]))
	cx = x1 + w/2
	cy = y1 + h/2
	x1 = int(cx - size/2)
	x2 = int(x1 + size)
	y1 = int(cy - size/2)
	y2 = int(y1 + size)

	dx = max(0, -x1)
	dy = max(0, -y1)
	x1 = max(0, x1)
	y1 = max(0, y1)

	edx = max(0, x2 - width)
	edy = max(0, y2 - height)
	x2 = min(width, x2)
	y2 = min(height, y2)
	return x1, x2, y1, y2, dx, dy, edx, edy

if __name__ == '__main__':
	
	fileUrl = './test5.mp4'
	font = cv2.FONT_HERSHEY_SIMPLEX

	cap = cv2.VideoCapture(fileUrl)
	ret, frame = cap.read()
	height, width = frame.shape[:2]
	
	model = cnn.load_model()

	face_model = AntiSpoofPredict(0)

	while(cap.isOpened()):

		
		start = time.time()
		ret, frame = cap.read()
		frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
		draw_mat = frame.copy()
		image_bbox = face_model.get_bbox(frame)
		face_x1 = image_bbox[0]
		face_y1 = image_bbox[1]
		face_x2 = image_bbox[0] + image_bbox[2]
		face_y2 = image_bbox[1] + image_bbox[3]
		face_w = face_x2 - face_x1
		face_h = face_y2 - face_y1

		crop_x1, crop_x2, crop_y1, crop_y2, dx, dy, edx, edy = crop_range(face_x1, face_x2, face_y1, face_y2, face_w, face_h)
		
		cropped = frame[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
		if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
			cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
		
		cropped = cv2.resize(cropped, (112, 112))


		cv2.rectangle(draw_mat, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 255), 2, cv2.LINE_AA) 
		pre_src = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
		pre_src = cv2.resize(pre_src,(256,256))
		img=Image.fromarray(np.uint8(pre_src))
		output=prediction(img, transformer, model)
		end = time.time()

		cv2.putText(draw_mat,output,(15,50), font, 1.4,(0,0,255),3,cv2.LINE_AA)
		cv2.putText(draw_mat,str(int(1/(end-start))),(15,100), font, 1.4,(0,0,255),3,cv2.LINE_AA)
		#frame = cv2.resize(frame,(360,640))
		cv2.imshow("draw_mat", draw_mat)
		cv2.imshow("cropped", cropped)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			break
