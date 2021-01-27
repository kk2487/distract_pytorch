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
warnings.filterwarnings('ignore')


# for headpose model

classes = cnn.read_classes('classes.txt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
headpose_model = './FacePose_pytorch/checkpoint/snapshot/checkpoint.pth.tar'

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

def headpose_status(yaw, pitch, roll):

	up_down = ''
	left_right = ''
	tilt = ''

	if(yaw > user_set.H_R):
		left_right = 'right'
	elif(yaw < user_set.H_L):
		left_right = 'left'
	else:
		left_right = 'normal'

	if(pitch > user_set.H_D):
		up_down = 'down'
	elif(pitch < user_set.H_U):
		up_down = 'up'
	else:
		up_down = 'normal'

	if(roll > user_set.T_L):
		tilt = 'left'
	elif(roll < user_set.T_R):
		tilt = 'right'
	else:
		tilt = 'normal'

	return left_right, up_down, tilt

count = 0

yaw_sum = np.zeros(3)
yaw_count = np.zeros(3)
pitch_sum = np.zeros(3)
pitch_count = np.zeros(3)
roll_sum = np.zeros(3)
roll_count = np.zeros(3)

deg_past = np.zeros(3)

def headpose_series(yaw, pitch, roll):

	global yaw_sum, yaw_count, pitch_sum, pitch_count, roll_sum, roll_count
	if(abs(yaw - deg_past[0])<8):
		#yaw
		if(yaw>user_set.H_R):
			yaw_sum[0] = yaw_sum[0] + yaw
			yaw_count[0] = yaw_count[0] +1
		elif(yaw<user_set.H_L):
			yaw_sum[2] = yaw_sum[2] + yaw
			yaw_count[2] = yaw_count[2] +1
		deg_past[0] = yaw
	if(abs(pitch - deg_past[1])<8):	
		#pitch
		if(pitch>user_set.H_D):
			pitch_sum[0] = pitch_sum[0] + pitch
			pitch_count[0] = pitch_count[0] +1
		elif(pitch<user_set.H_U):
			pitch_sum[2] = pitch_sum[2] + pitch
			pitch_count[2] = pitch_count[2] +1
		deg_past[1] = pitch
	if(abs(roll - deg_past[2])<8):
		#roll
		if(roll>user_set.T_L):
			roll_sum[0] = roll_sum[0] + roll
			roll_count[0] = roll_count[0] +1
		elif(roll<user_set.T_R):
			roll_sum[2] = roll_sum[2] + roll
			roll_count[2] = roll_count[2] +1
		deg_past[2] = roll

def headpose_output():

	left_right = ''
	up_down = ''
	tilt = ''
	if(yaw_count[0] > yaw_count[2] and yaw_sum[0]/yaw_count[0] >10):
		left_right = "right"
	elif(yaw_count[0] < yaw_count[2] and yaw_sum[2]/yaw_count[2] < -10):
		left_right = "left"
	else:
		left_right = "normal"

	if(pitch_count[0] > pitch_count[2] and pitch_sum[0]/pitch_count[0] >10):
		up_down = "down"
	elif(pitch_count[0] < pitch_count[2] and pitch_sum[2]/pitch_count[2] < -10):
		up_down = "up"
	else:
		up_down = "normal"

	if(roll_count[0] > roll_count[2] and roll_sum[0]/roll_count[0] >10):
		tilt = "left"
	elif(roll_count[0] < roll_count[2] and roll_sum[2]/roll_count[2] < -10):
		tilt = "right"
	else:
		tilt = "normal"

	return left_right, up_down, tilt

def dis_head(dis_status, lr, ud, ti):

	score = 0
	score_d = 0
	score_lr = 0
	score_ud = 0
	score_ti = 0
	if(dis_status != 'safe'):
		score_d = 40

	if(score_lr != 'normal' and (yaw_count[0] > 7 or yaw_count[2] > 7)):
		score_lr = 40
	elif(score_lr != 'normal' and (yaw_count[0] > 3 or yaw_count[2] > 3)):
		score_lr = 30
	elif(score_lr != 'normal' and (yaw_count[0] > 1 or yaw_count[2] > 1)):
		score_lr = 10

	if(score_ud != 'normal' and (pitch_count[0] > 7 or pitch_count[2] > 7)):
		score_ud = 40
	elif(score_ud != 'normal' and (pitch_count[0] > 3 or pitch_count[2] > 3)):
		score_ud = 30
	elif(score_ud != 'normal' and (pitch_count[0] > 1 or pitch_count[2] > 1)):
		score_ud = 10

	if(score_ti != 'normal' and (roll_count[0] > 7 or roll_count[2] > 7)):
		score_ti = 30
	elif(score_ti != 'normal' and (roll_count[0] > 3 or roll_count[2] > 3)):
		score_ti = 20
	elif(score_ti != 'normal' and (roll_count[0] > 1 or roll_count[2] > 1)):
		score_ti = 10

	score = score_d + np.sqrt(score_lr*score_lr + score_ud*score_ud) + score_ti
	return score


class Qt(QWidget):
    def mv_Chooser(self):    
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "./","Mp4 (*.mp4)", options=opt)
	
        return fileUrl[0]

if __name__ == '__main__':

	left_right = ""
	up_down = "" 
	tilt = ""
	result = ""
	distract_score = 0
	output_check = np.zeros(len(classes))
	qt_env = QApplication(sys.argv)
	process = Qt()
	fileUrl = process.mv_Chooser()
	print(fileUrl)
	if(fileUrl == ""):
		print("Without input file!!")
		sys.exit(0)
	#model for face detect
	face_model = AntiSpoofPredict(0)

	#model for landmarks
	checkpoint_h = torch.load(headpose_model, map_location=device)
	plfd_backbone = PFLDInference().to(device)
	plfd_backbone.load_state_dict(checkpoint_h['plfd_backbone'])
	plfd_backbone.eval()
	plfd_backbone = plfd_backbone.to(device)
	headpose_transformer = transforms.Compose([transforms.ToTensor()])

	#model for distract 
	checkpoint_d=torch.load(user_set.model_path)
	model=cnn.ConvNet(num_classes=6).to(device)
	model.load_state_dict(checkpoint_d)
	model.eval()
	distract_transformer=transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
	transforms.Normalize([0.485,0.456,0.406], # 0-1 to [-1,1] , formula (x-mean)/std
						[0.229,0.224,0.255])
	])


	font = cv2.FONT_HERSHEY_SIMPLEX

	cap = cv2.VideoCapture(fileUrl)
	ret, frame = cap.read()
	frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
	height, width = frame.shape[:2]

	fps = cap.get(cv2.CAP_PROP_FPS)
	videoWriter = cv2.VideoWriter("./result.avi",cv2.VideoWriter_fourcc('X','V','I','D'),fps,(width,height))

	n = 0

	while(ret):

		output_file = "result_"+str(n)+".jpg"
		f_start = time.time()
		ret, frame = cap.read()
		if(not ret):
			break
		frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
		draw_mat = frame.copy()

		# 尋找臉部範圍資訊
		start = time.time()
		image_bbox = face_model.get_bbox(frame)
		face_x1 = image_bbox[0]
		face_y1 = image_bbox[1]
		face_x2 = image_bbox[0] + image_bbox[2]
		face_y2 = image_bbox[1] + image_bbox[3]
		face_w = face_x2 - face_x1
		face_h = face_y2 - face_y1
		f_end = time.time()

		#尋找特徵點
		l_start = time.time()
		crop_x1, crop_x2, crop_y1, crop_y2, dx, dy, edx, edy = crop_range(face_x1, face_x2, face_y1, face_y2, face_w, face_h)
		
		cropped = frame[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
		if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
			cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
		ratio_w = face_w / 112
		ratio_h = face_h / 112

		cropped = cv2.resize(cropped, (112, 112))
		face_input = cropped.copy()
		face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
		face_input = headpose_transformer(face_input).unsqueeze(0).to(device)

		_, landmarks = plfd_backbone(face_input)
		pre_landmark = landmarks[0]
		pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]
		l_end = time.time()

		#頭部姿態
		h_start = time.time()
		point_dict = {}
		i = 0
		for (x,y) in pre_landmark.astype(np.float32):
			point_dict[f'{i}'] = [x,y]
			#cv2.circle(draw_mat,(int(face_x1 + x * ratio_w),int(face_y1 + y * ratio_h)), 2, (255, 0, 0), -1)
			i += 1

		#cv2.circle(draw_mat,(int(face_x1 + get_num(point_dict, 1, 0) * ratio_w),int(face_y1 + get_num(point_dict, 1, 1) * ratio_h)), 2, (255, 0, 0), -1)
		#cv2.circle(draw_mat,(int(face_x1 + get_num(point_dict, 31, 0) * ratio_w),int(face_y1 + get_num(point_dict, 31, 1) * ratio_h)), 2, (255, 0, 0), -1)
		#cv2.circle(draw_mat,(int(face_x1 + get_num(point_dict, 51, 0) * ratio_w),int(face_y1 + get_num(point_dict, 51, 1) * ratio_h)), 2, (255, 0, 0), -1)
		#計算各軸角度
		#計算各軸角度
		#計算各軸角度
		yaw, pitch, roll = find_pose(point_dict)
		#left_right, up_down, tilt = headpose_status(yaw, pitch, roll)

		#cv2.putText(draw_mat,f"LEFT_RIGHT: {left_right} ({yaw})",(280,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.3,(0,255,0),2)
		#cv2.putText(draw_mat,f"UP_DOWN: {up_down} ({pitch})",(280,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.3,(0,255,0),2)
		#cv2.putText(draw_mat,f"TILT: {tilt} ({roll})",(280,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.3,(0,255,0),2)

		h_end = time.time()
		# 分心偵測部分
		d_start = time.time()
		# 框出臉部位置
		cv2.rectangle(draw_mat, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 255), 2, cv2.LINE_AA) 
		pre_src = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
		pre_src = cv2.resize(pre_src,(256,256))
		img=Image.fromarray(np.uint8(pre_src))
		img = distract_transformer(img).unsqueeze(0).to(device)
		
		out=model(img)
		output=classes[out.argmax()]

		d_end = time.time()
		end = time.time()
		#cv2.putText(draw_mat,output,(15,50), font, 1.4,(0,0,255),3,cv2.LINE_AA)
		#cv2.putText(draw_mat,str(int(1/(end-start))),(15,100), font, 1.4,(0,0,255),3,cv2.LINE_AA)

		if(count < 8):
			headpose_series(yaw, pitch, roll)
			output_check[out.argmax()] = output_check[out.argmax()] + 1
			count = count +1
			
			
		else:
			distract_score = 0
			left_right, up_down, tilt = headpose_output()
			count = 0
			result = str(classes[output_check.argmax()])
			distract_score = dis_head(result, left_right, up_down, tilt)
			output_check = np.zeros(len(classes))
			yaw_sum = np.zeros(3)
			yaw_count = np.zeros(3)
			pitch_sum = np.zeros(3)
			pitch_count = np.zeros(3)
			roll_sum = np.zeros(3)
			roll_count = np.zeros(3)
			
			
		cv2.putText(draw_mat,f"LEFT_RIGHT: {left_right}",(280,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.3,(0,255,0),2)
		cv2.putText(draw_mat,f"UP_DOWN: {up_down}",(280,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.3,(0,255,0),2)
		cv2.putText(draw_mat,f"TILT: {tilt}",(280,150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.3,(0,255,0),2)
		cv2.putText(draw_mat,result,(15,50), font, 1.4,(0,0,255),3,cv2.LINE_AA)
		if(distract_score >= 30):
			cv2.putText(draw_mat,"dangerous !!! ",(100,250), font, 2.8,(120,0,255),3,cv2.LINE_AA)
			
		#cropped = cv2.resize(cropped, (360, 360))
		draw_mat = cv2.resize(draw_mat,(360,640))

		cv2.imshow("draw_mat", draw_mat)

		if(n%4 == 0):
			cv2.imwrite(output_file, draw_mat)
		n = n+1

		"""
		print("total : ", end - start, int(1/( end - start)))
		print("face_detect : ", f_end - f_start)
		print("landmarks_detect : ", l_end - l_start)
		print("headpose_detect : ", h_end - h_start)
		print("distract_detect : ", d_end - d_start)
		print("------------------------------------")
		"""
		
		videoWriter.write(draw_mat)
		#cv2.imshow("cropped", cropped)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			break

	videoWriter.release()

	cap.release()