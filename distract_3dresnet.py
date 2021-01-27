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
import torch.nn.functional as F
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
import PIL
from PIL import Image, ImageOps

from resnet_3d_old.opts import parse_opts
from resnet_3d_old.mean import get_mean, get_std
from resnet_3d_old.model_c import generate_model
from resnet_3d_old.spatial_transforms_winbus import (
    Compose, Normalize, RandomHorizontalFlip, ToTensor, RandomVerticalFlip, 
    ColorAugment)
from resnet_3d_old.temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop
# from resnet_3d_old.dataset import get_test_set
warnings.filterwarnings('ignore')

class Qt(QWidget):
    def mv_Chooser(self):    
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "./","Mp4 (*.mp4)", options=opt)
	
        return fileUrl[0]

classes = cnn.read_classes('classes.txt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def letterbox(img, resize_size, mode='square'):
    shape = [img.size[1],img.size[0]]  # current shape [height, width]
    new_shape = resize_size
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    if mode == 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode == 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode == 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode == 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = img.resize(new_unpad,PIL.Image.ANTIALIAS)
    img = ImageOps.expand(img, border=(left,top,right,bottom), fill=(128,128,128))
    return img

class letter_img(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        return letterbox(img, self.size)
    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

def get_test_data(images, spatial_transform):
	images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
	# pre_src = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	clip = [Image.fromarray(img) for img in images]
	clip = [img.convert('RGB') for img in clip]
	spatial_transform.randomize_parameters()
	clip = [spatial_transform(img) for img in clip]
	clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
	clip = torch.stack((clip,), 0)
	'''
	test_loader = torch.utils.data.DataLoader(
            clip,
            batch_size=1,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
	'''
	return clip

def predict(model, test_data):

	inputs = Variable(test_data, volatile=True).cuda()
	outputs = model(inputs) 
	outputs = F.softmax(outputs)
	#print(classes[outputs.argmax()])
	return classes[outputs.argmax()]

if __name__ == '__main__':
	print(classes)
	qt_env = QApplication(sys.argv)
	process = Qt()
	fileUrl = process.mv_Chooser()
	print(fileUrl)
	if(fileUrl == ""):
		print("Without input file!!")
		sys.exit(0)

	left_right = ""
	up_down = "" 
	tilt = ""

	distract_output = ""
	full_clip = []

	distract_score = 0
	output_check = np.zeros(len(classes))

	font = cv2.FONT_HERSHEY_SIMPLEX
# ----------------------------------------------------
	# 3D_resnet for distract detection
	opt = parse_opts()
	opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
	opt.std = get_std(opt.norm_value)
	if opt.no_mean_norm and not opt.std_norm:
		norm_method = Normalize([0, 0, 0], [1, 1, 1])
	elif not opt.std_norm:
		print('mean:', opt.mean)
		norm_method = Normalize(opt.mean, [1, 1, 1])
	else:
		norm_method = Normalize(opt.mean, opt.std)

	opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

	model, parameters = generate_model(opt)
	checkpoint = torch.load(opt.resume_path)
	opt.arch == checkpoint['arch']
	model.load_state_dict(checkpoint['state_dict'])

	spatial_transform = Compose([
		letter_img(opt.sample_size),
		#letter_img(112),
		ToTensor(opt.norm_value), 
		norm_method
    ])
	temporal_transform = TemporalCenterCrop(opt.sample_duration)

# -------------------------------------------------------------------------------

	#model for face detect
	face_model = AntiSpoofPredict(0)

	#model for landmarks
	headpose_model = './FacePose_pytorch/checkpoint/snapshot/checkpoint.pth.tar'
	checkpoint_h = torch.load(headpose_model, map_location=device)
	plfd_backbone = PFLDInference().to(device)
	plfd_backbone.load_state_dict(checkpoint_h['plfd_backbone'])
	plfd_backbone.eval()
	plfd_backbone = plfd_backbone.to(device)
	headpose_transformer = transforms.Compose([transforms.ToTensor()])

	cap = cv2.VideoCapture(fileUrl)
	ret, frame = cap.read()
	frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
	height, width = frame.shape[:2]

	fps = cap.get(cv2.CAP_PROP_FPS)
	videoWriter = cv2.VideoWriter("./result.avi",cv2.VideoWriter_fourcc('X','V','I','D'),fps,(720,1280))

	model.eval()
	while(ret):
		start=time.time()
		ret, frame = cap.read()
		if(not ret):
			break

		frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
		draw_mat = frame.copy()

		# 尋找臉部範圍資訊
		image_bbox = face_model.get_bbox(frame)
		face_x1 = image_bbox[0]
		face_y1 = image_bbox[1]
		face_x2 = image_bbox[0] + image_bbox[2]
		face_y2 = image_bbox[1] + image_bbox[3]
		face_w = face_x2 - face_x1
		face_h = face_y2 - face_y1

		#尋找特徵點
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

		#頭部姿態
		h_start = time.time()
		point_dict = {}
		i = 0
		for (x,y) in pre_landmark.astype(np.float32):
			point_dict[f'{i}'] = [x,y]
			#cv2.circle(draw_mat,(int(face_x1 + x * ratio_w),int(face_y1 + y * ratio_h)), 2, (255, 0, 0), -1)
			i += 1

		yaw, pitch, roll = find_pose(point_dict)

		#分心狀態偵測
		frame = cv2.resize(frame, (224,224))
		#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		full_clip.append(frame)

		if len(full_clip) > 9:
			
			test_data = get_test_data(full_clip, spatial_transform)
			
			distract_output = predict(model, test_data)
			full_clip = []

		#頭部姿態綜合分析
		if(count < 9):
			headpose_series(yaw, pitch, roll)
			count = count +1

		else:
			distract_score = 0
			left_right, up_down, tilt = headpose_output()
			if(face_w < 20 or face_h < 20):
				left_right, up_down, tilt = "", "", ""
			count = 0
			distract_score = dis_head(distract_output, left_right, up_down, tilt)
			output_check = np.zeros(len(classes))
			yaw_sum = np.zeros(3)
			yaw_count = np.zeros(3)
			pitch_sum = np.zeros(3)
			pitch_count = np.zeros(3)
			roll_sum = np.zeros(3)
			roll_count = np.zeros(3)
			print(left_right, up_down, tilt)

		end=time.time()

		cv2.rectangle(draw_mat, (0, 0), (width, 150), (255, 255, 255), -1, cv2.LINE_AA)
		
		cv2.rectangle(draw_mat, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 255), 2, cv2.LINE_AA)

		cv2.putText(draw_mat,"Head Pose ",(200,35), font,1.1,(0,0,0),2) 
		cv2.putText(draw_mat,"LEFT_RIGHT",(200,70), font,0.8,(255,0,0),2) 
		cv2.putText(draw_mat,"UP_DOWN   ",(200,100), font,0.8,(255,0,0),2)
		cv2.putText(draw_mat,"TILT      ",(200,130), font,0.8,(255,0,0),2)

		cv2.putText(draw_mat, ": "+str(left_right), (360,70), font,0.8,(255,0,0),2)
		cv2.putText(draw_mat, ": "+str(up_down), (360,100), font,0.8,(255,0,0),2)
		cv2.putText(draw_mat, ": "+str(tilt), (360,130), font,0.8,(255,0,0),2)

		cv2.putText(draw_mat,"Status",(15,35), font,1.1,(0,0,0),2) 
		cv2.putText(draw_mat,distract_output,(15,100), font, 1,(0,0,255),2)

		cv2.putText(draw_mat,"FPS : "+str(int(1/(end-start))),(550,135), font, 0.8,(0,0,0),2)
		

		if(distract_score >= 30):
			cv2.rectangle(draw_mat, (500, 10), (width-5, 100), (120, 0, 255), 2, cv2.LINE_AA)
			cv2.putText(draw_mat,"dangerous!",(510,60), font, 1.1,(120,0,255),2,cv2.LINE_AA)
		#draw_mat = cv2.resize(draw_mat,(360,640))
		#print((1/(end-start)))
		cv2.imshow("frame", draw_mat)
		videoWriter.write(draw_mat)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			break
	videoWriter.release()
	cap.release()