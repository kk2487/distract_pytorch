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

#Override transform's resize function(using letterbox)
class letter_img(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """

        #letterbox(img,self.size).save('out.bmp')
        return letterbox(img, self.size)
    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

# Can print the model structure

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
	# print('test')
	#for i, inputs in enumerate(test_data):
	# print('---------------------------',inputs.shape)
	# print(test_data)
	inputs = Variable(test_data, volatile=True).cuda()

	outputs = model(inputs) 
	outputs = F.softmax(outputs)
	print(classes[outputs.argmax()])




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
# ----------------------------------------------------


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

	print('--------------------------------------------------')
	model, parameters = generate_model(opt)
	print('--------------------------------------------------')
	checkpoint = torch.load(opt.resume_path)
	opt.arch == checkpoint['arch']
	model.load_state_dict(checkpoint['state_dict'])

	spatial_transform = Compose([
		letter_img(opt.sample_size),
		ToTensor(opt.norm_value), 
		norm_method
    ])
	temporal_transform = TemporalCenterCrop(opt.sample_duration)

# -------------------------------------------------------------------------------

	font = cv2.FONT_HERSHEY_SIMPLEX

	cap = cv2.VideoCapture(fileUrl)
	ret, frame = cap.read()
	frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
	height, width = frame.shape[:2]

	fps = cap.get(cv2.CAP_PROP_FPS)
	model.eval()
	while(ret):
		start_time = time.time()
		ret, frame = cap.read()
		if(not ret):
			break

		frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
		#frame = cv2.resize(frame, (112,112))
		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		full_clip.append(frame)
		if len(full_clip) > 8:
			
			test_data = get_test_data(full_clip, spatial_transform)
			
			predict(model, test_data)
			full_clip = []

			print(1/(time.time()-start_time))
		cv2.imshow("frame", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			break

	cap.release()