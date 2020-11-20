import os
import sys
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from tqdm import tqdm
import utils

transformer=transforms.Compose([
	transforms.Resize((256,256)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
	transforms.Normalize([0.485,0.456,0.406], # 0-1 to [-1,1] , formula (x-mean)/std
						[0.229,0.224,0.255])
])

class ConvNet(nn.Module):
	def __init__(self,num_classes=6):
		super(ConvNet,self).__init__()
		
		#Input shape= (3,256,256)
		
		self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
		#Shape= (32,256,256)
		self.relu1=nn.ReLU()
		#Shape= (32,256,256)
		self.pool1=nn.MaxPool2d(kernel_size=2)
		#Shape= (32,128,128)
		
		
		self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
		#Shape= (64,128,128)
		self.relu2=nn.ReLU()
		#Shape= (64,128,128)
		self.pool2=nn.MaxPool2d(kernel_size=2)
		#Shape= (64,64,64)
		
		
		self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
		#Shape= (128,64,64)
		self.relu3=nn.ReLU()
		#Shape= (128,64,64)
		self.pool3=nn.MaxPool2d(kernel_size=2)
		#Shape= (128,32,32)        
		
		self.conv4=nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1)
		#Shape= (64,32,32)
		self.relu4=nn.ReLU()
		#Shape= (64,32,32)
		self.pool4=nn.MaxPool2d(kernel_size=2)
		#Shape= (64,16,16)   
  
		self.conv5=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1)
		#Shape= (32,16,16)
		self.relu5=nn.ReLU()
		#Shape= (32,16,16)
		self.pool5=nn.MaxPool2d(kernel_size=2)
		#Shape= (32,8,8) 
		
		self.fc1=nn.Linear(in_features=8 * 8 * 32,out_features=1024)
		self.relu6=nn.ReLU()
		self.dropout1=nn.Dropout(p=0.6)

		self.fc2=nn.Linear(in_features=1024,out_features=512)
		self.relu7=nn.ReLU()
		self.dropout2=nn.Dropout(p=0.6)

		self.fc3=nn.Linear(in_features=512,out_features=num_classes)
		
		#Feed forwad function
		
	def forward(self,input):
		output=self.conv1(input)
		output=self.relu1(output)
		output=self.pool1(output)
		  
			
		output=self.conv2(output)
		output=self.relu2(output)
		output=self.pool2(output)
		
		output=self.conv3(output)
		output=self.relu3(output)
		output=self.pool3(output)
			
		output=self.conv4(output)
		output=self.relu4(output)
		output=self.pool4(output)    
		
		output=self.conv5(output)
		output=self.relu5(output)
		output=self.pool5(output)
		
		output=output.view(-1, 8 * 8 * 32)        
		
		output=self.fc1(output)
		output=self.relu6(output)
		output=self.dropout1(output)
		
		output=self.fc2(output)
		output=self.relu7(output)
		output=self.dropout2(output)
		
		output=self.fc3(output)
			
		return output

def load_model():
	checkpoint=torch.load(utils.model_path)
	model=ConvNet(num_classes=6)
	model.load_state_dict(checkpoint)
	return model

def write_classes(file_path, classes):

	fp = open(file_path, "w")
	for i in range(len(classes)):
		fp.write(str(classes[i]))
		if (i == len(classes)-1):
			break
		fp.write(',')
	fp.close()

def read_classes(file_path):
	fp = open(file_path, "r")
	classes = fp.readline()
	classes = classes. split(",")
	fp.close()

	return classes


if __name__ == '__main__':
	if(sys.argv[1] == '--train'):
		device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		print('-------------------------------------')
		print('Training Device :')
		print(device)
		print(torch.cuda.get_device_name(0))
		#print('Memory Usage:')
		#print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
		#print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
		print('-------------------------------------')
		transformer=transforms.Compose([
			transforms.Resize((256,256)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
			transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.255])
		])

		#Dataloader

		train_path=utils.train_path
		validation_path=utils.validation_path

		train_loader=DataLoader(
			torchvision.datasets.ImageFolder(train_path,transform=transformer),
			batch_size=64, shuffle=True
		)
		validation_loader=DataLoader(
			torchvision.datasets.ImageFolder(validation_path,transform=transformer),
			batch_size=64, shuffle=True
		)
		root=pathlib.Path(train_path)
		
		classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
		print(classes)
		write_classes('classes.txt',classes);

		model=ConvNet(num_classes=6).to(device)

		#Optmizer and loss function
		optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
		loss_function=nn.CrossEntropyLoss()


		num_epochs = utils.training_epochs
		#calculating the size of training and testing images
		
		train_count=len(glob.glob(train_path+'/**/*.jpg'))
		validation_count=len(glob.glob(validation_path+'/**/*.jpg'))
		
		print(train_count,validation_count)
		
		#Model training and saving best model

		best_accuracy=0.0

		for epoch in range(num_epochs):
	
			#Evaluation and training on training dataset
			model.train()
			train_accuracy=0.0
			train_loss=0.0
	
			for (images,labels) in tqdm(train_loader):
				if torch.cuda.is_available():
					images=Variable(images.cuda())
					labels=Variable(labels.cuda())
				
				optimizer.zero_grad()
		
				outputs=model(images)
				loss=loss_function(outputs,labels)
				loss.backward()
				optimizer.step()
		
		
				train_loss+= loss.cpu().data*images.size(0)
				_,prediction=torch.max(outputs.data,1)
		
				train_accuracy+=int(torch.sum(prediction==labels.data))
		
			train_accuracy=train_accuracy/train_count
			train_loss=train_loss/train_count
	
	
			# Evaluation on validation dataset
			model.eval()
			
			validation_accuracy=0.0
			for (images,labels) in tqdm(validation_loader):
				if torch.cuda.is_available():
					images=Variable(images.cuda())
					labels=Variable(labels.cuda())
				with torch.no_grad():    
					outputs=model(images)
				_,prediction=torch.max(outputs.data,1)
				validation_accuracy+=int(torch.sum(prediction==labels.data))
			
			validation_accuracy=validation_accuracy/validation_count
			
			
			print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' validation Accuracy: '+str(validation_accuracy))
			
			#Save the best model
			if validation_accuracy>best_accuracy:
				torch.save(model.state_dict(),utils.model_path)
				best_accuracy=validation_accuracy
	
