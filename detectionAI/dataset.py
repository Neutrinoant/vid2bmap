# 데이터셋 관리 목적

import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import PIL.ImageOps
import cv2

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True, gray=True, iternum=3):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        self.gray = gray
        self.iternum = iternum  # 전체 데이터를 몇번 사용할것인지
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        
        if self.gray:
            img0 = img0.convert("L")
            img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs) * self.iternum


class LabeledDataset(Dataset):
    
    def __init__(self,imageFolderDataset, transform=None,should_invert=True, gray=True):
        self.imageFolderDataset = imageFolderDataset 
        self.transform = transform
        self.should_invert = should_invert
        self.gray = gray
        
    def __getitem__(self,index):
        img_tuple = self.imageFolderDataset.imgs[index]
        img0 = Image.open(img_tuple[0])
        label = img_tuple[1]
        
        if self.gray:
            img0 = img0.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)

        if self.transform is not None:
            img0 = self.transform(img0)
        
        return img0, label
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class LabeledNumpyDataset(Dataset):
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        assert len(images) == len(labels)
        
    def __getitem__(self,index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)


class VideoBBoxDataset(Dataset):
    
    def __init__(self, saved_paths, bboxes, out_shape, should_invert=True, gray=True):
        self.saved_paths = saved_paths
        self.bbox_flat = bboxes.reshape((-1,4))
        self.out_shape = out_shape
        self.should_invert = should_invert
        self.gray = gray
        
    def __getitem__(self,index):
        img = Image.open(self.saved_paths[index])
        
        # 조건에 따라 변형 함수 중첩 정의
        f_gray = lambda m: (m.convert("L") if self.gray else m)
        f_inv = lambda m: (PIL.ImageOps.invert(f(m)) if self.should_invert else m)
        f_trans = transforms.Compose([transforms.Resize(self.out_shape), transforms.ToTensor()])
        f = lambda m: f_trans(f_inv(f_gray(m)))
        
        # 이미지에서 모든 bbox를 crop 후 변형
        imgrois = [f(img.crop((x1,y1,x2+1,y2+1))) for y1,x1,y2,x2 in self.bbox_flat]
        
        img_tensors = torch.stack(imgrois, dim=0)
        
        return img_tensors
    
    def __len__(self):
        return len(self.saved_paths)


class VideoFrameDataset(Dataset):
    
    def __init__(self, frame_paths, gray=True):
        self.saved_paths = frame_paths
        self.gray = gray
        
    def __getitem__(self,index):
        img = cv2.imread(self.saved_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img)
        return img_tensor
    
    def __len__(self):
        return len(self.saved_paths)
