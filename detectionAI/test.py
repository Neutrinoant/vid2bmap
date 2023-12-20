# 성능 테스트 목적

import os
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import SiameseNetwork
from dataset import LabeledDataset, LabeledNumpyDataset
from config import Config


def build_label_json(ckpt_path, outpath=""):
    
    Config.seed_everything(100)
    
    # image config
    transform = transforms.Compose([transforms.Resize(Config.imgsize), transforms.ToTensor()])
    gray = True
    should_invert = False
    
    # ready for test dataset
    folder_dataset_train = dset.ImageFolder(root=Config.training_dir)
    train_dataset = LabeledDataset(imageFolderDataset=folder_dataset_train,
                                            transform=transform,
                                            should_invert=should_invert,
                                            gray=gray)

    train_dataloader = DataLoader(train_dataset, num_workers=6, batch_size=64, shuffle=False)
    
    # load trained data
    checkpoint = torch.load(ckpt_path, map_location=Config.device)
    model_state_dict = checkpoint["model_state_dict"]
    
    # model setting 
    net = SiameseNetwork(imgsize=Config.imgsize)
    net.load_state_dict(model_state_dict)
    net = net.to(Config.device)
    
    net.eval()  # BatchNorm() 레이어를 끔
    Y_arr = []
    L_arr = []
    
    with torch.no_grad():   # gradient 계산 메모리 사용을 꺼버림
        
        for train_data in train_dataloader:
            X_train, L_train = train_data
            X_train = Variable(X_train).to(Config.device)
            Y_train = net.forward_once(X_train)
            Y_arr.append(Y_train)
            L_arr.append(L_train)
        
        Y_arr = torch.concat(Y_arr, dim=0)
        L_arr = torch.concat(L_arr, dim=0)
        
        # 라벨별로 핵심지표 계산
        L_uid = torch.unique(L_arr)
        
        Cid = []
        
        for label_id in L_uid:
            Y_label = Y_arr[L_arr == label_id]
            centroid = torch.mean(Y_label, dim=0)
            Dist2 = torch.sum((Y_label - centroid)**2, dim=1)
            id_min = torch.argmin(Dist2)
            Cid.append(id_min)
            
        Cid = torch.stack(Cid, dim=0)
        
        # cluster 대표 선정 (label dataset)
        centroids = []
        labels = []
        centroids_y = []
        for i, label_id in enumerate(L_uid):
            Ids = torch.argwhere(L_arr == label_id)
            id_centroid = Ids[Cid[i]]
            img_centroid, label_centroid = train_dataset[id_centroid]
            y_centroid = Y_arr[id_centroid].flatten()
            centroids.append(img_centroid)
            labels.append(label_centroid)
            centroids_y.append(y_centroid)
    
        centroids = torch.stack(centroids, dim=0)
        labels = torch.tensor(labels, dtype=torch.uint8)
        centroids_y = torch.stack(centroids_y, dim=0)
        
    centroids = centroids.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    centroids_y = centroids_y.detach().cpu().numpy()
        
    if outpath != "":
        # save y value of each centroid as json
        jsondata = {
            "ids": labels.tolist(), 
            "names": folder_dataset_train.classes, 
            "centroids_y": centroids_y.tolist()
        }
        with open(outpath, "wt") as f:
            json.dump(jsondata, f)
    
    dataset = LabeledNumpyDataset(centroids, labels)
    return dataset


def test(ckpt_path, label_json=None):
    
    import json
    def load_json(file):
        with open(file, "rt") as f:
            return json.load(f)
    
    Config.seed_everything(100)
    
    print(f"run on device {Config.device}")
    
    # load trained data
    checkpoint = torch.load(ckpt_path, map_location=Config.device)
    model_state_dict = checkpoint["model_state_dict"]
    
    # model setting 
    net = SiameseNetwork(imgsize=Config.imgsize)
    net.load_state_dict(model_state_dict)
    net = net.to(Config.device)
    
    net.eval()  # BatchNorm() 레이어를 끔
    
    correct = []
    
    # image config
    transform = transforms.Compose([transforms.Resize(Config.imgsize), transforms.ToTensor()])
    gray = True
    should_invert = False
    
    # ready for test dataset
    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
    test_dataset = LabeledDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transform,
                                            should_invert=should_invert,
                                            gray=gray)

    test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=64, shuffle=False)
    
    # # ready for labeling dataset
    # if label_dataset is not None:
    #     reference_dataset = label_dataset
    # else:
    #     reference_dataset = LabeledDataset(imageFolderDataset=dset.ImageFolder(root=Config.ref_dir),
    #                                             transform=transform,
    #                                             should_invert=should_invert,
    #                                             gray=gray)

    # reference_dataloader = DataLoader(reference_dataset, num_workers=0, batch_size=64, shuffle=False)
    if label_json is not None:
        reference_dataset = label_json
    else:
        reference_dataset = load_json(Config.ref_path)
    
    with torch.no_grad():   # gradient 계산 메모리 사용을 꺼버림
        
        # # ref dataset forward
        # Y_ref_arr = []
        # L_ref_arr = []
    
        # for reference_data in reference_dataloader:
        #     X_reference, L_reference = reference_data
        #     X_reference = Variable(X_reference).to(Config.device)
        #     Y_reference = net.forward_once(X_reference)
        #     Y_ref_arr.append(Y_reference)
        #     L_ref_arr.append(L_reference)
        
        # Y_ref_arr = torch.concat(Y_ref_arr, dim=0).to(Config.device)
        # L_ref_arr = torch.concat(L_ref_arr, dim=0).to(Config.device)
        Y_ref_arr = torch.tensor(reference_dataset["centroids_y"]).to(Config.device)
        L_ref_arr = torch.tensor(reference_dataset["ids"]).to(Config.device)
        
        assert torch.unique(L_ref_arr).shape == L_ref_arr.shape
        
        # test dataset forward
        Y_test_arr = []
        L_test_arr = []
        
        X_test_arr = []
        for test_data in test_dataloader:
            X_test, L_test = test_data
            X_test = Variable(X_test).to(Config.device)
            Y_test = net.forward_once(X_test)
            Y_test_arr.append(Y_test)
            L_test_arr.append(L_test)
            X_test_arr.append(X_test)
        
        Y_test_arr = torch.concat(Y_test_arr, dim=0).to(Config.device)
        L_test_arr = torch.concat(L_test_arr, dim=0).to(Config.device)
        X_test_arr = torch.concat(X_test_arr, dim=0).to(Config.device)
        
        # choose label
        D2_arr = []
        for label_id in torch.unique(L_ref_arr, sorted=True):
            Y_label = Y_ref_arr[L_ref_arr == label_id][0]   # 어차피 1개뿐
            Dist2 = torch.sum((Y_test_arr - Y_label)**2, dim=1, keepdim=True)
            D2_arr.append(Dist2)
            
        D_arr = torch.concat(D2_arr, dim=1)**0.5
        Id_min = torch.argmin(D_arr, dim=1)
        Id_filtered = torch.where(D_arr[np.arange(D_arr.shape[0]),Id_min] < Config.thr_max_dist, Id_min, 13)
        
        correct = (L_test_arr == Id_filtered)
        count_all = correct.shape[0]
        count_true = torch.count_nonzero(correct)
        
    correct_ratio = 100 * count_true/count_all
    
    return correct_ratio.detach().cpu().numpy()
    
    
def test_all_checkpoints(ckpt_paths):
    
    ratios = []
    for path in ckpt_paths:
        label_json = build_label_json(path)
        ratio = test(path, label_json)
        ratios.append(ratio)
        print(f"{path} completed")
    
    ratios = np.asarray(ratios)
        
    maxid = np.argmax(ratios)
    print(maxid, ratios[maxid])
    
    import json
    with open("all_correct_ratios.json", "wt") as f:
        json.dump(ratios.tolist(), f)


if __name__ == "__main__":
    build_label_json(Config.checkpoint_path, outpath="data/label.json")
    
    ratio = test(Config.checkpoint_path)
    print(ratio)