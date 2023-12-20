# 모델 트레이닝 목적

import os

import torch
from torch import optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import SiameseNetwork
from dataset import SiameseNetworkDataset
from config import Config
from logger import logger


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=10.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def train():
    
    Config.seed_everything(Config.seed)
    
    logger.info(f"run on device {Config.device}")
    
    folder_dataset = dset.ImageFolder(root=Config.training_dir)

    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize(Config.imgsize),
                                                                        transforms.ToTensor()
                                                                        ]),
                                            should_invert=False,
                                            iternum=Config.train_data_iter)

    train_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=0,
                            batch_size=Config.train_batch_size)
    
    os.makedirs(Config.checkpoint_dir, exist_ok=True)

    net = SiameseNetwork(imgsize=Config.imgsize).to(Config.device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0,Config.train_number_epochs):
        total_loss = 0.
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.to(Config.device), img1.to(Config.device) , label.to(Config.device)
            
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            
            optimizer.zero_grad()
            loss_contrastive.backward()
            optimizer.step()
            
            total_loss += loss_contrastive.item()
            
        else:
            avg_loss = total_loss/(i+1)
            logger.info(f"Epoch {epoch}: Average loss {avg_loss}")
            iteration_number += i+1
            counter.append(iteration_number)
            loss_history.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            torch.save({
                'model_state_dict': net.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss,
                'epoch': epoch+1,
            }, os.path.join(Config.checkpoint_dir, f"{epoch+1}.ckpt"))
    else:
        # save model
        total_epoch = epoch
        torch.save({
        'model_state_dict': net.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'losses': loss_history,
        'epoch': total_epoch
        }, os.path.join(Config.checkpoint_dir, f"last.ckpt"))
    
    plt.plot(counter,loss_history)
    plt.savefig("loss.jpg")


if __name__ == "__main__":
    train()