# 모델 설계 목적

import torch.nn as nn

class SiameseNetworkColor(nn.Module):
    def __init__(self):
        super(SiameseNetworkColor, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1*3, 4*3, kernel_size=3, groups=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4*3),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4*3, 8*3, kernel_size=3, groups=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8*3),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8*3, 8*3, kernel_size=3, groups=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8*3),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*3*100*100, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 10))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseNetwork(nn.Module):
    def __init__(self, imgsize):
        super(SiameseNetwork, self).__init__()
        self.imgsize = imgsize  # (h,w)
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*self.imgsize[0]*self.imgsize[1], 256),    # static image size
            nn.ReLU(inplace=True),

            nn.Linear(256, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 10))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
