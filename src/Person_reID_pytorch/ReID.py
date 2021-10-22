from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os

from torch.nn import init
from torchvision import models

import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import yaml
from shutil import copyfile

import copy
version =  torch.__version__


from PIL import Image
'''#transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

data_dir = 'C:/Users/sugimura/workspace/SkeletonTracking/src/Person_reID_pytorch/3pro_Data_val/pytorch'

#train = ''

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir,'train'),data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir,'val'),data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []'''


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.model = model_ft
        self.circle = circle
        
#         self.fc_f = torch.nn.Linear(2048, 1000)
        
        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
#         x = self.classifier(x)
        return x

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.ftnet = ft_net(751)
        
        self.ftnet.classifier = nn.Sequential()
        self.ftnet.model.fc2 = nn.Linear(1000, 500)
        self.ftnet.model.fc3 = nn.Linear(500, 2)
#         self.fc1 = nn.Linear(100, 2)
        
#    def load_ftnet_model():
#        model = self.ftnet
#        device = torch.device('cpu')
#        model.load_state_dict(torch.load("model/ft_ResNet50/net_last.pth", map_location=device), strict=False)
    
    def forward(self, x):
        x = self.ftnet.model.conv1(x)
        x = self.ftnet.model.bn1(x)
        x = self.ftnet.model.relu(x)
        x = self.ftnet.model.maxpool(x)
        x = self.ftnet.model.layer1(x)
        x = self.ftnet.model.layer2(x)
        x = self.ftnet.model.layer3(x)
        x = self.ftnet.model.layer4(x)
        x = self.ftnet.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.ftnet.model.fc(x)
        x = self.ftnet.model.fc2(x)
        x = self.ftnet.model.fc3(x)
        return x
    
    
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x

if __name__ == '__main__':
    
    transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
        }

    data_dir = 'C:/Users/sugimura/workspace/SkeletonTracking/src/Person_reID_pytorch/3pro_Data_val/pytorch'
    
    #train = ''
    
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir,'train'),data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir,'val'),data_transforms['val'])
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                 shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    use_gpu = torch.cuda.is_available()
    
    since = time.time()
    inputs, classes = next(iter(dataloaders['train']))
    print(time.time()-since)
    
    y_loss = {} # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []
    
    model_pred = Net()
    device = torch.device('cpu')
    model_pred.load_state_dict(torch.load("model/ft_ResNet50_finetuning/net_29.pth", map_location=device), strict=False)
    model_pred.eval()
    imsize = 256
    loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
    
    files = os.listdir('testData')
    i = 0
    for file_name in files:
        print(file_name)
        path = 'testData/' + file_name
        image = Image.open(path)
        plt.imshow(image)
        plt.show()
        
        #画像の前処理
        image = Image.open(path).convert("RGB")
        image = loader(image)
        image = Variable(image, requires_grad=True)
        inputs = image.unsqueeze(0)
        device = 'cpu'
        inputs = inputs.to(device)
        
        # 推論結果出力
    
        outputs = model_pred(inputs)  # torch.Size([1, 2])
        _, preds = torch.max(outputs.data, 1)
        print("推論値：", outputs)
        print("入力画像の推論結果：", class_names[preds])
    
    #     msg = osc_message_builder.OscMessageBuilder(address="/ who %s" % i)
    #     if(class_names[preds] == '001'):
    #         print(class_names[preds])
    #         msg.add_arg(0, osc_message_builder.OscMessageBuilder.ARG_TYPE_INT)
    #         msg = msg.build()
    #         client.send(msg)
    #     elif(class_names[preds] == '002'):
    #         print(class_names[preds])
    #         msg.add_arg(1, osc_message_builder.OscMessageBuilder.ARG_TYPE_INT)
    #         msg = msg.build()
    #         client.send(msg)
    #     else:
    #         msg.add_arg(2, osc_message_builder.OscMessageBuilder.ARG_TYPE_INT)
    #         msg = msg.build()
    #         client.send(msg)
        
        print()
        i += 1
    
    
def image_processing():
        return 1




def re_identification(id):
    if id == 0:
        return 1
