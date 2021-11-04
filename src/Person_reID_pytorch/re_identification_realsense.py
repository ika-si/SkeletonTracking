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
import cv2

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

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
        
#if __name__ == '__main__':
    
    
model_pred = 0
class_names = 0
loader = 0
dict_image_id = {}
    

def model_load():
    
    global model_pred, class_names, loader
    
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

    data_dir = 'C:/Users/sugimura/workspace/SkeletonTracking/src/Person_reID_pytorch/3pro_Data_5_a_1029/pytorch'
    
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
    device = torch.device("cuda")
    model_pred.load_state_dict(torch.load('C:/Users/sugimura/workspace/SkeletonTracking/src/Person_reID_pytorch/model/ResNet50_a_1029_both/net_last.pth', map_location="cuda:0"), strict=False)
    model_pred.to(device)
    model_pred.eval()
    
    imsize = 256
    loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

def pred_person():
    
    global dict_image_id
    
    files = os.listdir('C:/Users/sugimura/workspace/SkeletonTracking/src/data/temp')
    #files = os.listdir('C:/Users/sugimura/workspace/SkeletonTracking/src/Person_reID_pytorch/testData')
    
    id = 0
    re_id_list = []
    for file_name in files:
        print(file_name)
        path = os.path.join('C:/Users/sugimura/workspace/SkeletonTracking/src/data/temp', file_name)
        #path = os.path.join('C:/Users/sugimura/workspace/SkeletonTracking/src/Person_reID_pytorch/testData', file_name)
        
        image = Image.open(path)
        #plt.imshow(image)
        #plt.show()
        
        #画像の前処理
        image = Image.open(path).convert("RGB")
        image = loader(image)
        image = gamma_processing(image)
        image = Variable(image, requires_grad=True)
        inputs = image.unsqueeze(0)
        device = 'cuda'
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
        if(class_names[preds] == '001'):
            id = 1
        elif(class_names[preds] == '002'):
            id = 2
        elif(class_names[preds] == '003'):
            id = 3
        
        skeleton_id = file_name[-5]
        dict_image_id[skeleton_id] = id
        
        print()
        
        re_id_list.append(class_names[preds])
    
    print(re_id_list)
    
    
def gamma_processing(input_image):
    #CLAHE補正
    img_yuv = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            
    #両方
    gamma_both = 2.0
    gamma_cvt_both = np.zeros((256,1),dtype = 'uint8')
    for i in range(256):
        gamma_cvt_both[i][0] = 255 * (float(i)/255) ** (1.0/gamma_both)
    img_gamma_clahe = cv2.LUT(img_clahe,gamma_cvt_both)

    filter_out_b = cv2.bilateralFilter(img_gamma_clahe, 3, 1.5, 1.5)

    return filter_out_b


re_id_0 = 1
re_id_1 = 2
re_id_2 = 3

def re_identification(skeleton_2D_id):
    #reidがうごかないとき
    if skeleton_2D_id == 0:
        return 1
    elif skeleton_2D_id == 1:
        return 2
    elif skeleton_2D_id == 2:
        return 3
    else:
        return 4
    
    '''
    if dict_image_id.setdefault(str(skeleton_id)) == None:
        return 3
    else:
        return dict_image_id[str(skeleton_id)]
    '''
    