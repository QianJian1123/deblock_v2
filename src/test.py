import torch
import math,sys
import numpy
import argparse 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
#from label_toMemory import Dataset
from label import Dataset
#from CNN_1_keep_training import ConvNet
#from CNN_40 import ConvNet
from CNN import DnCNN as ConvNet
#from VRCNN import ConvNet
import cv2


def main():
    qp=45

    netname='../model/dncnn_learningrate_1e-2_step1.pth'
    imgname='/home/qj/Deblocking/test_with_deblocking/img/{}'.format(qp)
    labname='/home/qj/Deblocking/test_with_deblocking/lab'
    
    #load and test model
    
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = ConvNet()
    model.load_state_dict(torch.load(netname))
    model.to(device)
    model.eval()  
    
    #  test dataset
    test_dataset = Dataset(imgname,labname,train=False)
    # test Data loader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1,
                                                shuffle=False)
    # Test the model
    #model.to(device)
    with torch.no_grad():
        psnr_value_img=0
        psnr_value_out=0
        num=0
        sum_img=sum_out=0
        sum_rdo_add=0
        for images, labels in test_loader:
            
            images = images.type(torch.cuda.FloatTensor).to(device)
            labels = labels.type(torch.cuda.FloatTensor).to(device)
            
            outputs = model(images)
            psnr_value_out=calc_psnr(outputs,labels,0,1.0)
            psnr_value_img=calc_psnr(images,labels,0,1.0)
            print('ADD:{:.6f}  OUT:{:.6f}  IMG:{:.6f}'.format(psnr_value_out-psnr_value_img,psnr_value_out,psnr_value_img))
            num+=1
            sum_img+=psnr_value_img
            sum_out+=psnr_value_out
            if psnr_value_out>psnr_value_img:
                sum_rdo_add+=psnr_value_out-psnr_value_img
            if num is 1:
                cv2.imwrite('./out.png',outputs.cpu().numpy())
                cv2.imwrite('./img.png',images.cpu().numpy())
                cv2.imwrite('./lab.png',labels.cpu().numpy())
        t_out=sum_out/num
        t_img=sum_img/num
        t_add=t_out-t_img
        t_rdo=sum_rdo_add/num
        print('TOTAL:\nADD:{}\nRDO:{}\nOUT:{}\nIMG:{}\n'.format(t_add,t_rdo,t_out,t_img))

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    sr = sr.detach().cpu()
    hr = hr.detach().cpu()
    diff = (sr - hr).data.div(rgb_range)
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6

    valid = diff[:, :,:, :]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


if __name__ == '__main__':
    main()
