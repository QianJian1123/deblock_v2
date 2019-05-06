import os
import numpy as np
import random
from torch.utils.data import Dataset
class Dataset_FULL(Dataset):
    def __init__(self, path_pre_img, path_org_img,train,patch_size=32):
        self.path_pre_img = path_pre_img
        self.path_org_img = path_org_img
        self.train = train
        self.image_pre = []
        self.patch_size=patch_size
        #print(self.path_pre_img)
        ls_pre = os.listdir(path_pre_img)
        for file in ls_pre:
            if file.find('.y') != -1:
                pre_name = file
                width, height = (int(_) for _ in pre_name.split('_')[1].split('x'))
                widthnum=width//self.patch_size
                heightnum=height//self.patch_size
                total_num=widthnum*heightnum
                for idxnum in range(total_num):
                    self.image_pre.append(pre_name)

    def readYuvFile(self, filename, width, height):
        #print('readYUV')
        with open(filename, 'rb') as rfile:
            Y = np.fromfile(rfile, 'uint8', width * height).reshape([height, width])
            wuv = width // 2
            huv = height // 2
            #U = np.fromfile(rfile, 'uint8', wuv * huv).reshape([huv, wuv])
            #V = np.fromfile(rfile, 'uint8', wuv * huv).reshape([huv, wuv])
        return Y



    def __getitem__(self, idx):
        #print('getitem')
        pre_path = self.image_pre[idx]

        width, height = (int(_) for _ in pre_path.split('_')[1].split('x'))

        pre_path1 = os.path.join(self.path_pre_img, pre_path)
        org_path = os.path.join(self.path_org_img, pre_path)
        if os.path.exists(org_path):

            pY = self.readYuvFile(pre_path1, width, height)
            oY = self.readYuvFile(org_path, width, height)
        else :
            
            pre_path_27=pre_path.replace('_27.y','.y')
            pre_path_27_32=pre_path_27.replace('_32.y','.y')
            pre_path_27_32_38=pre_path_27_32.replace('_38.y','.y')
            pre_path_to_be_changed=pre_path_27_32_38.replace('_45.y','.y')
            org_path = os.path.join(self.path_org_img, pre_path_to_be_changed)
            pY = self.readYuvFile(pre_path1, width, height)
            oY = self.readYuvFile(org_path, width, height)            

        if self.train:
            pY, oY = self.getpatch(pY,oY,idx)

        pY=np.expand_dims(pY,axis=0)
        oY=np.expand_dims(oY, axis=0)
        return(pY/255.0,oY/255.0)

    def __len__(self):
        if self.train:
            return(len(self.image_pre))
        else:
            return 30
    def getpatch(self, p_r, o_r,idx):
        
        h, w = p_r.shape[:2]
        widthnum=w//self.patch_size
        num_h=idx//widthnum
        num_w=idx%widthnum
        px = ox = num_w*self.patch_size
        py = oy = num_h*self.patch_size
        p_r = p_r[px:px+self.patch_size, py:py+self.patch_size]
        o_r = o_r[ox:ox+self.patch_size, oy:oy+self.patch_size]
        #print(p_r,o_r)
        return p_r,o_r

    def getpatchYUV(self, pY, pU, pV, oY, oU, oV):
        pY, oY = self.getbatch(pY, oY)
        pU, oU = self.getbatch(pU, oU)
        pV, oV = self.getbatch(pV, oV)
        return pY, pU, pV, oY, oU, oV
class Dataset(Dataset):
    def __init__(self, path_pre_img, path_org_img,train,patch_size=32):
        self.path_pre_img = path_pre_img
        self.path_org_img = path_org_img
        self.train = train
        self.image_pre = []
        self.image_org = []
        self.patch_size=patch_size
        #print(self.path_pre_img)
        ls_pre = os.listdir(path_pre_img)
        for file in ls_pre:
            if file.find('.y') != -1:
                if self.train:
                    if int(file.split('.')[1].split('_')[1])==30 or int(file.split('.')[1].split('_')[1])==15 or int(file.split('.')[1].split('_')[1])==0:
                        pre_name = file
                        self.image_pre.append(pre_name)
                        #print(file)
                else:
                    pre_name = file
                    self.image_pre.append(pre_name)
                #print(file)
        #print(self.image_pre)
    def readYuvFile(self, filename, width, height):
        #print('readYUV')
        with open(filename, 'rb') as rfile:
            Y = np.fromfile(rfile, 'uint8', width * height).reshape([height, width])
            wuv = width // 2
            huv = height // 2
            #U = np.fromfile(rfile, 'uint8', wuv * huv).reshape([huv, wuv])
            #V = np.fromfile(rfile, 'uint8', wuv * huv).reshape([huv, wuv])
        return Y



    def __getitem__(self, idx):
        #print('getitem')
        pre_path = self.image_pre[idx]
        

        width, height = (int(_) for _ in pre_path.split('_')[1].split('x'))


        pre_path1 = os.path.join(self.path_pre_img, pre_path)
        org_path = os.path.join(self.path_org_img, pre_path)
        if os.path.exists(org_path):

            pY = self.readYuvFile(pre_path1, width, height)
            oY = self.readYuvFile(org_path, width, height)
        else :
            
            pre_path_27=pre_path.replace('_27.y','.y')
            pre_path_27_32=pre_path_27.replace('_32.y','.y')
            pre_path_27_32_38=pre_path_27_32.replace('_38.y','.y')
            pre_path_to_be_changed=pre_path_27_32_38.replace('_45.y','.y')

            org_path = os.path.join(self.path_org_img, pre_path_to_be_changed)
            pY = self.readYuvFile(pre_path1, width, height)
            oY = self.readYuvFile(org_path, width, height)            
        #print(pY-oY)
        #print('\n*****************\n')
        if self.train:
            pY, oY = self.getpatch(pY,oY)
        pY=np.expand_dims(pY,axis=0)
        oY=np.expand_dims(oY, axis=0)
        return(pY/255.0,oY/255.0)

    def __len__(self):
        if self.train:
            return(len(self.image_pre))
        else:
            return 30
    def getpatch(self, p_r, o_r):
        
        h, w = p_r.shape[:2]

        px = ox = random.randrange(0, h-self.patch_size+1)
        py = oy = random.randrange(0, w-self.patch_size+1)

        p_r = p_r[px:px+self.patch_size, py:py+self.patch_size]
        o_r = o_r[ox:ox+self.patch_size, oy:oy+self.patch_size]

        return p_r,o_r

    def getpatchYUV(self, pY, pU, pV, oY, oU, oV):
        pY, oY = self.getbatch(pY, oY)
        pU, oU = self.getbatch(pU, oU)
        pV, oV = self.getbatch(pV, oV)
        return pY, pU, pV, oY, oU, oV

if __name__=='__main__':
    d = Dataset()