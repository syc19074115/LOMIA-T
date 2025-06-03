from torch.utils.data import DataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np
from .transforms import Rot90, Flip, Identity, Compose, RandCrop3D, RandomShift, RandomRotion ,CenterCrop,RandomFlip,RandomIntensityChange
#from transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


class Train_Dataset(Dataset):
    def __init__(self,args):
        #super.__init__(self,args)
        self.args = args

        self.filename_list_region = self.load_file_name_list('')
        self.transformsq = Compose([
            RandomShift(3),
            RandomRotion(2),
            RandomIntensityChange((0.1,0.1)),
            #CenterCrop((40,64,64))
        ])  # 数据增强就体现在这里
        self.transformsq1 = RandCrop3D((32,48,48))
        
        self.transformsk = Compose([
            RandomShift(3),
            RandomRotion(2),
            RandomIntensityChange((0.1,0.1)),
            #CenterCrop((40,64,64))
        ])  # 数据增强就体现在这里
        self.transformsk1 = RandCrop3D((32,48,48))  #16 48 48 



    def __getitem__(self, index):
        pre_region_ct_array, pre_region_seg_array, = self.load_cut_or_region_array(self.filename_list_region,index)
        q = pre_region_ct_array#post_region_ct_array#pre_region_ct_array#
        k = pre_region_ct_array#post_region_ct_array
       
        q = self.transformsq1(self.transformsq(q))
        k = self.transformsk1(self.transformsk(k))

        q = self.normalization(q)
        k = self.normalization(k)  #Zscore在分类上的作用可能是好于视窗的，因为计算的是两者间距离
        
        q = torch.FloatTensor(q.copy())
        k = torch.FloatTensor(k.copy())
        #print(q.shape)
        return q, k


    def __len__(self):
        return len(self.filename_list_region)

    def load_cut_or_region_array(self, filename_list,index):
        ct_pre = sitk.ReadImage(filename_list[index][0], sitk.sitkFloat32)
        seg_pre = sitk.ReadImage(filename_list[index][1], sitk.sitkUInt8)

        pre_ct_array = sitk.GetArrayFromImage(ct_pre)
        pre_seg_array = sitk.GetArrayFromImage(seg_pre)
        pre_ct_array = pre_ct_array[None, ...]
        pre_seg_array = pre_seg_array[None, ...]
       
        return pre_ct_array,pre_seg_array,

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据,strip()函数去掉了行末的换行
                if not lines:
                    break
                file_name_list.append(lines.split())  #split()将lines分成两个元素，然后组成一个list

        # print(file_name_list[0])
        return file_name_list

    def normalize(self,image, chestwl=30.0, chestww=350.0):
        # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        # print(chestwl,chestww)
        Low = chestwl - 0.5 * chestww
        High = chestwl + 0.5 * chestww
        image[image > High] = High
        image[image < Low] = Low
        image = (image - Low) / (High - Low)
        return image

    def normalization(self, data):
        mask = data.sum(0) > 0
        x = data[0, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        data[0, ...] = x
        return data

    
if __name__ == '__main__':
    sys.path.append('./')
    from config import args
    train_ds = Train_Dataset(args)
    train_dl = DataLoader(train_ds, 1, False, num_workers=1)
    for i, (pre, post) in enumerate(train_dl):
        print("--------------------------------{}-------------------------------".format(i))
        print(pre.size(), post.size(),)
        if i != -1:
            pre = pre.numpy()
            post = post.numpy()
            plt.subplot(141)
            plt.axis('off')
            plt.imshow(pre[0][0][8],cmap='gray')
            plt.subplot(142)
            plt.axis('off')
            plt.imshow(pre[0][0][8],cmap='gray')
            plt.subplot(143)
            plt.axis('off')
            plt.imshow(post[0][0][8],cmap='gray')
            plt.subplot(144)
            plt.axis('off')
            plt.imshow(post[0][0][8],cmap='gray')
            plt.savefig()

