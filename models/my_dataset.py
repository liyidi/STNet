import numpy as np
import random
from torch.utils.data import Dataset
import cv2
import pickle

random.seed(1)
class myDataset(Dataset):
    def __init__(self,dataList, refData):
        self.refData = refData
        self.dataList = dataList
        self.minGCF = -0.01
        self.maxGCF = 0.50

    def __getitem__(self, index):
        imgSamplePath= self.dataList[index]
        img_file = open(imgSamplePath, 'rb')
        imgData = pickle.load(img_file) # get whole img data
        refname = imgSamplePath.split('/')[-2]
        refImg = self.refData[refname]#

        auSamplePath = imgSamplePath.replace('imgSample', "GCFmap_train")
        au_file = open(auSamplePath, 'rb')
        auData = pickle.load(au_file)

        #sample need resize
        sampleImg = imgData['sampleImg']
        sampleImg0 = cv2.resize(sampleImg, (300, 300)
                               , interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
        sampleImg1 = cv2.resize(sampleImg, (400, 400)
                               , interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
        sampleImg2 = cv2.resize(sampleImg, (500, 500)
                               , interpolation=cv2.INTER_NEAREST).transpose(2,0,1)

        sampleFace = imgData['sampleFace']
        gt2d = np.array([sampleFace[0] + sampleFace[2] / 2, sampleFace[1] + sampleFace[3] / 2])
        GCFmap = auData['GCFmap']  # 3*H*W
        GCF_nor = (GCFmap - self.minGCF) / (self.maxGCF - self.minGCF)
        GCF_nor = np.transpose(GCF_nor, (1, 2, 0))  # H*W*3
        reGCFmap = cv2.resize(GCF_nor, (400, 400)
                              , interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
        return refImg, sampleImg0, sampleImg1, sampleImg2, reGCFmap, gt2d

    def __len__(self):
        return len(self.dataList)





