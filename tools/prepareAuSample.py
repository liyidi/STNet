from tools import ops
import pickle
import glob
import os
import numpy as np
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls
from GCF.GCF_extract import getGCF

seqList = ['seq01-1p-0000','seq02-1p-0000','seq03-1p-0000']
datasetPath = '.../STNet'
#load audio data
audioData = ops.loadAudio(seqList, datasetPath)
au_observe = getGCF(audioData)
for sequence in seqList:
    for cam_number in range(1,4):
        folderPath = f'{datasetPath}/AVsample/imgSample/{sequence}_cam{cam_number}/'
        fileList = sorted(glob.glob(folderPath + '*.pkl'))
        error_curve = list()
        error_total = 0
        MAE_curve = list()
        for i in range(len(fileList)):
            pkl_file = open(f'{fileList[i]}', 'rb')
            sampleIf = pickle.load(pkl_file)
            sampleImg    = sampleIf['sampleImg']
            imgAnno      = sampleIf['imgAnno']
            refImg       = sampleIf['refImg']
            frameNum     = sampleIf['frameNum']
            imgPath      = sampleIf['imgPath']
            img          = ops.read_image(imgPath)
            sampleFace = sampleIf['sampleFace']
            sampleImgBox = sampleIf['sampleImgBox']#sample_img_box for square_img (square coordinate)
        #resize refimg as examplar, do simafc
            ops.showRecimg(sampleImg, sampleFace)
        # box, re_box(in sample coordinate) turn to img coordinate, first.
            new_x = sampleImgBox[0]
            new_y = sampleImgBox[1]
            sampleInImg = ops.square2img(sampleImgBox)
            ops.showRecimg(img, sampleInImg)
        #get GCFmap, and depth index
            GCFmap, depth_ind, gt3d = au_observe.au_observ(sequence, cam_number, frameNum,
                                          img, box=imgAnno, spl_box=sampleInImg)
            gcfData = {
                'GCFmap': GCFmap,
                'depth_ind': depth_ind,
                'GT3D':gt3d,
            }
        ###--- save the imgDataList as {sequence}_sampleList.npz
            folderPath = f'{datasetPath}/AVsample/ausample/{sequence}_cam{cam_number}'
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
            filename = str(10000 + i)[1:]  # '0000.pkl'
            outputPath = open(f'{folderPath}/{filename}.pkl', 'wb')
            pickle.dump(gcfData, outputPath)
            print(f'save gcfData.pkl for {sequence}_cam{cam_number}_{filename}')

