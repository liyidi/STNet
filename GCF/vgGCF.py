from tools import ops
import pickle
import glob
import cv2
import os
import numpy as np
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls
from GCF.GCF_extract_vg import getGCF
import matplotlib.pyplot as plt
import time
import random
#set seq
seqList = ['seq08-1p-0100','seq11-1p-0100','seq12-1p-0100']
datasetPath = '.../STNet'
au_observe = getGCF()
for sequence in seqList:
    GCCdata = ops.loadGCC(sequence, datasetPath)
    audioDATA = ops.loadaudioDATA(sequence, datasetPath)
    for cam_number in range(1,4):
        GCC = GCCdata[f'{sequence}_cam{cam_number}']
        DATA = audioDATA[f'{sequence}_cam{cam_number}']
        folderPath = f'{datasetPath}/AVsample/imgSample/{sequence}_cam{cam_number}/'
        fileList = sorted(glob.glob(folderPath + '*.pkl'))
        error_curve = list()
        error_total = 0
        MAE_curve = list()
        index = random.sample(range(0, len(fileList)), round(len(fileList)*0.75))#index of face guide
        start = time.time()
        for i in range(len(fileList)):#len(fileList)
            pkl_file = open(f'{fileList[i]}', 'rb')
            sampleIf = pickle.load(pkl_file)
            imgAnno      = sampleIf['imgAnno']
            frameNum     = sampleIf['frameNum']#0-index
            sampleFace = sampleIf['sampleFace']
            imgPath      = sampleIf['imgPath']
            img          = ops.read_image(imgPath)  # org img
            sampleImgBox = sampleIf['sampleImgBox']##sample_img_box for square_img (square coordinate)
        #resize refimg as examplar, do simafc
            # ops.showRecimg(sampleImg, boxInSample)
        # box, re_box(in sample coordinate) turn to img coordinate, first.
            new_x = sampleImgBox[0]
            new_y = sampleImgBox[1]
            sampleInImg = ops.square2img(sampleImgBox)
            # ops.showRecimg(img, sampleInImg)
        #get GCFmap, and depth index
            if i in index:
                area = imgAnno
            else:
                area = sampleInImg
            GCFmap, depth_ind = au_observe.au_observ(DATA, GCC, cam_number, frameNum,
                                      box=area, spl_box=sampleInImg)
            gcfData = {
                'GCFmap': GCFmap,
                'depth_ind': depth_ind,
            }
        ###---calculate the errors
            gcf_t = cv2.resize(np.sum(GCFmap, axis=0), (120, 120))
            ind = np.unravel_index(gcf_t.argmax(), gcf_t.shape)
            loc2d = np.array([ind[1], ind[0]])
            gt2d = np.array([sampleFace[0] + sampleFace[2] / 2, sampleFace[1] + sampleFace[3] / 2])
            error2d = np.sqrt(np.sum(np.asarray(loc2d - gt2d) ** 2))
            error_curve.append(error2d)
            error_total += error2d
            MAE = error_total / (i + 1)
            MAE_curve.append(MAE)
        print("[{}_cam{} sample:{:0>3}/{:0>3}] [error2d:{:.4f} MAE:{:.4f}] ".format(
                sequence, cam_number, i + 1,len(fileList), error2d, MAE))


        plt.plot(MAE_curve)
        plt.title(f'MAE of {sequence}cam_{cam_number} MAE={MAE}')
        plt.show()
