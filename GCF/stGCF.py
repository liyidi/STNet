import os
import glob
import cv2
import numpy as np
from tools import ops
from tools import prepareTools
from GCF.GCF_extract_stGCF import getGCF
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls
import time
import matplotlib.pyplot as plt
seqList = ['seq12-1p-0100'] #'seq08-1p-0100', 'seq11-1p-0100'
datasetPath_vi = '.../AV163'
datasetPath = '.../STNet'
au_observe = getGCF()
total_result = list()
for sequence in seqList:
    # load audio data
    audioDATA = ops.loadaudioDATA(sequence, datasetPath)
    GCCdata = ops.loadGCC(sequence, datasetPath)
    for cam_number in range(1, 4):
        GCC = GCCdata[f'{sequence}_cam{cam_number}']
        DATA = audioDATA[f'{sequence}_cam{cam_number}']
        startfr, endfr = ops.getSE(sequence, cam_number)#1-index
        seq_dir = f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/img/'
        total_file = sorted(glob.glob(seq_dir + '*.jpg'))
        total_GT = np.loadtxt(f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/'
                              f'{sequence}_cam{cam_number}_GT2D.txt')
        # 3.img_files: total path of imgs.
        img_files = total_file[startfr - 1:endfr]
        # 4.anno is [x,y,w,h] of each frame
        img_anno = total_GT[startfr - 1:endfr]
        img_anno = prepareTools.indexSwich(img_anno)  # "1-index to 0-index,[x,y,w,h]"
        #load total img
        img_data = list()
        for i in range(len(img_files)):
            img_org = ops.read_image(img_files[i])
            img_data.append(img_org)
    ###-----TRACKING----------
        error_curve = list()
        error_total = 0
        MAE_curve = list()
        seq_result = {}
        for i in range(len(img_data)):
            gt2d = np.array([img_anno[i][0]+img_anno[i][2]/2,
                            img_anno[i][1]+img_anno[i][3]/2])
            img_org = img_data[i]
            frameNum = i + startfr - 1  # 0-index
            loc2d, GCFmap = au_observe.au_observ(img_org, DATA, GCC, cam_number, frameNum)

            error2d = np.sqrt(np.sum(np.asarray(loc2d - gt2d) ** 2))
            error_curve.append(error2d)
            error_total += error2d
            MAE = error_total / (i + 1)
            MAE_curve.append(MAE)

            ##---show GCF mean and results
            # plt.imshow(GCFmap)
            # plt.plot(loc2d[0], loc2d[1], 'r x', markersize=15)
            # plt.plot(gt2d[0], gt2d[1], 'g x', markersize=15)
            # plt.show()
            print("[{}_cam{} sample:{:0>3}/{:0>3}] [error2d:{:.4f} MAE:{:.4f}]".format(
                sequence, cam_number, i + 1, len(img_data), error2d, MAE))

        #total results:
        seq_result['seq'] = f'{sequence}_cam{cam_number}'
        seq_result['MAE'] = MAE
        total_result.append(seq_result)

print('[{} MAE:{:.4f} time:{:.0f}s fps:{:.2f} len:{:.0f} update:{:.0f} reset: {:.0f}]'.
          format(seq_result['seq'], seq_result['MAE'], seq_result['time'], seq_result['fps']))
print('avg MAE:{:.4f}'.format(np.mean([total_result[i]['MAE'] for i in range(len(total_result))])))
