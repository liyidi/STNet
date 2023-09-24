import os
import glob
import cv2
import numpy as np
from tools import ops
from tools import prepareTools
import time
import pandas as pd
import matplotlib.pyplot as plt
seqList = ['seq25-2p-0111','seq30-2p-1101'] #'seq24-2p-0111'
datasetPath_vi = '.../AV163'
datasetPath = '.../STNet'

#============================ tracking =========================================
total_result = list()
for sequence in seqList:
    for cam_number in range(1, 4):#(1, 4)1-index
        for person_number in range(1, 3):

            startfr, endfr = ops.getSE(sequence, cam_number)#1-index
            seq_dir = f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/img/'
            total_file = sorted(glob.glob(seq_dir + '*.jpg'))
            total_GT = np.loadtxt(f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/'
                                  f'{sequence}_cam{cam_number}-person{person_number}_GT2D.txt')
            # 1.ref(path) is from the previous frame of the startfr:startfr - 2 (0-index)
            ref_file = total_file[startfr - 2]
            # 2.ref_anno
            ref_anno = total_GT[startfr - 2]  # GT2D.txt is 1-index
            ref_anno = prepareTools.indexSwich(ref_anno)  # "1-index to 0-index,[x,y,w,h]"
            # 2-2.ref_img-->examplar(crop as square shape)
            ref_examplar, ref_img = prepareTools.getExamplar(ref_file, ref_anno)
            # 3.img_files: total path of imgs.
            img_files = total_file[startfr - 1:endfr]
            # 4.anno is [x,y,w,h] of each frame
            img_anno = total_GT[startfr - 1:endfr]
            img_anno = prepareTools.indexSwich(img_anno)  # "1-index to 0-index,[x,y,w,h]"
            img_data = list()
            for i in range(len(img_files)):
                img_org = ops.read_image(img_files[i])
                img_data.append(img_org)

        ###-----TRACKING----------
            test_curve = list()
            error_curve = list()
            evl_curve = list()
            error_total = 0
            MAE_curve = list()
            traj = list()
            seq_result = {}
            since = time.time()
            s_size = 360
            for i in range(len(img_data)):
                gt2d = np.array([img_anno[i][0]+img_anno[i][2]/2,
                                img_anno[i][1]+img_anno[i][3]/2])
                img_org = img_data[i]

                center = np.array([int(img_org.shape[0]/2), int(img_org.shape[1]/2)])
            ### crop the img_org to 120*120,center is last box
                _, sample = ops.crop_and_resize(
                    img_org, center, size = 360,
                    out_size= 360,
                    border_value= np.mean(img_org, axis=(0, 1)))
        ###VISUAL MEASUREMENT: face detection
                boxInSample, re_boxInSample, scale_id = prepareTools.vi_observ(
                    ref_examplar, ref_anno, sample)
                # ops.showRecimg(sample, boxInSample)
            ###box_in_sample-->box_in_img_org
                box = np.array([boxInSample[0] + center[1] - s_size / 2, boxInSample[1] + center[0] - s_size / 2,
                                boxInSample[2], boxInSample[3]])
                # ops.showRecimg(img_org, box)
                loc2d = np.array([box[0]+box[2]/ 2, box[1]+box[3]/ 2])
                error2d = np.sqrt(np.sum(np.asarray(loc2d - gt2d) ** 2))
                error_curve.append(error2d)
                error_total += error2d
                MAE = error_total / (i + 1)
                MAE_curve.append(MAE)
                traj.append(loc2d)

            print("seq:{} cam:{} p:{} sample:{:0>3}/{:0>4} [error2d:{:.4f} MAE:{:.4f} ]".
                      format(sequence, cam_number, person_number, i, len(img_files), error2d, MAE))
