import os
import glob
import random
import cv2
import numpy as np
from tools import ops
from tools import prepareTools
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls
from GCF.GCF_extract_vg import getGCF
from models.stnet_model import STNet

def getdata(ref_examplar, sample, GCFmap):
    minGCF = -0.01
    maxGCF = 0.50
    refImg = ref_examplar.transpose(2, 0, 1)  # output is a ndarray[W*H*C]
    sampleImg0 = cv2.resize(sample, (300, 300)
                            , interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    sampleImg1 = cv2.resize(sample, (402, 402)
                            , interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    sampleImg2 = cv2.resize(sample, (550, 550)
                            , interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

    GCF_nor = (GCFmap - minGCF) / (maxGCF - minGCF)
    GCF_nor = np.transpose(GCF_nor, (1, 2, 0))  # H*W*3
    reGCFmap = cv2.resize(GCF_nor, (402, 402), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

    imgRef = Variable(torch.as_tensor(refImg, dtype=torch.float32).unsqueeze(0), requires_grad=False) \
        .cuda(device=device_ids[0])
    img0 = Variable(torch.as_tensor(sampleImg0, dtype=torch.float32).unsqueeze(0), requires_grad=True) \
        .cuda(device=device_ids[0])
    img1 = Variable(torch.as_tensor(sampleImg1, dtype=torch.float32).unsqueeze(0), requires_grad=True) \
        .cuda(device=device_ids[0])
    img2 = Variable(torch.as_tensor(sampleImg2, dtype=torch.float32).unsqueeze(0), requires_grad=True) \
        .cuda(device=device_ids[0])
    auFr = Variable(torch.as_tensor(reGCFmap, dtype=torch.float32).unsqueeze(0), requires_grad=False) \
        .cuda(device=device_ids[0])
    return imgRef, img0, img1, img2, auFr

date = 'stnet'
device_ids = [0]
random_seed = 1
random.seed(random_seed)
seqList = ['seq08-1p-0100', 'seq11-1p-0100','seq12-1p-0100'] #'seq08-1p-0100', 'seq11-1p-0100','seq12-1p-0100'
datasetPath_vi = '.../AV163'
datasetPath = '.../STNet'
s_size = 120
au_observe = getGCF()
'''set log path'''
BASE_DIR = '...'
log_dir = os.path.abspath(os.path.join(BASE_DIR, 'models', 'model_stnet_ep50.pth'))
# ============================ load network============================
net = STNet(net_path_vi = None, net_path_au=None)
net = torch.nn.DataParallel(net, device_ids=device_ids)
net = net.cuda(device = device_ids[0])
checkpoint = torch.load(log_dir, map_location=torch.device(f'cuda:{device_ids[0]}'))
net.load_state_dict(checkpoint['model'])
#============================ tracking =========================================
total_result = list()
for sequence in seqList:
    # load audio data
    audioDATA = ops.loadaudioDATA(sequence, datasetPath)
    GCCdata = ops.loadGCC(sequence, datasetPath)
    for cam_number in range(1, 4):#(1, 4)1-index
        GCC = GCCdata[f'{sequence}_cam{cam_number}']
        DATA = audioDATA[f'{sequence}_cam{cam_number}']

        startfr, endfr = ops.getSE(sequence, cam_number)#1-index
        seq_dir = f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/img/'
        total_file = sorted(glob.glob(seq_dir + '*.jpg'))
        total_GT = np.loadtxt(f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/'
                              f'{sequence}_cam{cam_number}_GT2D.txt')
        # 1.ref(path) is from the previous frame of the startfr
        ref_file = total_file[startfr - 2]
        # 2.ref_anno
        ref_anno = total_GT[startfr - 2]  # GT2D.txt is 1-index
        ref_anno = prepareTools.indexSwich(ref_anno)
        # 2-2.ref_img-->examplar(crop as square shape)
        ref_examplar, ref_img = prepareTools.getExamplar(ref_file, ref_anno)
        # 3.img_files: total path of imgs.
        img_files = total_file[startfr - 1:endfr]
        # 4.anno is [x,y,w,h] of each frame
        img_anno = total_GT[startfr - 1:endfr]
        img_anno = prepareTools.indexSwich(img_anno)
        # 5.load img
        img_data = list()
        for i in range(len(img_files)):
            img_org = ops.read_image(img_files[i])
            img_data.append(img_org)
    ###-----TRACKING----------
        error_curve = list()
        error_total = 0
        error3d_total = 0
        MAE_curve = list()
        seq_result = {}
        for i in range(len(img_data)):
            gt2d = np.array([img_anno[i][0]+img_anno[i][2]/2,
                            img_anno[i][1]+img_anno[i][3]/2])
            img_org = img_data[i]

            if i == 0:###first frame center = [c_y,c_x]
                center = np.array([ref_anno[1]+ref_anno[3]/2, ref_anno[0]+ref_anno[2]/2])
            else:
                center = np.array([loc2d[1], loc2d[0]])
    #crop the img_org to 120*120,center is last box
            _, sample = ops.crop_and_resize(
                img_org, center, size = 120,
                out_size= 120,
                border_value= np.mean(img_org, axis=(0, 1)))
    #VISUAL MEASUREMENT: face detection
            boxInSample, re_boxInSample, scale_id = prepareTools.vi_observ(
                ref_examplar, ref_anno, sample)
    #box_in_sample-->box_in_img_org
            box = np.array([boxInSample[0] + center[1] - s_size / 2, boxInSample[1] + center[0] - s_size / 2,
                            boxInSample[2], boxInSample[3]])
            sbox = np.array([0 + center[1] - s_size / 2, 0 + center[0] - s_size / 2,
                             s_size, s_size])
            frameNum = i + startfr - 1#0-index
    #AUDIO MEASUREMENT:
            GCFmap, depth_ind = au_observe.au_observ(DATA, GCC, cam_number, frameNum,
                                                            box=box, spl_box=sbox)
    #NETWORK
            imgRef, img0, img1,img2, auFr = getdata(ref_examplar,sample, GCFmap)
            outputs, evl_factor = net(imgRef, img0, img1,img2, auFr)
            #p_in_sample-->p_in_img_org
            output = outputs.detach().cpu().numpy()
            evl_factor = evl_factor.squeeze().detach().cpu().numpy()
            loc2d = np.array([output[0] + center[1] - s_size / 2, output[1] + center[0] - s_size / 2])
            error2d = np.sqrt(np.sum(np.asarray(loc2d - gt2d) ** 2))
            error_curve.append(error2d)
            error_total += error2d
            MAE = error_total / (i + 1)
            MAE_curve.append(MAE)
#3d loc
            depth_value, counts = np.unique(depth_ind, return_counts=True)
            depth = depth_value[counts.argmax()]
            z = DATA.cfgGCF.Blist[cam_number-1]+DATA.cfgGCF.Ilist[cam_number-1]*depth
            p2d = np.zeros(shape=(2, 3))
            p2d[:, :-1] = loc2d
            p3d = ops.p2dtop3d_2(p2d, z, DATA.cam, DATA.align_mat, cam_number)
            loc3d = p3d[:, 0]
            loc3d[2] = loc3d[2] - 0.1
            gtind = np.where(DATA.GT3D[:, cam_number + 3] == frameNum + 1)[0][0]
            gt3d = DATA.GT3D[gtind][1:4]

            err3d = np.sqrt(np.sum(np.asarray(gt3d - loc3d) ** 2))
            error3d_total += err3d
            MAE3d = error3d_total / (i + 1)

            print("seq:{} cam:{} sample:{:0>3}/{:0>4} [error2d:{:.4f} MAE:{:.4f} error3d:{:.4f} MAE3d:{:.4f}]".
                             format(sequence,cam_number,i,len(img_files), error2d, MAE, err3d, MAE3d))

        plt.plot(MAE_curve)
        plt.title('[MAE_curve][{}_cam{} MAE:{:.4f}]'.format(sequence,cam_number, MAE))
        plt.show()

        #total results:
        seq_result['seq'] = f'{sequence}_cam{cam_number}'
        seq_result['MAE'] = MAE
        seq_result['MAE3d'] = MAE3d
        total_result.append(seq_result)

#print results:
for i in range(len(total_result)):
    seq_result = total_result[i]
    print('[{} MAE:{:.4f} MAE3d:{:.4f} ]'
          .format(seq_result['seq'], seq_result['MAE'], seq_result['MAE3d']))
