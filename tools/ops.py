import os
import glob
import random
import math
import cv2
import numpy as np
import torch.nn as nn
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pickle
import torch
import py3nvml
random.seed(1)
def getSE(sequence,cam_number):#1-index
    if sequence == 'seq01-1p-0000':
        startFRlist = [109, 103, 74]
        endFRlist = [5314, 5308, 5279]
    elif sequence == 'seq02-1p-0000':
        startFRlist = [143, 183, 104]
        endFRlist = [4425, 4465, 4386]
    elif sequence == 'seq03-1p-0000':
        startFRlist = [243, 268, 217]
        endFRlist = [5742, 5767, 5716]
    elif sequence == 'seq11-1p-0100':
        startFRlist = [71, 70, 101]
        endFRlist = [549, 545, 578]
    elif sequence == 'seq08-1p-0100':
        startFRlist = [34, 28, 27]
        endFRlist = [515, 496, 513]
    elif sequence == 'seq12-1p-0100':
        startFRlist = [90, 124, 105]
        endFRlist = [1150, 1184, 1148]
    elif sequence == 'seq18-2p-0101':
        startFRlist = [125, 138, 100]
        endFRlist = [1326, 1339, 1301]
    elif sequence == 'seq24-2p-0111':
        startFRlist = [315, 315, 260]
        endFRlist =  [500, 528, 481]
    elif sequence == 'seq25-2p-0111':
        startFRlist = [125, 210, 80]
        endFRlist = [225, 351, 270]
    elif sequence == 'seq30-2p-1101':
        startFRlist = [128, 90, 60]
        endFRlist = [248, 195, 145]
    elif sequence == 'seq45-3p-1111':
        startFRlist = [302, 360, 360]
        endFRlist = [900, 900, 900]

    startfr, endfr = startFRlist[cam_number-1], endFRlist[cam_number-1]
    return startfr, endfr


def splitDataset(datasetPath, seqList, splitType, trainPct=0.9):
    if splitType == 'train&valid':
        dataList = list()
        for sequence in seqList:
            for cam_number in range(1,4):
                folderPath = f'{datasetPath}/AVsample/imgSample/{sequence}_cam{cam_number}/'
                sampleList = sorted(glob.glob(folderPath + '*.pkl'))
                for sample in sampleList:
                    dataList.append(sample)
        random.shuffle(dataList)
        ind = round(len(dataList) * trainPct)
        trainList = dataList[0:ind]
        validList = dataList[ind:]
        return trainList, validList
    elif splitType == 'test':
        dataList = list()
        for sequence in seqList:
            for cam_number in range(3,4):
                folderPath = f'{datasetPath}/AVsample/imgSample/{sequence}_cam{cam_number}/'
                sampleList = sorted(glob.glob(folderPath + '*.pkl'))
                for sample in sampleList:
                    dataList.append(sample)
        return dataList


def loadAudio(seqList, datasetPath_save):
    audioDataDic = {}
    for sequence in seqList:
        folderPath = f'{datasetPath_save}/audio/{sequence}'
        pkl_file = open(f'{folderPath}/{sequence}_audio.pkl', 'rb')
        data = pickle.load(pkl_file)
        audioDataDic.update(data)
        print(f'load audio data for {sequence}')
    return audioDataDic

def loadGCC(sequence, datasetPath_save):
    pkl_file = open(f'{datasetPath_save}/GCC/{sequence}_GCC.pkl', 'rb')
    data = pickle.load(pkl_file)
    print(f'load GCC data for {sequence}')
    return data

def loadaudioDATA(sequence, datasetPath_save):
    pkl_file = open(f'{datasetPath_save}/audio/{sequence}/{sequence}_audio.pkl', 'rb')
    data = pickle.load(pkl_file)
    print(f'load audioDATA data for {sequence}')
    return data

def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img

def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def crop_and_resize(img, center, size, out_size,
                    border_value,
                    border_type=cv2.BORDER_CONSTANT,
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch_crop = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch_crop, (out_size, out_size),
                       interpolation=interp)

    return patch, patch_crop

def showRecimg(img, box):
    plt.imshow(img)
    currentAxis = plt.gca()
    rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=3, edgecolor='r', facecolor='none')
    currentAxis.add_patch(rect)
    # plt.savefig("test.png")
    plt.show()

def showData(data):
    plt.imshow(data)
    plt.show()

def show_response(response, x_crop):
    response = cv2.resize(response, (x_crop.shape[1], x_crop.shape[0]))
    response_nor = cv2.normalize(response, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    heatmap = np.uint8(response_nor)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    frame_map = x_crop * 0.55 + heatmap * 0.45
    frame_map = frame_map.astype(np.uint8)

    cv2.namedWindow('response', 0)
    cv2.resizeWindow('response', 300, 300)
    cv2.imshow('response', frame_map)
    cv2.waitKey(10)

def show_heatmap(a,name):

    response_nor = cv2.normalize(a, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    heatmap = np.uint8(response_nor)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    cv2.resizeWindow(name, 300, 300)
    cv2.imshow(name, heatmap)
    cv2.waitKey(10)


def sample2img(boxInSample, new_x, new_y, bw = 20, bh = 56):
    '''[x,y, w,h] box in sample coordinate -->box in img_org coordinate
    [x,y] ---> [x+new_x-bw, y+new_y-bh]
    '''
    x = boxInSample[0]
    y = boxInSample[1]
    x_img = x+new_x-bw
    y_img = y+new_y-bh
    boxInImg = np.array([
        x_img, y_img, boxInSample[2], boxInSample[3]
    ])
    return boxInImg

def square2img(sampleImgBox, bw=20, bh=56):
    x = sampleImgBox[0]
    y = sampleImgBox[1]
    x_img = x - bw
    y_img = y - bh
    boxInImg = np.array([
        x_img, y_img, sampleImgBox[2], sampleImgBox[3]
    ])
    return boxInImg


def comp_distortion_oulu(xd,k):
    k1 = k[0]
    k2 = k[1]
    k3 = k[4]
    p1 = k[2]
    p2 = k[3]
    x = xd
    for _ in range(20):
        r_2 = np.sum(np.square(x))
        k_radial = 1 + k1 * r_2 + k2 * math.pow(r_2,2) + k3 * math.pow(r_2,2)
        delta_x = np.array([2*p1*x[0]*x[1] + p2*(r_2 + 2*math.pow(x[0],2)),
                            p1 * (r_2 + 2*math.pow(x[1],2))+2*p2*x[0]*x[1]   ])
        x = (xd - delta_x)/k_radial
    return x

pow_vec = np.vectorize(math.pow)

def undoradial(x_kk, K, kc, alpha_c):
    cc = K[0, 2]
    cc = np.append(cc, K[1, 2])#cc = K[[0,1], 2]
    fc = K[0, 0]
    fc = np.append(fc, K[1, 1])#fc = K[[0,1], [0,1]]
    # First: Subtract principal point, and divide by the focal length:
    x_distort = np.array([(x_kk[0] - cc[0]) / fc[0], (x_kk[1] - cc[1]) / fc[1]])
    # Second: compensate for skew
    x_distort[0] = x_distort[0] - alpha_c * x_distort[1]
    if np.linalg.norm(kc) != 0:
        xn = comp_distortion_oulu(x_distort, kc)
    else:
        xn = x_distort
    xl = np.dot(K, np.append(xn, 1))
    return xl

def p2dtop3d(p2d, z, cam, align_mat, cam_number):
    kc = np.append(cam.kc[cam_number - 1], 0)
    renew_x = undoradial(p2d, cam.K[cam_number - 1], kc, cam.alpha_c[cam_number - 1])
    renew_xz = renew_x * z
    rexyz = np.append(np.dot(np.linalg.inv(cam.Pmat[cam_number - 1][:, 0:3]),
                             (renew_xz - cam.Pmat[cam_number - 1][:, 3])), 1)
    rep3d = np.dot(align_mat, rexyz)
    rep3d = rep3d[0:3]
    return rep3d
'''
vectorization of function: p2dtop3d
input: p2d size [[x1,y1,1],[x2,y2,1]...]
'''
def p2dtop3d_2(p2d, z, cam, align_mat, cam_number):
    kc = np.append(cam.kc[cam_number - 1], 0)
    renew_x = undoradial2(p2d, cam.K[cam_number - 1], kc, cam.alpha_c[cam_number - 1])
    renew_xz = renew_x * z
    rexyz = np.append(np.dot(np.linalg.inv(cam.Pmat[cam_number - 1][:, 0:3]),
                             (renew_xz -np.array([cam.Pmat[cam_number - 1][:, 3]]).T)), np.ones([1,renew_xz.shape[1]]),axis = 0)
    rep3d = np.dot(align_mat, rexyz)
    rep3d = rep3d[0:3]
    return rep3d

def undoradial2(x_kk, K, kc, alpha_c):
    cc = K[[0,1], 2]
    fc = K[[0,1], [0,1]]
    # First: Subtract principal point, and divide by the focal length:
    x_distort = np.array([(x_kk[:,0] - cc[0]) / fc[0], (x_kk[:,1] - cc[1]) / fc[1]])
    # Second: compensate for skew
    x_distort[0] = x_distort[0] - alpha_c * x_distort[1]
    if np.linalg.norm(kc) != 0:
        xn = comp_distortion_oulu2(x_distort, kc)
    else:
        xn = x_distort
    xl = np.dot(K, np.append(xn, np.ones([1,xn.shape[1]]),axis = 0))
    return xl

def comp_distortion_oulu2(xd,k):
    k1 = k[0]
    k2 = k[1]
    k3 = k[4]
    p1 = k[2]
    p2 = k[3]
    x = xd
    for kk in range(20):
        r_2 = np.sum(np.square(x),axis = 0)
        k_radial = 1 + k1 * r_2 + k2 * np.square(r_2) + k3 * np.square(r_2)
        delta_x = np.array([2*p1*x[0]*x[1] + p2*(r_2 + 2*np.square(x[0])),
                            p1 * (r_2 + 2*np.square(x[1]))+2*p2*x[0]*x[1]   ])
        x = (xd - delta_x)/k_radial
    return x

def gcc_phat(sig, refsig, fs, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]
    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    return cc, max_shift

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_free_gpu(num_gpus=1, gpu_select=None, least_mem=2e9):
    nvml = py3nvml.py3nvml
    nvml.nvmlInit()
    deviceCount = nvml.nvmlDeviceGetCount()

    info = {}
    free = []
    for i in range(deviceCount):
        if (not gpu_select) or \
                (gpu_select and (i in gpu_select)):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            info[i] = {
                'name': nvml.nvmlDeviceGetName(handle),
                'memory': nvml.nvmlDeviceGetMemoryInfo(handle)
            }
            if info[i]['memory'].free > least_mem:
                free.append(i)

    if len(free) >= num_gpus:
        return free[0:num_gpus]
    else:
        return []