import numpy as np
import wave
from collections import namedtuple
import scipy.signal as signal
import glob
import scipy.io as scio

configCFGAttr = ['audiolaglist','Blist','Ilist','startGTlist','endGTlist','startFRlist','endFRlist',\
                 'map_num','m1','m2','fs','c','nw','inc','winfunc']
configGCF = namedtuple('configGCF', configCFGAttr)

camClsAttr = ['Pmat','K','alpha_c','kc']
camCls = namedtuple('camCls', camClsAttr)
def cfgSeqSet(sequence):
    if sequence == 'seq01-1p-0000':
        cfg = {
            'audiolaglist': [5.84, 6.08, 7.24],
            'Blist': [1.5, 2.75, 2.3],
            'Ilist': [0.35, 0.25, 0.35],
            'startGTlist': [1, 1, 1],
            'endGTlist': [2248, 2248, 2248],
            'startFRlist': [109, 103, 74],#1-index
            'endFRlist': [5314, 5308, 5279]
        }

    elif sequence == 'seq02-1p-0000':
        cfg = {
            'audiolaglist': [7.88, 6.08, 7.24],
            'Blist': [1.5, 2.75, 2.3],
            'Ilist': [0.35, 0.25, 0.35],
            'startGTlist': [1, 1, 1],
            'endGTlist': [2407, 2407, 2407],
            'startFRlist': [143, 183, 104],#1-index
            'endFRlist': [4425, 4465, 4386]
        }

    elif sequence == 'seq03-1p-0000':
        cfg = {
            'audiolaglist': [5.96, 4.96, 7.00],
            'Blist': [1.5, 2.75, 2.3],
            'Ilist': [0.35, 0.25, 0.35],
            'startGTlist': [1, 1, 1],
            'endGTlist': [2636, 2636, 2636],
            'startFRlist': [243, 268, 217],  # 1-index
            'endFRlist': [5742, 5767, 5716]
        }

    elif sequence == 'seq08-1p-0100':
        cfg = {
            'audiolaglist': [4.8, 5.6, 5.24],
            'Blist': [1.5, 2.5, 2.1],
            'Ilist': [0.35, 0.25, 0.3],
            'startGTlist': [14, 28, 18],
            'endGTlist': [495, 496, 504],
            'startFRlist': [34, 28, 27],
            'endFRlist': [515, 496, 513]
        }
    elif sequence == 'seq11-1p-0100':
        cfg = {
            'audiolaglist': [6.64, 6.72, 5.52],
            'Blist': [1.5, 2.5, 2],
            'Ilist': [0.35, 0.2, 0.3],
            'startGTlist': [69, 70, 70],
            'endGTlist': [547, 545, 547],
            'startFRlist': [71, 70, 101],#1-index
            'endFRlist': [549, 545, 578]
        }
    elif sequence == 'seq12-1p-0100':
        cfg = {
            'audiolaglist': [0.44, -0.92, 0.52],
            'Blist': [1.5, 2.5, 2],
            'Ilist': [0.35, 0.2, 0.3],
            'startGTlist': [88,88,105],
            'endGTlist': [1148,1148,1148],
            'startFRlist': [90,124,105],
            'endFRlist': [1150,1184,1148]
        }
    elif sequence == 'seq18-2p-0101':
        cfg = {
            'audiolaglist': [2.32, 1.80, 3.32],
            'Blist': [1.5, 2.5, 2],
            'Ilist': [0.35, 0.25, 0.3],
            'startGTlist': [0, 0, 0],
            'endGTlist': [0, 0, 0],
            'startFRlist': [125, 138, 100],
            'endFRlist': [1326, 1339, 1301]
        }
    elif sequence == 'seq24-2p-0111':
        cfg = {
            'audiolaglist': [21/25, -7/25, 1.6],
            'Blist': [1.5, 2.5, 2],
            'Ilist': [0.35, 0.25, 0.3],
            'startGTlist': [0,223,0],
            'endGTlist': [0,473,0],
            'startFRlist': [0, 270, 0],
            'endFRlist': [0, 520, 0]
        }
    elif sequence == 'seq45-3p-1111':
        cfg = {
            'audiolaglist': [2+12/25, 18/25, 3+18/25],
            'Blist': [1.5, 2.5, 2],
            'Ilist': [0.35, 0.25, 0.3],
            'startGTlist': [0,0,187],
            'endGTlist': [0,0,687],
            'startFRlist': [0, 0, 230],
            'endFRlist':  [0, 0, 730]
        }

    cfg['map_num'] = 9
    cfg['m1'] = 14
    cfg['m2'] = 5
    cfg['fs'] = 16000
    cfg['c'] = 340
    cfg['nw'] = 640
    cfg['inc'] = 320
    cfg['winfunc'] = signal.hamming(cfg['nw'])
    cfgGCF = configGCF(**cfg)
    return cfgGCF

def enframe(signal, nw, inc, winfunc):

  signal_length=len(signal)
  if signal_length<=nw:
    nf=1
  else:
    nf=int(np.ceil((1.0*signal_length-nw+inc)/inc))

  pad_length=int((nf-1)*inc+nw)
  zeros=np.zeros((pad_length-signal_length,))
  pad_signal=np.concatenate((signal,zeros))
  indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T
  indices=np.array(indices,dtype=np.int32)
  frames=pad_signal[indices]
  win=np.tile(winfunc,(nf,1))
  return frames*win


class DataMain:
    def __init__(self,cfgGCF):
        self.audio = []
        self.cfgGCF = cfgGCF

    def readSynAudio(self, dataPath, audiolag):
        f = wave.open(dataPath, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)
        # synchronism
        if audiolag >= 0:
            wave_data = np.fromstring(str_data, dtype=np.short)[int(audiolag * framerate):]
        elif audiolag< 0:
            wave_org = np.fromstring(str_data, dtype=np.short)
            wave_zero = np.zeros((abs(int(audiolag * framerate)),))
            wave_data = np.concatenate((wave_zero,wave_org),axis = 0)
        wave_data = wave_data * 1.0 / (max(abs(wave_data)))
        return wave_data

    def readAudio(self, dataPath, cam_number):
        audiolag = self.cfgGCF.audiolaglist[cam_number - 1]
        frameOrg = self.readSynAudio(dataPath, audiolag)
        frameWin = enframe(frameOrg, self.cfgGCF.nw, self.cfgGCF.inc, self.cfgGCF.winfunc)
        return frameWin

    def loadAudio(self, datasetPath, sequence, cam_number):
        for arrayNum  in range(1,3):
            for micNum in range(1,9):
                dataPath = f'{datasetPath}/{sequence}/{sequence}_array{arrayNum}_mic{micNum}.wav'
                frameWin = self.readAudio(dataPath, cam_number)
                self.audio.append(frameWin)

    def loadmicPos(self, datasetPath):
        dataPath = f'{datasetPath}/gt.mat'
        gtDict = scio.loadmat(dataPath)
        micPosData = gtDict['gt'][0][0][7]
        self.micPos = np.transpose(micPosData)

    def loadImgPath(self, datasetPath, sequence, cam_number):
        img_path = f'{datasetPath}/{sequence}/{sequence}_cam{cam_number}_jpg/img/'
        img_files = sorted(glob.glob(img_path + '*.jpg'))
        self.imgPath = img_files

    def loadCamAlign(self,datasetPath):
        dataPath = f'{datasetPath}/cam.mat'
        data = scio.loadmat(dataPath)['cam'][0]
        cam = {
            'Pmat': np.concatenate(([data[0][0]], [data[1][0]], [data[2][0]]), axis=0),
            'K': np.concatenate(([data[0][1]], [data[1][1]], [data[2][1]]), axis=0),
            'alpha_c': np.concatenate(([data[0][2]], [data[1][2]], [data[2][2]]), axis=0),
            'kc': np.concatenate(([data[0][3]], [data[1][3]], [data[2][3]]), axis=0),
        }

        # self.cam = namedtuple('cam',cam.keys())(**cam)
        self.cam = camCls(**cam)
        dataPath = f'{datasetPath}/rigid010203.mat'
        data = scio.loadmat(dataPath)['rigid'][0]
        self.align_mat = data[0][1]
    def loadGT3D(self, datasetPath, sequence):
        dataPath = f'{datasetPath}/{sequence}/{sequence}-person1_myDataGT3D.mat'
        GT3D = scio.loadmat(dataPath)
        GT3DData = GT3D['DataGT3D']
        self.GT3D = GT3DData
    def loadGT2D(self, datasetPath, sequence, cam_number):
        dataPath = f'{datasetPath}/{sequence}/{sequence}_cam{cam_number}_jpg/{sequence}_cam{cam_number}-person1_GT2D.txt'
        GT2DData = np.loadtxt(dataPath)
        self.GT2D = GT2DData

def InitGCF(datasetPath, sequence, cam_number):
    CFGforGCF = cfgSeqSet(sequence)
    DATA = DataMain(CFGforGCF)
    DATA.loadAudio(datasetPath, sequence, cam_number)
    DATA.loadmicPos(datasetPath)
    DATA.loadImgPath(datasetPath, sequence, cam_number)
    DATA.loadCamAlign(datasetPath)
    DATA.loadGT3D(datasetPath, sequence)
    DATA.loadGT2D(datasetPath, sequence, cam_number)
    print('initialfinished' )
    return DATA, CFGforGCF
