import numpy as np
from tools import ops
import os
import pickle
seqList = []
datasetPath = '.../STNet'
m3 = 15
interp=1
audioData = ops.loadAudio(seqList, datasetPath)
for sequence in seqList:
    GCCDic = {}
    for cam_number in range(1, 4):
        DATA = audioData[f'{sequence}_cam{cam_number}']
        startfr, endfr = ops.getSE(sequence, cam_number)#1-index
        startfr = startfr -1
        startfa = int(2 * startfr - 2) - m3
        endfr = endfr -1
        endfa = int(2 * endfr - 2)
        GCCPHATlist = list()
        for fa in range(len(DATA.audio[0])):
            GCCPHAT = list()
            if fa>= startfa and fa<= endfa:
                for mici in range(len(DATA.audio)):
                    for micj in range(mici + 1, len(DATA.audio)):
                        sig = DATA.audio[mici][fa]
                        refsig = DATA.audio[micj][fa]
                        cc, _ = ops.gcc_phat(sig, refsig, fs=DATA.cfgGCF.fs, max_tau=None, interp=interp)
                        GCCPHAT.append(cc)
                        print(f'calculate GCC for {sequence}_cam{cam_number}: {fa}/{len(DATA.audio[0])}')
            GCCPHATlist.append(np.array(GCCPHAT))
        data = {f'{sequence}_cam{cam_number}': GCCPHATlist}
        GCCDic.update(data)
    folderPath = f'{datasetPath}/GCC'
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    output = open(f'{folderPath}/{sequence}_GCC.pkl', 'wb')
    pickle.dump(GCCDic, output)
    print(f'save _GCC.pkl for {sequence}')

