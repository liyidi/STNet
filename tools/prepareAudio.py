import os
import pickle
from tools.prepareClass import InitGCF
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls

if __name__ == '__main__':
    seqList = ['seq01-1p-0000','seq02-1p-0000','seq03-1p-0000']
    datasetPath_au = '.../AV163'
    datasetPath_save = '.../STNet'

    for sequence in seqList:
        audioDataDic= {}
        for cam_number in range(1, 4):
            DATA, CFGforGCF = InitGCF(datasetPath_au, sequence, cam_number)
            audioData = {f'{sequence}_cam{cam_number}': DATA}
            audioDataDic.update(audioData)

        folderPath = f'{datasetPath_save}/audio/{sequence}'
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        output = open(f'{folderPath}/{sequence}_audio.pkl', 'wb')
        pickle.dump(audioDataDic, output)
        print(f'save audio.npz for {sequence}')
