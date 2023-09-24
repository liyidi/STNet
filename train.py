from models.stnet_model import STNet
from models.my_dataset import myDataset
import numpy as np
import torch
from tools import ops
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import pickle



class trainer():
    def main(self):
        self.loaddata()
        self.setmodel()
        self.train()
    def __init__(self):
        self.device_ids = [0]
        '''set seq and path'''
        self.trainSeqList = ['seq01-1p-0000', 'seq02-1p-0000', 'seq03-1p-0000']
        self.datasetPath = '.../STNet'
        '''set log path'''
        self.date = 'stnet'
        self.BASE_DIR = '...'
        self.log_dir = os.path.abspath(os.path.join(self.BASE_DIR, 'log', "model_{0}.pth".format(self.date)))
        '''set flag : train/test'''
        self.train_flag = True
        self.saveNetwork_flag = True
        self.drawCurve_flag = True
        ops.set_seed(1)
        self.MAX_EPOCH = 50
        self.BATCH_SIZE = 16
        self.LR = 0.0001
        self.log_interval = 10
        self.val_interval = 1
        self.save_interval = 10

    def loaddata(self):
        trainList, validList = ops.splitDataset(self.datasetPath, self.trainSeqList, splitType='train&valid', trainPct=0.8)
        refpath = f'{self.datasetPath}/AVsample/ref_seq123.pkl'
        with open(refpath, 'rb') as data:
            refData = pickle.load(data)
        train_data = myDataset(dataList=trainList, refData=refData)
        valid_data = myDataset(dataList=validList, refData=refData)
        self.train_loader = DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True, drop_last= True)
        self.valid_loader = DataLoader(dataset=valid_data, batch_size=self.BATCH_SIZE, shuffle=True, drop_last= True)

    def setmodel(self):
        net_path_vi = os.path.join(self.BASE_DIR, 'siamfc','visualnet_pre.pth')
        net_path_au = os.path.join(self.BASE_DIR, 'GCF','GCFnet_pre.pth')
        net = STNet(net_path_vi = net_path_vi, net_path_au=net_path_au)
        net = torch.nn.DataParallel(net, device_ids=self.device_ids)
        self.net = net.cuda(device = self.device_ids[0])
        torch.enable_grad()
        self.lossFn1 = nn.MSELoss(reduction='mean')
        self.lossFn2 = nn.MSELoss(reduction='mean')
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.LR, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    def train(self):
        if self.train_flag:
            train_curve = list()
            valid_curve = list()
            for epoch in range(self.MAX_EPOCH):
                loss_mean = 0.
                self.net.train()
                for i, data in enumerate(self.train_loader):
                    refImg, sampleImg0, sampleImg1, sampleImg2, GCFmap, sampleFace = data
                    imgRef = Variable(torch.as_tensor(refImg, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    img0 = Variable(torch.as_tensor(sampleImg0, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    img1 = Variable(torch.as_tensor(sampleImg1, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    img2 = Variable(torch.as_tensor(sampleImg2, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    auFr = Variable(torch.as_tensor(GCFmap, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    labels = Variable(torch.as_tensor(sampleFace, dtype=torch.float32), requires_grad=False) \
                        .cuda(device=self.device_ids[0])
                    self.optimizer.zero_grad()
                    outputs, evl_factor = self.net(imgRef, img0, img1,img2, auFr)
                    loss1 = self.lossFn1(outputs, labels)
                    r = outputs.detach()
                    dist = torch.sqrt(torch.sum((r - labels) ** 2, axis=1))
                    label2 = torch.div(2, torch.exp(0.05*dist) + 1)
                    loss2 = self.lossFn2(evl_factor.squeeze(), label2)
                    loss = loss1 + loss2
                    loss.backward()
                    self.optimizer.step()

                    loss_mean += loss.item()
                    train_curve.append(loss.item())
                    if (i+1) % self.log_interval == 0:
                        loss_mean = loss_mean / self.log_interval
                        print("[{}] Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                            self.date, epoch, self.MAX_EPOCH, i+1, len(self.train_loader), loss_mean))
                        loss_mean = 0.
                self.scheduler.step()  
                # validate the model
                if (epoch+1) % self.val_interval == 0:
                    loss_val = 0.
                    with torch.no_grad():
                        for j, data in enumerate(self.valid_loader):
                            refImg, sampleImg0, sampleImg1, sampleImg2, GCFmap, sampleFace = data
                            imgRef = Variable(torch.as_tensor(refImg, dtype=torch.float32), requires_grad=True) \
                                .cuda(device=self.device_ids[0])
                            img0 = Variable(torch.as_tensor(sampleImg0, dtype=torch.float32), requires_grad=True) \
                                .cuda(device=self.device_ids[0])
                            img1 = Variable(torch.as_tensor(sampleImg1, dtype=torch.float32), requires_grad=True) \
                                .cuda(device=self.device_ids[0])
                            img2 = Variable(torch.as_tensor(sampleImg2, dtype=torch.float32), requires_grad=True) \
                                .cuda(device=self.device_ids[0])
                            auFr = Variable(torch.as_tensor(GCFmap, dtype=torch.float32), requires_grad=True) \
                                .cuda(device=self.device_ids[0])
                            labels = Variable(torch.as_tensor(sampleFace, dtype=torch.float32), requires_grad=False) \
                                .cuda(device=self.device_ids[0])
                            self.optimizer.zero_grad()
                            outputs, evl_factor = self.net(imgRef, img0, img1, img2, auFr)
                            loss1 = self.lossFn1(outputs, labels)
                            dist = torch.sqrt(torch.sum((outputs - labels) ** 2, axis=1))
                            label2 = torch.div(2, torch.exp(0.05*dist) + 1)
                            loss2 = self.lossFn2(evl_factor.squeeze(), label2)
                            loss = loss1 + loss2
                            loss_val += loss.item()
                        loss_val_epoch = loss_val / len(self.valid_loader)
                        valid_curve.append(loss_val_epoch)
                        print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                            epoch, self.MAX_EPOCH, j+1, len(self.valid_loader), loss_val_epoch))
                if (epoch + 1) % self.save_interval == 0:
                    log_dir = os.path.abspath(os.path.join(self.BASE_DIR, "log", "model_{0}_ep{1}.pth".format(self.date,epoch+1)))
                    state = {'model': self.net.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
                    torch.save(state, log_dir)
        if self.drawCurve_flag:
            train_x = range(len(train_curve))
            train_y = train_curve
            train_iters = len(self.train_loader)
            valid_x = np.arange(1, len(valid_curve)+1) * train_iters*self.val_interval
            valid_y = valid_curve
            plt.plot(train_x, train_y, label='Train')
            plt.plot(valid_x, valid_y, label='Valid')
            plt.legend(loc='upper right')
            plt.ylabel('loss value')
            plt.xlabel('Iteration')
            plt.show()
            print('end train')

if __name__ == '__main__':
    stnet = trainer()
    stnet.main()