import numpy as np
from tools import ops
import cv2
from itertools import *
class getGCF(object):

    def GCFextract(self, DATA, GCC, fa, cam_number, face, spl_box):

        #step1: generate 2D sample points
        w = int(face[2] / 3)
        h = int(face[3] / 3)
        grid_x, grid_y = np.mgrid[face[0]:face[0]+face[2]:w*1j, face[1]:face[1]+face[3]:h*1j]
        sample2d = np.concatenate((grid_x.reshape(w*h,-1),grid_y.reshape(w*h,-1)),axis = 1)

        #step2: generate 3D sample points
        sample3d = np.zeros((DATA.cfgGCF.map_num, sample2d.shape[0],3))
        p2d = np.zeros(shape=(sample2d.shape[0], 3))
        p2d[:, :-1] = sample2d
        for i in range(DATA.cfgGCF.map_num):
            z = DATA.cfgGCF.Blist[cam_number - 1] + DATA.cfgGCF.Ilist[cam_number - 1] * (i)
            p3d = ops.p2dtop3d_2(p2d, z, DATA.cam, DATA.align_mat, cam_number)
            sample3d[i,:] = p3d.T
        #find outside samples
        outlier = [[],[]]
        for g in range(DATA.cfgGCF.map_num):
            for i in range(sample3d[0].shape[0]):
                p = sample3d[g][i]
                if p[0] <= -1.8  or p[0] >= 1.8 or \
                   p[1] <= -7.2  or p[1] >= 2   or \
                   p[2] <= -0.04 or p[2] >= 1.56:
                    outlier[0].append(g)
                    outlier[1].append(i)
        #step3: calculate tau in tau3d
        pairNum = (1 + len(DATA.audio) - 1) * (len(DATA.audio) - 1) / 2
        tau3d = np.zeros((DATA.cfgGCF.map_num, int(pairNum), sample2d.shape[0]))
        interp = 1
        max_shift = int(interp * DATA.audio[0].shape[1])
        for g in range(DATA.cfgGCF.map_num):
            tau3dlist = np.zeros(shape=(int(pairNum), sample2d.shape[0]))
            t = 0
            for mici in range(len(DATA.audio)):
                di = np.sqrt(np.sum(np.asarray(DATA.micPos[mici] - sample3d[g]) ** 2, axis=1))
                for micj in range(mici + 1, len(DATA.audio)):
                    dj = np.sqrt(np.sum(np.asarray(DATA.micPos[micj] - sample3d[g]) ** 2, axis=1))
                    tauijk = (di - dj) / DATA.cfgGCF.c
                    taun = np.transpose(tauijk * DATA.cfgGCF.fs)
                    taun = np.rint(taun * interp + max_shift)
                    tau3dlist[t, :] = taun
                    t = t + 1
            tau3d[g, :] = tau3dlist
        # step4:[fa-m3,fa]
        rGCF, rGCFmax = cal_rGCFmax(DATA, sample2d,  tau3d, GCC, outlier, fa)

        #step5:find top-3's t and depth, get indes: max_t_ind, max_d_ind
        top_num = 3
        max_t_ind = np.argpartition(rGCFmax, -3)[-3:]
        max_d_ind  = np.zeros(shape = top_num)
        pmax = []
        for i in range(top_num):
            ind = np.unravel_index(rGCF[max_t_ind[i]].argmax(), rGCF[max_t_ind[i]].shape)
            max_d_ind[i] = ind[0]
            loc = sample3d[ind[0]][ind[1]]
            pmax.append(loc)

    ####step6: caculate gcf of top-3's t and depth, shape is same as response_box
        #step6-1: generate 2D sample points
        w = int(spl_box[2]/3)
        h = int(spl_box[3]/3)
        grid_x, grid_y = np.mgrid[spl_box[0]:spl_box[0]+spl_box[2]:w*1j, spl_box[1]:spl_box[1]+spl_box[3]:h*1j]
        sample2d = np.concatenate((grid_x.reshape(w*h,-1),grid_y.reshape(w*h,-1)),axis = 1)

        #step6-2: generate 3D sample points
        sample3d2 = np.zeros((top_num, sample2d.shape[0], 3))
        p2d = np.zeros(shape=(sample2d.shape[0], 3))
        p2d[:, :-1] = sample2d
        for i in range(top_num):
            d = max_d_ind[i]
            z = DATA.cfgGCF.Blist[cam_number - 1] + DATA.cfgGCF.Ilist[cam_number - 1] * (d)
            p3d = ops.p2dtop3d_2(p2d, z, DATA.cam, DATA.align_mat, cam_number)
            sample3d2[i, :] = p3d.T
    #####find outside samples
        outlier = [[], []]
        for g in range(top_num):
            for i in range(sample3d2[0].shape[0]):
                p = sample3d2[g][i]
                if p[0] <= -1.8 or p[0] >= 1.8 or \
                        p[1] <= -7.2 or p[1] >= 2 or \
                        p[2] <= -0.04 or p[2] >= 1.56:
                    outlier[0].append(g)
                    outlier[1].append(i)

    #####step6-3: tau3d
        pairNum = (1+len(DATA.audio)-1)*(len(DATA.audio)-1)/2
        tau3d = np.zeros((top_num, int(pairNum), sample2d.shape[0]))
        interp = 1
        max_shift = int(interp * DATA.audio[0].shape[1])
        for g in range(top_num):
            tau3dlist = np.zeros(shape = (int(pairNum),sample2d.shape[0]))
            t = 0
            for mici in range(len(DATA.audio)):
                di = np.sqrt(np.sum(np.asarray( DATA.micPos[mici]- sample3d2[g])**2, axis=1))
                for micj in range(mici + 1, len(DATA.audio)):
                    dj = np.sqrt(np.sum(np.asarray( DATA.micPos[micj]- sample3d2[g])**2, axis=1))
                    tauijk = (di - dj)/DATA.cfgGCF.c
                    taun = np.transpose(tauijk * DATA.cfgGCF.fs)
                    taun = np.rint(taun * interp + max_shift)
                    tau3dlist[t,:] = taun
                    t = t + 1
            tau3d[g,:] = tau3dlist

        # step6-4:gcfmap
        reGCFmaporg =cal_GCF2_v3(top_num, sample2d, tau3d, max_t_ind, fa, GCC, outlier, grid_x, pairNum)
        reGCFmap = np.zeros(shape = (top_num, int(spl_box[2]),int(spl_box[3])))
        for i in range(top_num):
            reGCFmap[i] = cv2.resize(reGCFmaporg[i],(int(spl_box[2]),int(spl_box[3])))#resize to spl_box shape

        return reGCFmaporg, max_d_ind



    def au_observ(self, DATA, GCC, cam_number, frameNum, box, spl_box):
        #box, spl_box are in img coordinate
        face = box
        fr = frameNum
        fa = int(2*fr-2)
        gcfmap = self.GCFextract(DATA, GCC, fa, cam_number, face, spl_box)
        return gcfmap

def cal_tau(DATA, sample2d, sample3d):
    pairNum = (1+len(DATA.audio)-1)*(len(DATA.audio)-1)/2
    tau3d = np.zeros((DATA.cfgGCF.map_num, int(pairNum), sample2d.shape[0]))
    interp = 1
    max_shift = int(interp * DATA.audio[0].shape[1])
    for g in range(DATA.cfgGCF.map_num):
        tau3dlist = np.zeros(shape = (int(pairNum),sample2d.shape[0]))
        t = 0
        for mici in range(len(DATA.audio)):
            di = np.sqrt(np.sum(np.asarray( DATA.micPos[mici]- sample3d[g])**2, axis=1))
            for micj in range(mici + 1, len(DATA.audio)):
                dj = np.sqrt(np.sum(np.asarray( DATA.micPos[micj]- sample3d[g])**2, axis=1))
                tauijk = (di - dj)/DATA.cfgGCF.c
                taun = np.transpose(tauijk * DATA.cfgGCF.fs)
                taun = np.rint(taun * interp + max_shift)
                tau3dlist[t,:] = taun
                t = t + 1
        tau3d[g,:] = tau3dlist
    return tau3d


def cal_rGCFmax(DATA, sample2d, tau3d, GCC, outlier,fa):
    m3 = 15
    rGCF = np.zeros(shape=(m3, DATA.cfgGCF.map_num, sample2d.shape[0]))
    rGCFmax = np.zeros(shape=m3)
    gccx = get_chain_np(tau3d.shape[2], 0, tau3d.shape[1], tau3d.shape[0])
    gccy = tau3d.reshape(-1).astype(int)

    for i in range(m3):
        fn = fa - i
        cc = GCC[fn]
        rPHAT= cc[gccx, gccy].reshape(*tau3d.shape)
        rGCForg = np.mean(rPHAT, axis=1)
        rGCForg[outlier[0], outlier[1]] = 0  # set outside sample as 0
        rGCF[i] = rGCForg
        rGCFmax[i] = np.max(rGCF[i]) * (1 - i * 0.0125)  ###punishment for time lag
    return rGCF, rGCFmax


def cal_GCF2_v3(top_num, sample2d, tau3d, max_t_ind, fa, GCC, outlier, grid_x, pairNum):
    reGCF = np.zeros(shape=(top_num, sample2d.shape[0]))
    for i in range(top_num):
        t = max_t_ind[i]
        fn = fa - t
        rPHAT = np.zeros(shape=(sample2d.shape[0], int(pairNum)))
        for j in range(tau3d.shape[1]):
            cc = GCC[fn][j]
            indexij = tau3d[i][j]
            rPHAT[:, j] = cc[indexij.astype(int)]
        retGCF = np.mean(rPHAT, axis=1)
        reGCF[i] = retGCF * (1 - t * 0.0125)
    reGCF[outlier[0], outlier[1]] = 0  # set outside sample as 0
    reGCFmaporg = reGCF.reshape(-1, grid_x.shape[0], grid_x.shape[1], order='F')  # reshape to w*h
    return reGCFmaporg

def get_chain(repeat_times , range_start , range_end , total_repeat_times ):
    p = list(chain(*[[x] * repeat_times for x in range(range_start, range_end)])) * total_repeat_times
    p = np.array(p)
    return p

def get_chain_np(repeat_times , range_start , range_end , total_repeat_times ):
    p2 = np.tile(np.repeat(np.arange(range_start, range_end),
                           repeat_times), (1, total_repeat_times)).flatten()
    return p2