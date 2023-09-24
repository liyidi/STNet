import random
import cv2
import numpy as np
from tools import ops
from visualnet.visual_pre import TrackerSiamFC

random.seed(1)

def indexSwich(annoFromeOne):
     "1-index to 0-index,[x,y,w,h]"
     if len(annoFromeOne.shape) == 1:# ref_anno dimention = 1
         annoFromeOne[0] = annoFromeOne[0] - 1
         annoFromeOne[1] = annoFromeOne[1] - 1
     else:
         annoFromeOne[:,0] = annoFromeOne[:, 0] - 1
         annoFromeOne[:,1] = annoFromeOne[:, 1] - 1
     annoFromZero = annoFromeOne
     return annoFromZero


def getSample(img, box):
    sampleSize = 120  # n: size of each training sample
    addBorder = 400  # N: size of the square img
    # 1.box-->large box
    for i in range(2,100):
        boxLarge = np.array([
            box[0] - box[2] / i,
            box[1] - box[3] / i,
            box[2] + box[2]*2*(1/i) ,
            box[3] + box[3]*2*(1/i) ])
        if boxLarge[2]<119 and boxLarge[3]<119:
            break

    # 2.img-->square img,large box -->boxLarge2 for square img
    square_img = cv2.copyMakeBorder(
        img, int((addBorder - img.shape[0]) / 2), int((addBorder - img.shape[0]) / 2),
        int((addBorder - img.shape[1]) / 2), int((addBorder - img.shape[1]) / 2),
        cv2.BORDER_CONSTANT, value=np.mean(img, axis=(0, 1)))
    bw = (addBorder - img.shape[1]) / 2
    bh = (addBorder - img.shape[0]) / 2
    boxLarge2 = np.array([
        boxLarge[0] + bw,
        boxLarge[1] + bh,
        boxLarge[2],
        boxLarge[3]])

    # 3.get a region
    regionLeftUp = np.array([
        boxLarge2[0] - (sampleSize - boxLarge2[2]),
        boxLarge2[1] - (sampleSize - boxLarge2[3])])
    regionRightDown = np.array([
        boxLarge2[0],
        boxLarge2[1]])
    if regionLeftUp[0] < 0: regionLeftUp[0] = 0
    if regionLeftUp[1] < 0: regionLeftUp[0] = 0
    if regionRightDown[0] + sampleSize > addBorder:
        regionRightDown[0] = addBorder - sampleSize
    if regionRightDown[1] + sampleSize > addBorder:
        regionRightDown[1] = addBorder - sampleSize
    # 4.get a random point in the region
    new_x = random.uniform(regionLeftUp[0], regionRightDown[0])
    new_y = random.uniform(regionLeftUp[1], regionRightDown[1])
    sample_img_box = np.array([
        new_x, new_y,
        sampleSize, sampleSize])

    # 5.sample_gtboxlarge in sample_img, sample_face in sample_img
    sample_img = square_img[int(new_y): int(new_y + sampleSize),
                 int(new_x): int(new_x + sampleSize), :]
    sample_gtboxlarge = np.array([
        boxLarge2[0] - new_x, boxLarge2[1] - new_y,
        boxLarge2[2], boxLarge2[3]])

    sample_facebox = np.array([sample_gtboxlarge[0] + box[2] / i,
                            sample_gtboxlarge[1] + box[3] / i,
                            box[2], box[3]])

    return sample_img, sample_img_box, sample_facebox


tracker = TrackerSiamFC(device_id = 0, net_path='.../visualnet_e50.pth')
def vi_observ(examplar, refGT, sample_img):
    boxInSample, re_boxInSample, scale_id = tracker.observ(examplar, refGT, sample_img)
    return boxInSample, re_boxInSample, scale_id


examplar_sz = 127
def getExamplar(ref_file, box): #box:ref_anno 0-index,[x,y,w,h]
    img = ops.read_image(ref_file)
    box = np.array([
        box[1] + box[3] / 2,
        box[0] + box[2] / 2,
        box[3], box[2]], dtype=np.float32)
    z_center, target_sz = box[:2], box[2:]
    # exemplar and search sizes
    z_sz = np.max(target_sz)
    avg_color = np.mean(img, axis=(0, 1))
    # z: examplar image, z_crop: ref_img
    z, z_crop = ops.crop_and_resize(
        img, z_center, z_sz,
        out_size=examplar_sz,
        border_value=avg_color)
    return z, z_crop

def getExamplar2(img, box): #box:ref_anno 0-index,[x,y,w,h]
    examplar_sz = 127
    box = np.array([
        box[1] + box[3] / 2,
        box[0] + box[2] / 2,
        box[3], box[2]], dtype=np.float32)
    z_center, target_sz = box[:2], box[2:]
    # exemplar and search sizes
    z_sz = np.max(target_sz)
    avg_color = np.mean(img, axis=(0, 1))
    # z: examplar image, z_crop: ref_img
    z, z_crop = ops.crop_and_resize(
        img, z_center, z_sz,
        out_size=examplar_sz,
        border_value=avg_color)
    return z, z_crop
