# STNet
STNet: Deep Audio-Visual Fusion Network for Robust Speaker Tracking

## Requirements
- Python 3.6, PyTorch 1.7, opencv-python, numpy, scipy, matplotlib


## Data Preparation

* AV16.3: the original dataset, available at > http://www.glat.info/ma/av16.3/.
* CAV3D DATASET: Co-located Audio-Visual streams with 3D tracks, available at > https://speechtek.fbk.eu/cav3d-dataset/
* To construct the camera model, you need to download the camera parameters from AV16.3, including the `cam.mat` and `rigid010203.mat`.
* For training the STNet, you should prepare the audio-visual samples: 
    * `tools/prepareAudio.py,prepare_gccphat.py`: preprocess the audio signals.
    * `tools/prepareSample.py,prepareAuSample`: collect image samples and corresponding audio samples.

## Download Pretrained Weight

You can download the pre-trained weigt for STNet which is used in the main paper from follow url:

* filename: model_stnet_ep50.pth (64MB), visualnet_pre.pth (8.9MB), GCFnet_pre.pth (13.7MB)
* https://drive.google.com/drive/folders/1aQ2ZiHiQcxGktMLz8omLXyi9khuNFpfM?usp=sharing

## Descriptions

#### Train 

* To train STNet, you need to download **seq01, 02, 03**, and camera parameters from the AV16.3 dataset. Use the preprocessing files provided in the `tools` for audio and video synchronization, audio preprocessing, and prepare audio-visual sample pairs for training.
* After preparing the dataset and training samples, set the path of the correct image samples and GCF samples path in `models/my_dataset.py` and run `train.py`.


#### Tracking

* SOT sequences are from **seq08,11,12** in AV16.3 dataset. MOT sequences are from **seq24,25,30**.
* Run `tracking/test_tracking_SOT/MOT.py` to track.


#### Audio measurement
* The stGCF and improved vgGCF method are provided in `GCF/GCF_extract_stGCF,vg.py`. 
* Run `GCF/stGCF,vgGCF.py` to evluate the AO methods.

#### visual measurement
* A pre-trained Siamese network is employed to extract the visual features.
The implementation of SiamFC tracker is described in the paper: [Fully-Convolutional Siamese Networks for Object Tracking](https://www.robots.ox.ac.uk/~luca/siamese-fc.html).
* Run `visualnet/VO_MOT.py` to evluate the VO methods.
