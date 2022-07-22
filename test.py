from astropy.io import fits
from multiprocessing import Pool
from scipy.signal import medfilt2d
from tqdm import tqdm
import datetime
import glob
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import sys
import cv2
import os
import shutil
import glob
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models.detection import retinanet_resnet50_fpn
import torchvision.transforms
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from albumentations.pytorch.transforms import img_to_tensor
from albumentations import (
     Normalize, Compose, HorizontalFlip, RandomRotate90, RandomCrop, CenterCrop, Rotate,
     ElasticTransform, VerticalFlip, RandomBrightness, RandomContrast, MedianBlur, GaussNoise, BboxParams,
)

from mycommon import *
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


if len(sys.argv) != 5:
    print('usage: datadir cachedir modeldir outsolutioncsvpath')
    sys.exit(1)


datapath, cachepath, modeldir, solution_outpath = sys.argv[1:]


def get_model():
    model = retinanet_resnet50_fpn(pretrained=False,
                     num_classes=1,
             pretrained_backbone=True)
    return model

def _read_if_exists(fpath):
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((IMAGEDIM, IMAGEDIM), dtype=np.uint8)
    return img

def collate_fn(data):
    images = torch.cat([item[0].unsqueeze(0) for item in data], dim=0)
    dicts = [item[1] for item in data]
    return images, dicts

def transform_img(ix, imlist):
    ret = np.zeros((IMAGEDIM + 2 * BOXOFFS, IMAGEDIM + 2 * BOXOFFS, 3), dtype=np.uint8)
    
    ret[BOXOFFS:IMAGEDIM+BOXOFFS, BOXOFFS:IMAGEDIM+BOXOFFS, 0] = _read_if_exists(imlist[ix])
    if ix + 1 < len(imlist):
        ret[BOXOFFS:IMAGEDIM+BOXOFFS, BOXOFFS:IMAGEDIM+BOXOFFS, 1] = _read_if_exists(imlist[ix + 1])
    if ix + 2 < len(imlist):
        ret[BOXOFFS:IMAGEDIM+BOXOFFS, BOXOFFS:IMAGEDIM+BOXOFFS, 2] = _read_if_exists(imlist[ix + 2])
    
    return ret

class BoxTestDataset(Dataset):
    def __init__(self, seq_list, cache_dir, transform=None):
        self.transform = transform
        
        self.seq_list = seq_list
        
        self.imglist = []
        
        self.fnamelist = []
        self.flist = []

    
        for seq_dir in seq_list:
            seq_id = os.path.split(seq_dir)[-1]

            fnames = sorted(glob.glob(os.path.join(seq_dir, '*.fts')))

            fnames_diff1 = [os.path.join(cache_dir, seq_id, 
                    os.path.split(fname)[-1] + '.diff1.png') for fname in fnames[:-1]]

            for ix, fullpath in enumerate(fnames):
                if ix + 2 < len(fnames_diff1) and os.path.exists(fnames_diff1[ix]):
                    self.imglist.append(ix)
                    self.fnamelist.append(fnames)
                    self.flist.append(fnames_diff1)

        
    def __len__(self):
        return len(self.imglist)
    
    def __getitem__(self, idx):
        image_ix, fnames, flist = self.imglist[idx], self.fnamelist[idx], self.flist[idx]
        
        imdata = transform_img(image_ix, flist)
        
        if self.transform is not None:
            imdata = self.transform(image=imdata)
        
        return img_to_tensor(imdata['image']), fnames[image_ix]


def tocenter(box):
    return np.array([box[0]+box[2], box[1]+box[3]]) * 0.5


def predict(model, device, test_loader, solpath='solution.csv'):
    print('starting predict...')
    model.eval()
    
    ret = {}
    
    with torch.no_grad():
        for data, img_path in tqdm(test_loader, total=len(test_loader)):
            data = data.to(device)
            detections = model(data)
            
            assert len(detections) == 1
            
            pred = detections[0]
            pboxes = pred['boxes'].cpu().numpy()
            pscores = pred['scores'].cpu().numpy()

            seq_id, fname = img_path[0].split(os.sep)[-2:]
            
            if seq_id not in ret:
                ret[seq_id] = []
                
            if len(pboxes) > 0:
                pcenter = tocenter(pboxes[0])
                score = pscores[0]
                
                box = pboxes[0]
                
                wh = (abs(box[2]-box[0]),abs(box[3]-box[1]))
                
                ret[seq_id].append((fname, (IMAGEDIM-int(pcenter[0]-BOXOFFS),int(pcenter[1]-BOXOFFS),
                                           int(wh[0]),int(wh[1])), score))
            
        
    print('writing pre-submission to %s...' % (solpath,))
    with open(solpath, 'w') as fd:
        for seq_id in ret.keys():
            plist = ret[seq_id]
            
            tot_score = np.mean([item[-1] for item in plist])
            
            fields = [str(seq_id)]
            fields += [subitem for item in plist for subitem in [item[0], 
                str(item[1][0]), str(item[1][1]), str(item[1][2]), str(item[1][3]), str(item[2])]]
            fields += [str(tot_score)]
            
            print(','.join(fields), file=fd)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model()
model.load_state_dict(torch.load(os.path.join(modeldir, BEST_MODELNAME), map_location=device))
model.to(device)
print('loaded model from ', os.path.join(modeldir, BEST_MODELNAME))


test_seqs = sorted(glob.glob(os.path.join(datapath, '*')))
print('total test seqs: ', len(test_seqs))

test_transform = Compose([
    Normalize(),
])
test_dataset = BoxTestDataset(test_seqs, cachepath, transform=test_transform)

test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available())

predict(model, device, test_loader, solution_outpath)
