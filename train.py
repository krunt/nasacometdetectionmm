
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



if len(sys.argv) != 4:
    print('usage: datadir cachedir outmodeldir')
    sys.exit(1)


datapath, cachepath, outmodeldir = sys.argv[1:]

def get_model():
    model = retinanet_resnet50_fpn(pretrained=False,
                     num_classes=1,
             pretrained_backbone=True)
    return model

train_seqs = sorted(glob.glob(os.path.join(datapath, '*')))

def _read_if_exists(fpath):
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((IMAGEDIM, IMAGEDIM), dtype=np.uint8)
    return img

def transform_img(ix, imlist):
    ret = np.zeros((IMAGEDIM + 2 * BOXOFFS, IMAGEDIM + 2 * BOXOFFS, 3), dtype=np.uint8)
    
    ret[BOXOFFS:IMAGEDIM+BOXOFFS, BOXOFFS:IMAGEDIM+BOXOFFS, 0] = _read_if_exists(imlist[ix])
    if ix + 1 < len(imlist):
        ret[BOXOFFS:IMAGEDIM+BOXOFFS, BOXOFFS:IMAGEDIM+BOXOFFS, 1] = _read_if_exists(imlist[ix + 1])
    if ix + 2 < len(imlist):
        ret[BOXOFFS:IMAGEDIM+BOXOFFS, BOXOFFS:IMAGEDIM+BOXOFFS, 2] = _read_if_exists(imlist[ix + 2])
    
    return ret

def getbbox(xylist):
    xyarr = np.array(xylist)
    xymin = np.min(xyarr, axis=0)
    xymax = np.max(xyarr, axis=0)
    return xymin, xymax


class BoxDataset(Dataset):
    def __init__(self, seq_list, cache_dir, transform=None, brightness_filter_list=["A","B","C"]):
        self.transform = transform
        self.imglist = []
        self.flist = []
        self.xylist = []
        
        for seq_dir in seq_list:
            brightness_cat = "A"
            lines = []

            metadata_list = glob.glob(os.path.join(seq_dir, "*.txt"))

            if len(metadata_list) == 0:
                continue

            with open(metadata_list[0], "r") as fd:
                lines = fd.readlines()
                
                brightness_cat = lines[2].split()[-1].strip()
                
                lines = lines[5:]
                
            if brightness_cat not in brightness_filter_list:
                continue

            seq_id = os.path.split(seq_dir)[-1]
                
            fnames_diff1 = [os.path.join(cache_dir, seq_id, line.split()[2]
                    + '.diff1.png') for line in lines[:-1]]
            
            xycoords = []
            for ix, line in enumerate(lines):
                fields = line.split()
                xy = IMAGEDIM-float(fields[3]), float(fields[4])
                xycoords.append(xy)
                
            for ix, line in enumerate(lines):
                fields = line.split()
                if ix + 2 < len(fnames_diff1) and os.path.exists(fnames_diff1[ix]):
                    self.imglist.append(ix)
                    self.flist.append(fnames_diff1)
                    self.xylist.append(xycoords)
                    
        
    def __len__(self):
        return len(self.imglist)
    
    def __getitem__(self, idx):
        image_ix, flist, xycoords = self.imglist[idx], self.flist[idx], self.xylist[idx]
        
        xymin, xymax = getbbox(xycoords[image_ix:image_ix+3])

        rad = BOXRADIUS
        bbox = (xymin[0]-rad+BOXOFFS, xymin[1]-rad+BOXOFFS,
               xymax[0]+rad+BOXOFFS, xymax[1]+rad+BOXOFFS)
    
        for num in bbox:
            assert num >= 0
            assert num < IMAGEDIM + 2 * BOXOFFS
        
        imdata = transform_img(image_ix, flist)
        
        if self.transform is not None:
            imdata = self.transform(image=imdata, bboxes=np.array(bbox).reshape(1, -1), category_ids=np.array([0]))
            
        gtdict = {
            'boxes': torch.tensor(imdata['bboxes']),
            'labels': torch.tensor(imdata['category_ids']),
        }
        
        return img_to_tensor(imdata['image']), gtdict



train_transform = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    Normalize(),
],
    bbox_params=BboxParams(format='pascal_voc', label_fields=['category_ids']),
)

trn_dataset = BoxDataset(
    train_seqs,
    cachepath,
    transform=train_transform,
    brightness_filter_list=["A","B"]
)

print('trn_dataset size:', len(trn_dataset))

batch_size = 8
num_workers = max(multiprocessing.cpu_count() // 2, 1)

def collate_fn(data):
    images = torch.cat([item[0].unsqueeze(0) for item in data], dim=0)
    dicts = [item[1] for item in data]
    return images, dicts

trn_loader = DataLoader(
    trn_dataset,
    sampler=RandomSampler(trn_dataset),
    batch_size=batch_size,
    drop_last=True,
    num_workers=num_workers,    
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available())


def train(model, device, train_loader, optimizer, epoch, tot_epochs=-1, nhead=-1):
    model.train()
    bbox_losses, classify_losses = [], []
    
    if nhead == -1:
        nhead = len(train_loader)
        
    for batch_idx, (data, target) in enumerate(train_loader):
        nhead -= 1
        if nhead < 0:
            break
                
        data, target = data.to(device), [
            {'boxes': item['boxes'].to(device),
              'labels': item['labels'].to(device)
             } for item in target]
        optimizer.zero_grad()
        output = model(data, target)
                    
        bbox_losses.append(output['bbox_regression'].item())
        classify_losses.append(output['classification'].item())
        
        if batch_idx % 100 == 0 and batch_idx > 0:
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tbbox_regr: {:.6f}\tclassification: {:.6f}'.format(
                epoch, tot_epochs, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                np.mean(bbox_losses), 
                np.mean(classify_losses)))
        
        sum_loss = (output['bbox_regression'] + output['classification'])
        sum_loss.backward()
        
        optimizer.step()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

epochs = 9

model = get_model()

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)

model_paths = []
for epoch in range(epochs):
    train(model, device, trn_loader, optimizer, epoch, epochs, nhead=1200)

    model_path = os.path.join(outmodeldir, 'model.comet.diff3.e_%d.pth' % (epoch,))
    
    torch.save(model.state_dict(), model_path)

    model_paths.append(model_path)
    
    lr_scheduler.step()


print('training finished...')

best_model_path = os.path.join(outmodeldir, BEST_MODELNAME)
shutil.copyfile(model_paths[-1], best_model_path)

print("model is copied here: ", best_model_path)

