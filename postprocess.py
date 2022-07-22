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
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from astropy.io import fits
import datetime
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


if len(sys.argv) != 5:
    print('usage: datadir cachedir insolutionpath outsolutionpath')
    sys.exit(1)


datapath, cachepath, solution_inpath, solution_outpath = sys.argv[1:]

def construct_traintsdata(dpath):
    start_date = datetime.datetime(1970,1,1)
    tot_folders = sorted(glob.glob(os.path.join(dpath, '*')))
    recs = []
    for seq_path in tot_folders:
        for fts_path in sorted(glob.glob(os.path.join(seq_path + '/*.fts'))):
            img,hdr = fits.getdata(fts_path, header=True)
    
            date_obs = datetime.datetime.strptime( hdr['DATE-OBS']+' '+hdr['TIME-OBS'], '%Y/%m/%d %H:%M:%S.%f')
    
            ts = (date_obs-start_date).total_seconds()
    
            seq_id = os.path.split(seq_path)[-1]
            fts_name = os.path.split(fts_path)[-1]
    
            recs.append({
                'seq_id': seq_id,
                'name': fts_name,
                'ts': ts
            })

    return recs


tsdata_dic = {}
seq2fname_dic = {}
for rec in construct_traintsdata(datapath):
    key = (rec['seq_id'], rec['name'])
    tsdata_dic[key] = rec['ts']
        
    if rec['seq_id'] not in seq2fname_dic:
        seq2fname_dic[rec['seq_id']] = []
    seq2fname_dic[rec['seq_id']].append((rec['ts'], rec['name']))
for k in seq2fname_dic.keys():
    seq2fname_dic[k] = sorted(seq2fname_dic[k])


def parse_predline(line):
    fields = line.split(',')
    fnames = []
    xycoords = []
    whs = []
    scores = []
    for i in range(0, len(fields) - 2, 6):
        fname = fields[1 + i]
        xcoord = float(fields[1 + i + 1])
        ycoord = float(fields[1 + i + 2])
        width = float(fields[1 + i + 3])
        height = float(fields[1 + i + 4])
        score = float(fields[1 + i + 5])
        fnames.append(fname)
        xycoords.append((xcoord, ycoord))
        whs.append((width, height))
        scores.append(score)

    return np.array(xycoords).reshape(-1, 2), whs, fnames, scores

def choose_max_inliers_group(xycoords, indices, vec, gap_tol=200):
    dists = np.dot(xycoords[indices], vec.T)
    
    dist_indices = np.argsort(dists)
    
    group = [indices[dist_indices[0]]]
    m = len(dist_indices)
    
    groups = []
    for ix in range(1, m):
        ii0 = dist_indices[ix-1]
        ii1 = dist_indices[ix]
        d0 = dists[ii0]
        d1 = dists[ii1]
        assert d0 <= d1
        if abs(d0-d1) < gap_tol:
            group.append(indices[ii1])
        else:
            groups.append(group)
            group = [indices[ii1]]
    
    groups.append(group)
    
    best_group = None
    for group in groups:
        if best_group is None or len(best_group) < len(group):
            best_group = group
    return best_group
    
def filter_inliers_ransac(xycoords, line_tol=40, gap_tol=200):
    n = min(len(xycoords), 1000)
    inlier_indices_max = None
    best_score = None
    for i in range(n - 1):
        for j in range(i+1, n):
            xy0 = xycoords[i]
            xy1 = xycoords[j]
            
            dx, dy = xy1[0] - xy0[0], xy1[1] - xy0[1]
            
            vec_orig = np.array([dx, dy])
            vec_orig /= np.linalg.norm(vec_orig)
            
            vec = np.array([-dy, dx])
            vec /= np.linalg.norm(vec)
            
            offs = np.dot(vec, xy0.T)
            
            dists = np.dot(xycoords, vec.T)
            
            indices = np.nonzero(np.abs(dists - offs) < line_tol)[0]
            
            if len(indices) < 1:
                continue
            
            indices = choose_max_inliers_group(xycoords, indices, vec_orig, gap_tol)
            
            score = len(indices) + 30 / (np.std(dists[indices] - offs) + 10)
            
            if inlier_indices_max is None or best_score < score:
                inlier_indices_max = indices
                best_score = score
                
    return inlier_indices_max

def fit_line_from_points(xycoords):
    return cv2.fitLine(xycoords, cv2.DIST_L2, 0, 1e-2, 1e-2)

def calc_mean_speed(seq_id, fnames, xycoords, inlier_indices):
    n = min(len(inlier_indices), 1000)
    ret = []
    for i in range(n-1):
        for j in range(i+1, n):
            ix0 = inlier_indices[i]
            ix1 = inlier_indices[j]
            
            fn0 = fnames[ix0]
            fn1 = fnames[ix1]
            
            ts0 = tsdata_dic[(seq_id, fn0)]
            ts1 = tsdata_dic[(seq_id, fn1)]
            
            xy0 = xycoords[ix0]
            xy1 = xycoords[ix1]
            
            ds = np.linalg.norm(xy1-xy0)
            dt = np.abs(ts1-ts0)+1e-7
            
            ret.append(ds/dt)
            
    return np.mean(ret)

# 38.24
def derive_points(seq_id, fnames, xycoords, whs, scores, inlier_indices):
    if len(inlier_indices) < 2:
        return xycoords[0].reshape(1, -1), fnames, -1
    
    assert(len(inlier_indices) >= 2)
    
    xy0 = xycoords[inlier_indices[0], :]
    xy1 = xycoords[inlier_indices[1], :]
    
    fname0 = fnames[inlier_indices[0]]
    fname1 = fnames[inlier_indices[1]]
    
    ts0 = tsdata_dic[(seq_id, fname0)]
    ts1 = tsdata_dic[(seq_id, fname1)]
    
    if ts0 > ts1:
        inlier_indices[0], inlier_indices[1] = inlier_indices[1], inlier_indices[0]
        xy0 = xycoords[inlier_indices[0], :]
        xy1 = xycoords[inlier_indices[1], :]

        fname0 = fnames[inlier_indices[0]]
        fname1 = fnames[inlier_indices[1]]

        ts0 = tsdata_dic[(seq_id, fname0)]
        ts1 = tsdata_dic[(seq_id, fname1)]
    
    speed_px = calc_mean_speed(seq_id, fnames, xycoords, inlier_indices)
    
    fnlist_all = seq2fname_dic[seq_id]
    
    fninlier_map = { fnames[ix]: ix for ix in inlier_indices }
    
    vx, vy, _, _ = fit_line_from_points(xycoords[inlier_indices])
    dvec = np.array([vx[0], vy[0]])
    dvec /= np.linalg.norm(dvec)
    
    dvec2=xy1-xy0
    dvec2 /= np.linalg.norm(dvec2)
    
    if np.dot(dvec, dvec2) < 0:
        dvec *= -1
    
    n = len(fnlist_all)
    start_ix = -1
    end_ix = -1
    subxycoords = np.zeros((n, 2))
    inlier_bitmap = np.zeros(n)
    for i in range(n):
        fname = fnlist_all[i][1]
        if fname in fninlier_map:
            subxycoords[i + 1] = xycoords[fninlier_map[fname]]
            inlier_bitmap[i + 1] = 1
            if start_ix == -1:
                start_ix = i
                continue
            end_ix = i

    assert start_ix != -1
    assert end_ix != -1
    assert end_ix + 1 < n
    
    start_ix += 1
    end_ix += 1
    
    assert inlier_bitmap[start_ix]
    assert inlier_bitmap[end_ix]
    
    for i in range(n):
        fname = fnlist_all[i][1]
        ts = tsdata_dic[(seq_id, fname)]
        if i < start_ix:
            fname_ref = fnlist_all[start_ix][1]
            ts_ref = tsdata_dic[(seq_id, fname_ref)]
            xy_ref = subxycoords[start_ix]
            
            assert ts_ref >= ts
            
            next_ix = [j for j in range(start_ix+1,n) if inlier_bitmap[j]][0]
            
            xy = xy_ref - speed_px * (ts_ref - ts) * dvec
            subxycoords[i] = xy
            continue
        if i > end_ix:
            fname_ref = fnlist_all[end_ix][1]
            ts_ref = tsdata_dic[(seq_id, fname_ref)]
            xy_ref = subxycoords[end_ix]
            
            assert ts >= ts_ref
            
            xy = xy_ref + speed_px * (ts - ts_ref) * dvec
            subxycoords[i] = xy
            continue
        if inlier_bitmap[i]:
             continue
            
        left_ix = [j for j in range(0,i) if inlier_bitmap[j]][-1]
        right_ix = [j for j in range(i+1,n) if inlier_bitmap[j]][0]
        
        fname_left = fnlist_all[left_ix][1]
        fname_right = fnlist_all[right_ix][1]
        
        ts_left = tsdata_dic[(seq_id, fname_left)]
        ts_right = tsdata_dic[(seq_id, fname_right)]
        
        xy_left = subxycoords[left_ix]
        xy_right = subxycoords[right_ix]
        
        tot_ts = (ts_right - ts_left)
        subxycoords[i] = (ts_right - ts) / tot_ts * xy_left + (ts - ts_left) / tot_ts * xy_right
        
    alpha = math.atan2(abs(dvec[1]), abs(dvec[0]))
    score = -1 if abs(alpha) > math.radians(20) else 0
        
    return subxycoords, [item[1] for item in fnlist_all], score



print('postprocessing submission and writing to %s ...' % (solution_outpath,))

with open(solution_inpath, 'r') as fd:
    with open(solution_outpath, 'w') as wfd:    
        lines = fd.readlines()

        fnames = []
        xycoords = []
        for line in tqdm(lines):
            fields = line.split(',')
            seq_id = fields[0]
            
            xycoords, whs, fnames, scores = parse_predline(line)
            
            pred_indices = filter_inliers_ransac(xycoords, line_tol=10, gap_tol=70)
            
            if pred_indices is None or len(pred_indices) < 1:
                continue
            
            try:
                new_xycoords, new_fnames, new_score = derive_points(seq_id, fnames, 
                        xycoords, whs, scores, pred_indices)
            except IndexError:
                continue
                
            score = fields[-1].rstrip() if new_score < 0 else str(new_score)
            
            fields1 = [seq_id]
            fields1 += [subitem for item in zip(new_xycoords, new_fnames) for subitem in [item[1], str(item[0][0]), str(item[0][1])]]
            fields1 += [score]
            
            print(','.join(fields1), file=wfd)
