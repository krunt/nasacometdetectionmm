
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
import random

random.seed(0)
np.random.seed(0)


def rdiff(indata, lasco_files, seq_id, outpath, ndiff=1):
    
    """  ## Note about running difference  ##
    The process of running difference involves subtracting the previous image
    in a sequence from the current image. This process removes static structures
    but emphasizes features in motion.
    """
    
    #Perform running difference. See note above.
    rdiff = np.diff(indata, axis=2, n=ndiff)
    
    # Write PNGS files
    dpi = 80            # dots per inch
    width = 1024        # image size (assuming 1024x1024 images)
    height = 1024       # image size (assuming 1024x1024 images)
    imgmin = -3.        # MAX value for display <- User defined
    imgmax = 3.         # MIN value for display <- User defined

    figsize = width / float(dpi), height / float(dpi) # Set output image size
    numfiles = rdiff.shape[2]       # For loop counter
    
    plt.ioff()                  # Turn off interactive plotting
    # Create images
    for i in range( numfiles ):
        #print("Writing image %i of %i with running-difference processing" % (i+1,numfiles))
        
        # The following commands just set up a figure with no borders, and writes the image to a png.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(np.fliplr(rdiff[:,:,i]), vmin=imgmin,vmax=imgmax,cmap='gray', interpolation='nearest',origin='lower')
        fname = os.path.split(lasco_files[i])[-1]
        outname = os.path.join(outpath, seq_id, fname + ('.diff%d.png' % (ndiff)))
        ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
        fig.savefig(outname, dpi=dpi, transparent=True)   
        plt.close()
        
    return 1



if len(sys.argv) != 3:
    print('usage: datadir outdir')
    sys.exit(1)


datapath, outpath = sys.argv[1:]

print('running preprocessing...')

pths = sorted(glob.glob(os.path.join(datapath, "*")))

def do_rdiff(ix):
    seq_path = pths[ix]
    seq_id = os.path.split(seq_path)[-1]

    try:
        os.mkdir(os.path.join(outpath, seq_id))
    except FileExistsError:
        pass

    lasco_files = sorted(glob.glob(os.path.join(seq_path, '*.fts')))
    
    # number of files
    nf = len(lasco_files) 
    
    # Create 3D data cube to hold data, assuming all LASCO C2 data have
    # array sizes of 1024x1024 pixels.
    data_cube = np.empty((1024,1024,nf))
    
    for i in range(nf):
        # read image and header from FITS file
        img,hdr = fits.getdata(lasco_files[i], header=True)
        
        # Normalize by exposure time (a good practice for LASCO data)
        img = img.astype('float64') / hdr['EXPTIME']
        
        # Store array into datacube (3D array)
        data_cube[:, :, i] = img

    rdiff(data_cube, lasco_files, seq_id, outpath)


ncores = max(multiprocessing.cpu_count() // 2, 1)
with Pool(ncores) as p:
    list(tqdm(p.imap(do_rdiff, range(len(pths))), total=len(pths)))


print('running preprocessing... done')
