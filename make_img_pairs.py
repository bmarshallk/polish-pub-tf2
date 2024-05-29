import sys, os

import matplotlib.pylab as plt
import numpy as np
import glob
import cv2
from scipy import signal, interpolate
import optparse

import simulation

try:
    from data_augmentation import elastic_transform
except:
    print("Could not load data_augmentation")

try:
    from astropy.io import fits 
except:
    print("Could not load astropy.io.fits")


def readfits(fnfits):
    """
    Read in fits file using astropy.io.fits
    """

    # Open the FITS file
    hdulist = fits.open(fnfits)

    # Get the shape of the data in the first HDU
    dshape = hdulist[0].shape 

    # Depending on the shape of the data, extract the data accordingly
    if len(dshape) == 2:
        # If the data is 2D, just take it as is
        data = hdulist[0].data
    elif len(dshape) == 3:
        # If the data is 3D, take the first slice
        data = hdulist[0].data[0]
    elif len(dshape) == 4:
        # If the data is 4D, take the first slice of the first slice
        data = hdulist[0].data[0, 0]

    # Extract the header from the first HDU
    header = hdulist[0].header

    # Extract the pixel scale and number of pixels from the header
    pixel_scale = abs(header['CDELT1'])
    num_pix = abs(header['NAXIS1'])

    # Return the data, header, pixel scale, and number of pixels
    return data, header, pixel_scale, num_pix


def gaussian2D(coords, amplitude=1, xo=0, yo=0, sigma_x=1, sigma_y=1, rho=0, offset=0, rot=0):
    """
    2D ellipsoidal Gaussian function, including rotation.
    """
    # Extract x and y coordinates
    x, y = coords

    # Convert rotation angle from degrees to radians
    rot = np.deg2rad(rot)

    # Rotate x and y coordinates
    x_rot = np.cos(rot) * x - y * np.sin(rot)
    y_rot = np.sin(rot) * x + np.cos(rot) * y

    # Convert peak center coordinates to float
    xo = float(xo)
    yo = float(yo)

    # Rotate peak center coordinates
    xo_rot = np.cos(rot) * xo - yo * np.sin(rot)
    yo_rot = np.sin(rot) * xo + np.cos(rot) * yo

    # Update coordinates and peak center with rotated values
    x, y, xo, yo = x_rot, y_rot, xo_rot, yo_rot

    # Create covariance matrix
    mat_cov = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                        [rho * sigma_x * sigma_y, sigma_y**2]])

    # Calculate inverse of covariance matrix
    mat_cov_inv = np.linalg.inv(mat_cov)

    # Stack the coordinates along the last axis
    mat_coords = np.stack((x - xo, y - yo), axis=-1)

    # Calculate Gaussian distribution
    G = amplitude * np.exp(-0.5 * np.einsum('...i,ij,...j', mat_coords, mat_cov_inv, mat_coords)) + offset

    # Remove single-dimensional entries from the shape of the array
    return G.squeeze()


def normalize_data(data, nbit=16):
    """ Normalize data to fit in bit range, 
    convert to specified dtype
    """
    data = data - data.min()
    data = data/data.max()
    data *= (2**nbit-1)
    if nbit==16:
        data = data.astype(np.uint16)
    elif nbit==8:
        data = data.astype(np.uint8)
    return data

def convolvehr(data, kernel, plotit=False, 
               rebin=4, norm=True, nbit=16, 
               noise=True, cmap='afmhot'):
    """ Take input data and 2D convolve with kernel 
    
    Parameters:
    ----------
    data : ndarray 
        data to be convolved 
    kernel : ndarray 
        convolutional kernel / PSF 
    """ 
    if len(data.shape)==3:
        kernel = kernel[..., None]
        ncolor = 1
    else:
        ncolor = 3
    
    if noise:
        data_noise = data + np.random.normal(0,5,data.shape)
    else:
        data_noise = data

    dataLR = signal.fftconvolve(data_noise, kernel, mode='same')

    if norm is True:
         dataLR = normalize_data(dataLR, nbit=nbit)
         data = normalize_data(data, nbit=nbit)

    dataLR = dataLR[rebin//2::rebin, rebin//2::rebin]

    if plotit:
        plt.figure()
        dataLRflat = dataLR.flatten()
        dataLRflat = dataLRflat[dataLRflat!=0]
        dataflat = data.flatten()
        dataflat = dataflat[dataflat!=0]
        plt.hist(dataLRflat, color='C1', alpha=0.5, 
                 density=True, log=True, bins=255)
        plt.hist(dataflat, bins=255, color='C0', alpha=0.25, 
                 density=True, log=True)
        plt.title('Bit value distribution', fontsize=20)
        plt.xlabel('Pixel value')
        plt.ylabel('Number of pixels')
        plt.legend(['Convolved','True'])
        plt.figure()
        if norm is False:
            data = data.reshape(data.shape[0]//4,4,
                                data.shape[-2]//4, 4, 
                                ncolor).mean(1).mean(-2)
            plt.imshow(dataLR[..., 0], cmap=cmap, 
                        vmax=dataLR[..., 0].max()*0.025)
        else:
            plt.imshow(dataLR, vmax=dataLR[..., 0].max(), cmap=cmap)
        plt.title('Convolved', fontsize=15)
        plt.figure()
        if norm is False:
            plt.imshow(data[..., 0], cmap=cmap, vmax=data.max()*0.1)
        else:
            plt.imshow(data, cmap=cmap,vmax=data.max()*0.1)
        plt.title('True', fontsize=15)
        plt.figure()
        plt.imshow(kernel[...,0],cmap='Greys',vmax=kernel[...,0].max()*0.35)
        plt.title('Kernel / PSF', fontsize=20)
        plt.show()

    return dataLR, data_noise

def create_LR_image(fl, kernel, fdirout=None, 
                    galaxies=False, plotit=False, 
                    norm=True, sky=False, rebin=4, nbit=16, 
                    distort_psf=False, subset='train',
                    nimages=800, nchan=1, save_img=True, nstart=0):
    """ Create a set of image pairs (true sky, dirty image) 
    and save to output directory 

    Parameters:
    ----------
    fl : str / list 
        Input file list 
    kernel : ndarray 
        PSF array 
    fdirout : str 
        Path to save output data to 
    galaxies : bool 
        Simulate galaxies 
    plotit : bool 
        Display plots for each image pair
    norm : bool 
        Normalize data 
    sky : bool 
        Use SKA sky data as input 
    rebin : int 
        1D resolution factor between true sky and convolved image 
    nbit : int 
        Number of bits for image data 
    distort_psr : bool 
        Distort each image pair's PSF with a difference perturbation 
    nimages : int 
        Number of image pairs 
    nchan : int 
        Number of radio frequency channels 
    save_img : bool 
        Save down images 

    Returns: 
    --------
    dataLR: ndarray 
        Convolved image arrays
    data, data_noise : ndarray 
    """
    if type(fl) is str:
        fl = glob.glob(fl+'/*.png')
        if len(fl)==0:
            print("Input file list is empty")
            exit()
    elif type(fl) is list:
        fl.sort()
    elif fl==None:
        pass
    else:
        print("Expected a list or a str as fl input")
        return

    assert subset in ['train', 'valid']

    fdiroutHR = options.fdout+'/POLISH_%s_HR/'%subset
    fdiroutLR = options.fdout+'/POLISH_%s_LR_bicubic/X%d/'%(subset,rebin)

    for ii in range(nimages):
        if fl is not None:
            fn = fl[ii]
            if fdiroutLR is None:
                fnout = fn.strip('.png')+'-conv.npy'
            else:
                fnoutLR = fdiroutLR + fn.split('/')[-1][:-4] + 'x%d.png' % rebin
        else:
            fn = '%04d.png'%(ii+nstart)
            fnoutLR = fdiroutLR + fn[:-4] + 'x%d.png' % rebin

        if os.path.isfile(fnoutLR):
            print("File exists, skipping %s"%fnoutLR)
            continue

        if ii%10==0:
            print("Finished %d/%d" % (ii, nimages))

        if galaxies:
            Nx, Ny = NSIDE, NSIDE
            data = np.zeros([Nx,Ny])

            # Get number of sources in this simulated image 
            nsrc = np.random.poisson(int(src_density*(Nx*Ny*PIXEL_SIZE**2/60.**2)))
            fdirgalparams = fdirout+'/galparams/'
            if not os.path.isdir(fdirgalparams):
                os.system('mkdir %s' % fdirgalparams)
            fnblobout = fdirgalparams + fn.split('/')[-1].strip('.png')+'GalParams.txt'
            SimObj = simulation.SimRadioGal(nx=Nx, ny=Ny)
            data = SimObj.sim_sky(distort_gal=False, fnblobout=fnblobout)

            if len(data.shape)==2:
                data = data[..., None]
            norm = True
        elif sky:
            data = np.load('SKA-fun-model.npy')
            data = data[800:800+4*118, 800:800+4*124]
            mm=np.where(data==data.max())[0]
            data[data<0] = 0
            data /= (data.max()/255.0/12.)
            data[data>255] = 255
            data = data.astype(np.uint8)
            data = data[..., None]
        else:
            data = cv2.imread(fn)
        
        if distort_psf:
            for aa in [1]:
                kernel_ = kernel[..., None]*np.ones([1,1,3])
#                alphad = np.random.uniform(0,5)
                alphad = np.random.uniform(0,20)
                if plotit:
                    plt.subplot(131)
                    plt.imshow(kernel,vmax=0.1,cmap='Greys')
                kernel_ = elastic_transform(kernel_, alpha=alphad,
                                           sigma=3, alpha_affine=0)
                if plotit:
                    plt.subplot(132)
                    plt.imshow(kernel_[..., 0], vmax=0.1, cmap='Greys')
                    plt.subplot(133)
                    plt.imshow(kernel-kernel_[..., 0],vmax=0.1, vmin=-0.1, cmap='Greys')
                    plt.colorbar()
                    plt.show()

                kernel_ = kernel_[..., 0]
                fdiroutPSF = fdirout[:-6]+'/psf/'
                fnout1=fdirout+'./test%0.2f.png'%aa
                fnout2=fdirout+'./test%0.2fx4.png'%aa
                np.save(fdiroutPSF+fn.split('/')[-1][:-4] + '-%0.2f-.npy'%alphad, kernel_)
        else:
            kernel_ = kernel

        dataLR, data_noise = convolvehr(data, kernel_, plotit=plotit, 
                                        rebin=rebin, norm=norm, nbit=nbit, 
                                        noise=True)

        data = normalize_data(data, nbit=nbit)
        dataLR = normalize_data(dataLR, nbit=nbit)

        if nbit==8:
            if save_img:
                cv2.imwrite(fnoutLR, dataLR.astype(np.uint8))
            else:
                np.save(fnoutLR[:-4], dataLR)
        elif nbit==16:
            if save_img:
                cv2.imwrite(fnoutLR, dataLR.astype(np.uint16))
            else:
                np.save(fnoutLR[:-4], dataLR)

        if nbit==8:
            if save_img:
                cv2.imwrite(fnoutLR, dataLR.astype(np.uint8))
            else:
                np.save(fnoutLR[:-4], dataLR)
        elif nbit==16:
            if save_img:
                cv2.imwrite(fnoutLR, dataLR.astype(np.uint16))
            else:
                np.save(fnoutLR[:-4], dataLR)

        if galaxies or sky:
            fnoutHR = fdiroutHR + fn.split('/')[-1][:-4] + '.png'
            fnoutHRnoise = fdiroutHR + fn.split('/')[-1][:-4] + 'noise.png'

            if nbit==8:
                if save_img:
                    cv2.imwrite(fnoutHR, data.astype(np.uint8))
                else:
                    np.save(fnoutHR, data)
            elif nbit==16:
                if save_img:
                    cv2.imwrite(fnoutHR, data.astype(np.uint16))
#                    cv2.imwrite(fnoutHRnoise, data_noise.astype(np.uint16))
                else:
                    np.save(fnoutHR, data)

        del dataLR, data, data_noise
 
if __name__=='__main__':
    parser = optparse.OptionParser(prog="hr2lr.py",
                   version="",
                   usage="%prog [OPTIONS]",
                   description="Take high resolution images, convolve them, \
                   and save output.")

    parser.add_option('-d', dest='fdirin', default=None,
                      help="input directory if high-res images already exist")
    parser.add_option('-k', '--kernel', dest='kernel', type='str',
                      help="", default='Gaussian')
    parser.add_option("-s", "--ksize", dest='ksize', type=int,
                      help="size of kernel", default=256)
    parser.add_option('-o', '--fdout', dest='fdout', type='str',
                      help="output directory", default='./')
    parser.add_option('-p', '--plotit', dest='plotit', action="store_true",
                      help="plot")
    parser.add_option('-x', '--galaxies', dest='galaxies', action="store_true",
                      help="only do point sources", default=True)
    parser.add_option('--sky', dest='sky', action="store_true",
                      help="use SKA mid image as input")
    parser.add_option('-r', '--rebin', dest='rebin', type=int,
                      help="factor to spatially rebin", default=4)
    parser.add_option('-b', '--nbit', dest='nbit', type=int,
                      help="number of bits for image", default=16)
    parser.add_option('-n', '--nchan', dest='nchan', type=int,
                      help="number of frequency channels for image", default=1)
    parser.add_option('--ntrain', dest='ntrain', type=int,
                      help="number of training images", default=800)
    parser.add_option('--nvalid', dest='nvalid', type=int,
                      help="number of validation images", default=100)
    parser.add_option('--distort_psf', dest='distort_psf', action="store_true",
                      help="perturb PSF for each image generated")
    parser.add_option('--pix', dest='pixel_size', type=float, default=0.25,
                      help="pixel size of true sky in arcseconds")
    parser.add_option('--src_density', dest='src_density', type=float, default=5,
                      help="source density per sq arcminute")
    parser.add_option('--nside', dest='nside', type=int, default=2048,
                      help="dimension of HR image")

    # Frequency range in GHz
    FREQMIN, FREQMAX = 0.7, 2.0

    options, args = parser.parse_args()
    PIXEL_SIZE = options.pixel_size
    src_density = options.src_density
    NSIDE = options.nside

    # Read in kernel. If -k is not given, assume Gaussian kernel 
    if options.kernel.endswith('npy'):
        kernel = np.load(options.kernel)
        nkern = len(kernel)
        kernel = kernel[nkern//2-options.ksize//2:nkern//2+options.ksize//2, 
                        nkern//2-options.ksize//2:nkern//2+options.ksize//2]
    elif options.kernel in ('Gaussian', 'gaussian'):
        kernel1D = signal.windows.gaussian(8, std=1).reshape(8, 1)
        kernel = np.outer(kernel1D, kernel1D)
    elif options.kernel.endswith('fits'):
        from skimage import transform
        kernel, header, pixel_scale_psf, num_pix = readfits(options.kernel)
        nkern = len(kernel)
        kernel = kernel[nkern//2-options.ksize//2:nkern//2+options.ksize//2, 
                        nkern//2-options.ksize//2:nkern//2+options.ksize//2]
        pixel_scale_psf *= 3600
        if abs((1-pixel_scale_psf/PIXEL_SIZE)) > 0.025:
            print("Stretching PSF by %0.3f to match map" % (pixel_scale_psf/PIXEL_SIZE))
            kernel = transform.rescale(kernel, pixel_scale_psf/PIXEL_SIZE)

    # Input directory
    if options.fdirin is None:
        fdirinTRAIN = None
        fdirinVALID = None 
    else:
        fdirinTRAIN = options.fdirin+'/POLISH_train_HR/'
        fdirinVALID = options.fdirin+'/POLISH_valid_HR/'

    # Output directories for training and validation. 
    # If they don't exist, create them
    fdiroutTRAIN_HR = options.fdout+'/POLISH_train_HR'
    fdiroutVALID_HR = options.fdout+'/POLISH_valid_HR'
    fdiroutTRAIN_LR = options.fdout+'/POLISH_train_LR_bicubic/X%d'%options.rebin
    fdiroutVALID_LR = options.fdout+'/POLISH_valid_LR_bicubic/X%d'%options.rebin
    
    fdiroutPSF = options.fdout+'/psf/'

    if not os.path.isdir(fdiroutTRAIN_HR):
        print("Making output training directory")
        os.system('mkdir -p %s' % fdiroutTRAIN_HR)

    if not os.path.isdir(fdiroutTRAIN_LR):
        print("Making output training directory")
        os.system('mkdir -p %s' % fdiroutTRAIN_LR)

    if not os.path.isdir(fdiroutVALID_HR):
        print("Making output training directory")
        os.system('mkdir -p %s' % fdiroutVALID_HR)

    if not os.path.isdir(fdiroutVALID_LR):
        print("Making output training directory")
        os.system('mkdir -p %s' % fdiroutVALID_LR)

    if not os.path.isdir(fdiroutPSF):
        print("Making output PSF directory")
        os.system('mkdir -p %s' % fdiroutPSF)

    print("saving idealized PSF")
    np.save('%s/psf_ideal.npy' % fdiroutPSF, kernel)

    # Create image pairs for training
    create_LR_image(fdirinTRAIN, kernel, fdirout=options.fdout, 
            plotit=options.plotit, galaxies=options.galaxies, 
            sky=options.sky, rebin=options.rebin, nbit=options.nbit, 
            distort_psf=options.distort_psf, nchan=options.nchan, subset='train',
            nimages=options.ntrain, nstart=0)   
    # Create image pairs for validation set
    create_LR_image(fdirinVALID, kernel, fdirout=options.fdout, 
            plotit=options.plotit, galaxies=options.galaxies, 
            sky=options.sky, rebin=options.rebin, nbit=options.nbit,
            distort_psf=options.distort_psf, nchan=options.nchan, subset='valid',
            nimages=options.nvalid, nstart=options.ntrain)





