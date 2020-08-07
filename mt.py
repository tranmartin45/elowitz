# mt functions

import numpy as np
import pandas as pd

import skimage
import skimage.io
import skimage.filters
import skimage.morphology
from skimage.filters import threshold_otsu, threshold_multiotsu, try_all_threshold
from skimage import registration
from skimage.feature import ORB, match_descriptors
from skimage.transform import matrix_transform
from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb

from scipy import ndimage as ndi
from scipy.ndimage import fourier_shift

import os
import glob

import bebi103

import colorcet

import bokeh
bokeh.io.output_notebook()

import holoviews as hv
hv.extension('bokeh')
bebi103.hv.set_defaults()

import panel as pn

from sklearn import linear_model
import scipy
import copy

def test():
    print('Hello world.')

def default_slice(dapi_raw, cfp_raw, chfp_raw, select_slice):
    '''
    Use default slice provided to slice all images.
    Returns dapi, cfp, and chfp as the selected slice within the repsective channel.
    
    ----------
    Parameters
    ----------
    dapi_raw : array
        z-stack of dapi
    cfp_raw : array
        z-stack of cfp
    chfp_raw : array
        z-stack of chfp
    select_slice : int
        use this slice to cut all z-stacks
        
    -------
    Output
    -------
    dapi : array
        default slice of dapi_raw
    cfp : array
        default slice of cfp_raw
    chfp : array
        default slice of chfp_raw
    '''
    
    dapi = dapi_raw[select_slice]
    cfp = cfp_raw[select_slice]
    chfp = chfp_raw[select_slice]
    
    return dapi, cfp, chfp

def best_slice(dapi_raw, cfp_raw, chfp_raw):
    '''
    Calculates best slice of z-stack based on highest dapi+ area.
    Returns dapi, cfp, and chfp as the selected slice within the repsective channel.
    
    ----------
    Parameters
    ----------
    dapi_raw : array
        z-stack of dapi
    cfp_raw : array
        z-stack of cfp
    chfp_raw : array
        z-stack of chfp
        
    -------
    Output
    -------
    dapi : array
        best slice of dapi_raw
    cfp : array
        best slice of cfp_raw
    chfp : array
        best slice of chfp_raw
    '''
    dapi_areas = []
    dapi_filt_gauss = skimage.filters.gaussian(dapi_raw, 1.5)
    threshold1 = threshold_otsu(dapi_filt_gauss)

    for i in range(len(dapi_raw)):
        dapi_filt_gauss_slice = dapi_filt_gauss[i]
        dapi_filt_gauss_bw = dapi_filt_gauss_slice > threshold1
        dapi_area = dapi_filt_gauss_bw.sum()
        dapi_areas.append(dapi_area)

    slices = [i for i in range(len(dapi_raw))]

    data = {'slice': slices, 'areas': dapi_areas}
    df_dapi = pd.DataFrame(data)

    hv.Points(data=df_dapi,
             kdims=['slice', 'areas'],
             )
    select_slice = dapi_areas.index(max(dapi_areas))
    print(select_slice)
    
    dapi = dapi_raw[select_slice]
    cfp = cfp_raw[select_slice]
    chfp = chfp_raw[select_slice]
    
    return dapi, cfp, chfp

def register_images(cfp, chfp):
    '''
    Registers the two channels using register_translation from skimage.features.
    Prints detected pixel offset (y,x). If shift is more than 10 pixels in y or x, will default to [y, x] = [0, -5]
    Returns corrected moving array.
    
    ----------
    Parameters
    ----------
    cfp : array
        best slice of cfp
    chfp : array
        best slice of chfp
    
    -------
    Output
    -------
    chfp : array
        fixed chfp slice
    '''
    # pixel precision first
    shift, error, diffphase = register_translation(cfp, chfp)
    print(f"Detected pixel offset (y, x): {shift}")
    
    if abs(shift[0]) > 10.0:
        shift = [0,-5]
        print('detected pixel offset is more than 10, default shift to [0,-5].')
    elif abs(shift[1]) > 10.0:
        shift = [0,-5]
        print('detected pixel offset is more than 10, default shift to [0,-5].')
    else:
        print('pass check: shift is less than 10 in both dimensions')
        
    chfp_shift = ndi.shift(chfp, shift)

    # corrected chfp_slice
    chfp = chfp_shift
    return chfp, shift

def imadjust(img, lower_bound=0.25, upper_bound=99.75):
    lower = np.percentile(img, lower_bound)
    upper = np.percentile(img, upper_bound)
    out = (img - lower) * (255 / (upper - lower))
    return np.clip(out, 0, 255, out)

def show_two_ims(
    im_1,
    im_2,
    titles=[None, None],
    interpixel_distances=[0.13, 0.13],
    cmap=[None, None],
    colorbar=[False, False]
):
    """
    Convenient function for showing two images side by side.
    
    ----------
    Parameters
    ----------
    im_1 : array
    im_2 : array
    titles : list
        follows the format of [im_1 title, im_2 title]
    interpixel_distances : list
        follows the format of [im_1 ip dist, im_2 ip dist]
    cmap : list
        follows the format of [im_1 cmap, im_2 cmap]
    colorbar : list
        follows the format of [im_1 colorbar, im_2 colorbar]
    """
    p_1 = bebi103.image.imshow(
        im_1,
        frame_height=225,
        title=titles[0],
        cmap=cmap[0],
        #interpixel_distance=interpixel_distances[0],
        #length_units="µm",
        colorbar=colorbar[0]
    )
    p_2 = bebi103.image.imshow(
        im_2,
        frame_height=225,
        title=titles[1],
        cmap=cmap[1],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[1]
    )
    p_2.x_range = p_1.x_range
    p_2.y_range = p_1.y_range

    return bokeh.layouts.gridplot([p_1, p_2], ncols=2)

def show_three_ims(
    im_1,
    im_2,
    im_3,
    titles=[None, None, None],
    interpixel_distances=[0.13, 0.13, 0.13],
    cmap=[None, None, None],
    colorbar=[False, False, False]
):
    """Convenient function for showing three images side by side."""
    p_1 = bebi103.image.imshow(
        im_1,
        frame_height=225,
        title=titles[0],
        cmap=cmap[0],
        #interpixel_distance=interpixel_distances[0],
        #length_units="µm",
        colorbar=colorbar[0]
    )
    p_2 = bebi103.image.imshow(
        im_2,
        frame_height=225,
        title=titles[1],
        cmap=cmap[1],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[1]
    )
    p_3 = bebi103.image.imshow(
        im_3,
        frame_height=225,
        title=titles[2],
        cmap=cmap[2],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[2]
    )
    p_2.x_range = p_1.x_range
    p_2.y_range = p_1.y_range
    p_3.x_range = p_1.x_range
    p_3.y_range = p_1.y_range
    
    return bokeh.layouts.gridplot([p_1, p_2, p_3], ncols=3)

def show_four_ims(
    im_1,
    im_2,
    im_3,
    im_4,
    titles=[None, None, None, None],
    interpixel_distances=[0.13, 0.13, 0.13, 0.13],
    cmap=[None, None, None, None],
    colorbar=[False, False, False, False]
):
    """Convenient function for showing four images side by side."""
    p_1 = bebi103.image.imshow(
        im_1,
        frame_height=225,
        title=titles[0],
        cmap=cmap[0],
        #interpixel_distance=interpixel_distances[0],
        #length_units="µm",
        colorbar=colorbar[0]
    )
    p_2 = bebi103.image.imshow(
        im_2,
        frame_height=225,
        title=titles[1],
        cmap=cmap[1],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[1]
    )
    p_3 = bebi103.image.imshow(
        im_3,
        frame_height=225,
        title=titles[2],
        cmap=cmap[2],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[2]
    )
    p_4 = bebi103.image.imshow(
        im_4,
        frame_height=225,
        title=titles[3],
        cmap=cmap[3],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[3]
    )
    p_2.x_range = p_1.x_range
    p_2.y_range = p_1.y_range
    p_3.x_range = p_1.x_range
    p_3.y_range = p_1.y_range
    p_4.x_range = p_1.x_range
    p_4.y_range = p_1.y_range
    
    return bokeh.layouts.gridplot([p_1, p_2, p_3, p_4], ncols=2)

def show_five_ims(
    im_1,
    im_2,
    im_3,
    im_4,
    im_5,
    titles=[None, None, None, None, None],
    interpixel_distances=[0.13, 0.13, 0.13, 0.13, 0.13],
    cmap=[None, None, None, None, None],
    colorbar=[False, False, False, False, False]
):
    """Convenient function for showing five images side by side."""
    p_1 = bebi103.image.imshow(
        im_1,
        frame_height=225,
        title=titles[0],
        cmap=cmap[0],
        #interpixel_distance=interpixel_distances[0],
        #length_units="µm",
        colorbar=colorbar[0]
    )
    p_2 = bebi103.image.imshow(
        im_2,
        frame_height=225,
        title=titles[1],
        cmap=cmap[1],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[1]
    )
    p_3 = bebi103.image.imshow(
        im_3,
        frame_height=225,
        title=titles[2],
        cmap=cmap[2],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[2]
    )
    p_4 = bebi103.image.imshow(
        im_4,
        frame_height=225,
        title=titles[3],
        cmap=cmap[3],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[3]
    )
    p_5 = bebi103.image.imshow(
        im_5,
        frame_height=225,
        title=titles[4],
        cmap=cmap[4],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[4]
    )
    p_2.x_range = p_1.x_range
    p_2.y_range = p_1.y_range
    p_3.x_range = p_1.x_range
    p_3.y_range = p_1.y_range
    p_4.x_range = p_1.x_range
    p_4.y_range = p_1.y_range
    p_5.x_range = p_1.x_range
    p_5.y_range = p_1.y_range
    
    return bokeh.layouts.gridplot([p_1, p_2, p_3, p_4, p_5], ncols=3)

def show_six_ims(
    im_1,
    im_2,
    im_3,
    im_4,
    im_5,
    im_6,
    titles=[None, None, None, None, None, None],
    interpixel_distances=[0.13, 0.13, 0.13, 0.13, 0.13, 0.13],
    cmap=[None, None, None, None, None, None],
    colorbar=[False, False, False, False, False, False]
):
    """Convenient function for showing six images side by side."""
    p_1 = bebi103.image.imshow(
        im_1,
        frame_height=225,
        title=titles[0],
        cmap=cmap[0],
        #interpixel_distance=interpixel_distances[0],
        #length_units="µm",
        colorbar=colorbar[0]
    )
    p_2 = bebi103.image.imshow(
        im_2,
        frame_height=225,
        title=titles[1],
        cmap=cmap[1],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[1]
    )
    p_3 = bebi103.image.imshow(
        im_3,
        frame_height=225,
        title=titles[2],
        cmap=cmap[2],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[2]
    )
    p_4 = bebi103.image.imshow(
        im_4,
        frame_height=225,
        title=titles[3],
        cmap=cmap[3],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[3]
    )
    p_5 = bebi103.image.imshow(
        im_5,
        frame_height=225,
        title=titles[4],
        cmap=cmap[4],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[4]
    )
    p_6 = bebi103.image.imshow(
        im_6,
        frame_height=225,
        title=titles[5],
        cmap=cmap[5],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[5]
    )
    p_2.x_range = p_1.x_range
    p_2.y_range = p_1.y_range
    p_3.x_range = p_1.x_range
    p_3.y_range = p_1.y_range
    p_4.x_range = p_1.x_range
    p_4.y_range = p_1.y_range
    p_5.x_range = p_1.x_range
    p_5.y_range = p_1.y_range
    p_6.x_range = p_1.x_range
    p_6.y_range = p_1.y_range
    
    return bokeh.layouts.gridplot([p_1, p_2, p_3, p_4, p_5, p_6], ncols=3)

def show_nine_ims(
    im_1,
    im_2,
    im_3,
    im_4,
    im_5,
    im_6,
    im_7,
    im_8,
    im_9,
    titles=[None, None, None, None, None, None, None, None, None],
    interpixel_distances=[0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13],
    cmap=[None, None, None, None, None, None, None, None, None],
    colorbar=[False, False, False, False, False, False, False, False, False]
):
    """Convenient function for showing nine images side by side."""
    p_1 = bebi103.image.imshow(
        im_1,
        frame_height=225,
        title=titles[0],
        cmap=cmap[0],
        #interpixel_distance=interpixel_distances[0],
        #length_units="µm",
        colorbar=colorbar[0]
    )
    p_2 = bebi103.image.imshow(
        im_2,
        frame_height=225,
        title=titles[1],
        cmap=cmap[1],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[1]
    )
    p_3 = bebi103.image.imshow(
        im_3,
        frame_height=225,
        title=titles[2],
        cmap=cmap[2],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[2]
    )
    p_4 = bebi103.image.imshow(
        im_4,
        frame_height=225,
        title=titles[3],
        cmap=cmap[3],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[3]
    )
    p_5 = bebi103.image.imshow(
        im_5,
        frame_height=225,
        title=titles[4],
        cmap=cmap[4],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[4]
    )
    p_6 = bebi103.image.imshow(
        im_6,
        frame_height=225,
        title=titles[5],
        cmap=cmap[5],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[5]
    )
    p_7 = bebi103.image.imshow(
        im_7,
        frame_height=225,
        title=titles[6],
        cmap=cmap[6],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[6]
    )
    p_8 = bebi103.image.imshow(
        im_7,
        frame_height=225,
        title=titles[7],
        cmap=cmap[7],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[7]
    )
    p_9 = bebi103.image.imshow(
        im_8,
        frame_height=225,
        title=titles[8],
        cmap=cmap[8],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
        colorbar=colorbar[8]
    )
    p_2.x_range = p_1.x_range
    p_2.y_range = p_1.y_range
    p_3.x_range = p_1.x_range
    p_3.y_range = p_1.y_range
    p_4.x_range = p_1.x_range
    p_4.y_range = p_1.y_range
    p_5.x_range = p_1.x_range
    p_5.y_range = p_1.y_range
    p_6.x_range = p_1.x_range
    p_6.y_range = p_1.y_range
    p_7.x_range = p_1.x_range
    p_7.y_range = p_1.y_range
    p_8.x_range = p_1.x_range
    p_8.y_range = p_1.y_range
    p_9.x_range = p_1.x_range
    p_9.y_range = p_1.y_range
    
    return bokeh.layouts.gridplot([p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9], ncols=3)

def filt_gauss(dapi, cfp, chfp, size):
    '''
    Applies a gaussian filter with supplied size using skimage.filters.gaussian. Returns filtered images.
    
    ----------
    Parameters
    ----------
    dapi : array
        dapi raw intensity image slice
    cfp : array
        cfp raw intensity image slice
    chfp : array
        chfp raw intensity image slice
        
    ------
    Output
    ------
    dapi_filt_gauss : array
        dapi filtered with gaussian
    cfp_filt_gauss : array
        cfp filtered with gaussian
    chfp_filt_gauss : array
        chfp filtered with gaussian
    '''
    dapi_filt_gauss = skimage.filters.gaussian(dapi, size)
    cfp_filt_gauss = skimage.filters.gaussian(cfp, size)
    chfp_filt_gauss = skimage.filters.gaussian(chfp, size)
    return dapi_filt_gauss, cfp_filt_gauss, chfp_filt_gauss

def threshold(dapi_filt_gauss, cfp_filt_gauss, chfp_filt_gauss):
    '''
    Threshold given images based on Otsu's method. Returns thresholded images, as well as dapi_sum.
    
    ----------
    Parameters
    ----------
    dapi_filt_gauss : array
        dapi filtered with gaussian
    cfp_filt_gauss : array
        cfp filtered with gaussian
    chfp_filt_gauss : array
        chfp filtered with gaussian
        
    ------
    Output
    ------
    dapi_filt_gauss_bw : array
        dapi filted with gaussian and segmented
    cfp_filt_gauss_bw : array
        cfp filted with gaussian and segmented
    chfp_filt_gauss_bw : array
        chfp filted with gaussian and segmented
    threshold2 : float
        cfp threshold as float (original range was 0 to 65535)
    threshold3 : float
        chfp threshold as float (original range was 0 to 65535)
    '''
    
    threshold1 = threshold_otsu(dapi_filt_gauss)
    print(threshold1, 'is where the dapi cutoff point is.')
    dapi_filt_gauss_bw = dapi_filt_gauss > threshold1

    threshold2 = threshold_otsu(cfp_filt_gauss)
    print(threshold2, 'is where the cfp cutoff point is.')
    cfp_filt_gauss_bw = cfp_filt_gauss > threshold2

    threshold3 = threshold_otsu(chfp_filt_gauss)
    print(threshold3, 'is where the chfp cutoff point is.')
    chfp_filt_gauss_bw = chfp_filt_gauss > threshold3
    return dapi_filt_gauss_bw, cfp_filt_gauss_bw, chfp_filt_gauss_bw, threshold2, threshold3

def dilate(dapi_filt_gauss_bw, cfp_filt_gauss_bw, chfp_filt_gauss_bw, size):
    '''
    Dilates given images with supplied size structuring element (disk shape). Returns dilated images.
    
    ----------
    Parameters
    ----------
    dapi_filt_gauss_bw : array
        dapi filtered with gaussian and segmented
    cfp_filt_gauss_bw : array
        cfp filtered with gaussian and segmented
    chfp_filt_gauss_bw : array
        chfp filtered with gaussian and segmented
        
    ------
    Output
    ------
    dapi_dil : array
        dapi filted with gaussian, segmented, and dilated
    cfp_dil : array
        cfp filted with gaussian, segmented, and dilated
    chfp_dil : array
        chfp filted with gaussian, segmented, and dilated
    '''

    # Make the structuring element 1 pixel radius disk
    selem = skimage.morphology.disk(size)

    # Dilate image
    dapi_dil = skimage.morphology.dilation(dapi_filt_gauss_bw, selem)
    cfp_dil = skimage.morphology.dilation(cfp_filt_gauss_bw, selem)
    chfp_dil = skimage.morphology.dilation(chfp_filt_gauss_bw, selem)
    return dapi_dil, cfp_dil, chfp_dil

def ws(image, selem_size, footprint, min_size, intensity_image):
    '''
    Function to perform watershed and remove small objects. Will generate a dataframe and output total area
    of the watershed cells.
    
    ----------
    Parameters
    ----------
    image : array
        image should be filtered and segmented
    footprint : np.ones((int,int))
        represents the local region within which to search for peaks at every point in image.
    min_size : int
        will remove all objects smaller than min_size
    intensity_image : array
        image should be raw image slice
    
    ---------
    Output
    ---------
    label_image : array
        array of labeled image after watershed, each cell has unique label
    df : DataFrame
        DataFrame of label_image with 'label', 'centroid', 'area', and 'mean_intensity'
    area : int
        sum of watershed cells 
    '''
    
    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, footprint=footprint,
                                labels=image)
    selem = skimage.morphology.disk(selem_size)
    # increased dilation of minima to prevent oversegmentation
    local_maxi_dil = skimage.morphology.dilation(local_maxi, selem)
    markers = ndi.label(local_maxi_dil)[0]
    labels = skimage.morphology.watershed(-distance, markers, mask=image, watershed_line=True)
    ws_mask = skimage.morphology.remove_small_objects(labels, min_size=min_size)
    label_image = skimage.measure.label(ws_mask)
    if label_image.sum() != 0:
        props = skimage.measure.regionprops_table(label_image, intensity_image=intensity_image, properties=('label',
                                                                                                 'centroid',
                                                                                                 'area',
                                                                                                 'mean_intensity'))
        df = pd.DataFrame(props)
        area = df['area'].sum()
    else:
        props = {'label': [0], 'centroid': [0], 'area': [0], 'mean_intensity': [0]}
        df = pd.DataFrame(props)
        area = df['area'].sum()
    return label_image, df, area, distance, markers

def ws_dot(image):
    '''
    Function to perform watershed. Will generate labels, distance transform, and markers.
    Made with parameters specific for dot segmentation (distance=0, default selem (1))
    
    ----------
    Parameters
    ----------
    image : array
        image should be filtered and segmented
    
    ---------
    Output
    ---------
    label_image : array
        array of labeled image after watershed, each cell has unique label
    df : DataFrame
        DataFrame of label_image with 'label', 'centroid', 'area', and 'mean_intensity'
    area : int
        sum of watershed cells 
    '''
    
    distance = -1 * ndi.distance_transform_edt(image)
    local_mini = skimage.morphology.h_minima(distance, 0)
    
    selem = skimage.morphology.ball(3)
    # increased dilation of minima to prevent oversegmentation
    local_mini_dil = skimage.morphology.dilation(local_mini, selem)
    
    mrkr = ndi.label(local_mini_dil)[0]
    labels = skimage.morphology.watershed(distance, markers = mrkr, mask=image, watershed_line=True)
    return labels, distance, mrkr

def ws_nuc(image):
    '''
    Function to perform watershed. Will generate labels, distance transform, and markers of the watershed cells.
    Made specifically for nuclear segmentation (distance=0, selem=6)
    
    ----------
    Parameters
    ----------
    image : array
        image should be filtered and segmented
    
    ---------
    Output
    ---------
    label_image : array
        array of labeled image after watershed, each cell has unique label
    df : DataFrame
        DataFrame of label_image with 'label', 'centroid', 'area', and 'mean_intensity'
    area : int
        sum of watershed cells 
    '''
    
    distance = -1 * ndi.distance_transform_edt(image)
    selem = skimage.morphology.ball(6)
    # increased selem to compensate for 0 min distance
    local_mini = skimage.morphology.h_minima(distance, 0, selem=selem)
    
    selem = skimage.morphology.ball(3)
    # increased dilation of minima to prevent oversegmentation
    local_mini_dil = skimage.morphology.dilation(local_mini, selem)
    
    mrkr = ndi.label(local_mini_dil)[0]
    labels = skimage.morphology.watershed(distance, markers = mrkr,\
                                      mask=image, watershed_line=True)
    return labels, distance, mrkr

def remove_small_df(labels, min_size, intensity_image):
    '''
    Function to remove small objects and generate DataFrame from watershed labels.
    
    ----------
    Parameters
    ----------
    labels : array
        label from watershed
    min_size : int
        min_size to remove small objects
    intensity image : array (same size as labels)
        array of intensity image, should not be adjusted
        
    -------
    Output
    -------
    label_image : array
        relabeled image after remove small objects
    df : DataFrame
        DataFrame of label_image
    area : float
        total area of labels
    
    '''
    ws_mask = skimage.morphology.remove_small_objects(labels, min_size=min_size)
    label_image = skimage.measure.label(ws_mask)
    if label_image.sum() != 0:
        props = skimage.measure.regionprops_table(label_image, intensity_image=intensity_image, properties=('label',
                                                                                                 'centroid',
                                                                                                 'area',
                                                                                                 'mean_intensity'))
        df = pd.DataFrame(props)
        area = df['area'].sum()
    else:
        props = {'label': [0], 'centroid': [0], 'area': [0], 'mean_intensity': [0]}
        df = pd.DataFrame(props)
        area = df['area'].sum()
    return label_image, df, area

def remove_large(label_image, max_size, df, intensity_image):
    '''
    Function to remove large segments from image.
    
    ----------
    Parameters
    ----------
    label_image : array
        watershed image with labels
    max_size : int
        remove all segments larger than this size. Default is 1000
    df : DataFrame
        DataFrame of watershed image
    intensity_image : array
        image should be raw image slice
    
    --------
    Output
    --------
    large_sub : array
        image with large segments subtracted
    df_large : DataFrame
        DataFrame with large segments subtracted
        
    '''
    if label_image.sum() != 0:
        empty_array = np.zeros_like(label_image)
        max_size = max_size

        for i, label in enumerate(df.loc[df.loc[:, 'area'] > max_size]['label']):
            x = label_image == label
            y = x * label
            empty_array += y

        large_sub = label_image - empty_array

        props_large = skimage.measure.regionprops_table(large_sub, intensity_image=intensity_image, properties=('label',
                                                                                             'centroid',
                                                                                             'area',
                                                                                             'mean_intensity'))
        df_large = pd.DataFrame(props_large)
        area_large = df_large['area'].sum()
    else:
        large_sub = label_image
        props = {'label': [0], 'centroid': [0], 'area': [0], 'mean_intensity': [0]}
        df_large = pd.DataFrame(props)
    return large_sub, df_large

def coloc(cfp_large_sub, chfp_large_sub, cfp, chfp):
    '''
    Function to get all nuclei with mean intensity for each channel. Can do two channels.
    
    ---------
    Parameters
    ---------
    cfp_large_sub : array
        watershed image with small and large segments subtracted
    chfp_large_sub : array
        watershed image with small and large segments subtracted
    cfp : array
        raw intensity image
    chfp : array
        raw intensity image
    
    -------
    Output
    -------
    df_co : DataFrame
        DataFrame with all nuclei with mean intensity for each channel.
    '''
    selem = skimage.morphology.disk(3)
    cfp_z4_e = skimage.morphology.erosion(cfp_large_sub, selem)
    chfp_z4_e = skimage.morphology.erosion(chfp_large_sub, selem)
    c_ch = np.logical_or(cfp_z4_e, chfp_z4_e)
    c_ch_small = skimage.morphology.remove_small_objects(c_ch, min_size=50)
    c_ch_label = skimage.measure.label(c_ch_small)

    props_cfp_co = skimage.measure.regionprops_table(c_ch_label, intensity_image=cfp, properties=('label',
                                                                                             'centroid',
                                                                                             'area',
                                                                                             'mean_intensity'))
    df_cfp_co = pd.DataFrame(props_cfp_co)
    

    props_chfp_co = skimage.measure.regionprops_table(c_ch_label, intensity_image=chfp, properties=('label',
                                                                                                 'centroid',
                                                                                                 'area',
                                                                                                 'mean_intensity'))
    df_chfp_co = pd.DataFrame(props_chfp_co)
    
    cfp_vals = df_cfp_co['mean_intensity'].values
    chfp_vals = df_chfp_co['mean_intensity'].values
    data = {'CFP mean intensity': cfp_vals, 'ChFP mean intensity': chfp_vals}
    df_co = pd.DataFrame(data)
    return df_co

def freedman_diaconis_bins(data):
    """Number of bins based on Freedman-Diaconis rule."""
    h = 2 * (np.percentile(data, 75) - np.percentile(data, 25)) / np.cbrt(len(data))
    return int(np.ceil((data.max() - data.min()) / h))

def classify_dots(fr_c0, fr_c1, fr_c2, r_c0, r_c1, r_c2, min_dist):
    '''
    Function to classify dots as ONLY dots or CO dots based on distance from dots in another channel.
    Classifies the dots of the first channel provided against the distance from dots in the second channel.
    
    ----------
    Parameters
    ----------
    fr_c0 : array of far-red centroid-0
    fr_c1 : array of far-red centroid-1
    fr_c2 : array of far-red centroid-2
    r_c0 : array of red centroid-0
    r_c1 : array of red centroid-0
    r_c2 : array of red centroid-0
    min_dist : int
        min_dist to classify dots as CO dots or ONLY dots. Should be around 4.
        
    -------
    Output
    -------
    co_dot : list of the CO dots
    fr_dot : list of the ONLY dots
    
    '''
    co_dot = []
    fr_dot = []

    for i, (a, b, c) in enumerate(zip(fr_c0, fr_c1, fr_c2)):
        list = []

        for j, (d, e, f) in enumerate(zip(r_c0, r_c1, r_c2)):
            p1 = np.array([a, b, c])
            p2 = np.array([d, e, f])

            squared_dist = np.sum((p1-p2)**2, axis=0)
            dist = np.sqrt(squared_dist)

            list.append(dist)

        if (np.asarray(list) < min_dist).sum() == 1:
            co_dot.append((a, b, c))
        elif (np.asarray(list) < min_dist).sum() < 1:
            fr_dot.append((a, b, c))
        elif (np.asarray(list) < min_dist).sum() > 1:
            print('There are multiple dots nearby.')
            co_dot.append((a, b, c))
    return co_dot, fr_dot

def make_df(fr_dot):
    '''
    Make a DataFrame from an array of centroid positions (that was made using classify_dots() function).
    
    ----------
    Parameters
    ----------
    fr_dot : list
        List of centroid positions. Each entry in the list is a tuple containing the centroid0, 1, and 2.
    
    -------
    Output
    -------
    df_fr_dot : DataFrame
        DataFrame of centroid positions.
    '''
    fr_dot_0 = []
    fr_dot_1 = []
    fr_dot_2 = []

    for i, j in enumerate(fr_dot):
        fr_dot_0.append(fr_dot[i][0])
        fr_dot_1.append(fr_dot[i][1])
        fr_dot_2.append(fr_dot[i][2])

    data = {'fr_dot_0': fr_dot_0,
            'fr_dot_1': fr_dot_1,
            'fr_dot_2': fr_dot_2,
           }
    df_fr_dot = pd.DataFrame(data)
    return df_fr_dot

def dots_to_nuc(df_fr_dot, fr_mo1, label_image_cfp_small_dil):
    '''
    Finds dots that lie within CFP positive nuclei.
    
    ----------
    Parameters
    ----------
    df_fr_dot : DataFrame
        DataFrame of centroid positions
    fr_mo1 : array
        array of suitable size
    label_image_cfp_small_dil : array
        dilated labels of CFP channel
    
    -------
    Output
    -------
    df_fr_nuc : DataFrame
        DataFrame of ALL dots within CFP+ nuclei
    '''
    
    fr_only_c0 = df_fr_dot['fr_dot_0'].values
    fr_only_c1 = df_fr_dot['fr_dot_1'].values
    fr_only_c2 = df_fr_dot['fr_dot_2'].values

    fr_dot_nuc = np.zeros_like(fr_mo1)

    for i, (a, b, c) in enumerate(zip(fr_only_c0, fr_only_c1, fr_only_c2)):
        fr_dot_nuc[a, b, c] = 1

    fr_dot_nuc_bool = fr_dot_nuc.astype(bool)

    fr_nuc = label_image_cfp_small_dil * fr_dot_nuc_bool

    fr_nuc_relabel = skimage.measure.label(fr_nuc)

    labels_fr_nuc = skimage.measure.regionprops_table(fr_nuc_relabel, properties=('label','centroid'))

    df_fr_nuc = pd.DataFrame(labels_fr_nuc)
    return df_fr_nuc

def log_filter(af488_g, gauss_size, selem_size, zero_crossing_filter_size):
    '''
    Applies a LoG filter to gaussian filtered image - works well to segment all dots.
    This involves finding the max and min filter, then using this to find zero-crossings.
    Uses the function zero_crossing_filter() to find zero-crossings.
    
    ----------
    Parameters
    ----------
    af488_g: array
        gaussian filtered image
    gauss_size: int
        size of gaussian filter
    selem_size: int
        size of structuring element for the max and min filter
    zero_crossing_filter_size: int
        size of zero-crossing filter
    
    -------
    Output
    -------
    af488_edge_zero: array
        zero-crossing filter image
    '''
    af488_LoG = ndi.filters.gaussian_laplace(af488_g, gauss_size)
    # 3x3 square structuring element
    # this structuring element seems to be important
    selem = skimage.morphology.square(selem_size)

    # Do max filter and min filter
    af488_LoG_max = ndi.filters.maximum_filter(af488_LoG, footprint=selem)
    af488_LoG_min = ndi.filters.minimum_filter(af488_LoG, footprint=selem)

    # Image of zero-crossings
    af488_edge = ((af488_LoG >= 0) & (af488_LoG_min < 0)) | ((af488_LoG <= 0) & (af488_LoG_max > 0))

    # Find zero-crossings
    # lower --> more dots, higher --> less dots
    # lower is needed for bigger dots (0.2 seems good)
    af488_edge_zero = zero_crossing_filter(af488_LoG, zero_crossing_filter_size)
    return af488_edge_zero, af488_LoG, af488_edge

def zero_crossing_filter(im, thresh):
    """
    Returns image with 1 if there is a zero crossing and 0 otherwise.

    thresh is the the minimal value of the gradient, as computed by Sobel
    filter, at crossing to count as a crossing.
    """
    # Square structuring element
    # the bigger this is, the thicker the lines are 
    selem = skimage.morphology.square(5)

    # Do max filter and min filter
    im_max = ndi.filters.maximum_filter(im, footprint=selem)
    im_min = ndi.filters.minimum_filter(im, footprint=selem)

    # Compute gradients using Sobel filter
    im_grad = skimage.filters.sobel(im)

    # Return edges
    return ( (  ((im >= 0) & (im_min < 0))
              | ((im <= 0) & (im_max > 0)))
            & (im_grad >= thresh) )

def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        cos_distances = []
        for a, b in enumerate(zip(df['x'].values, df['y'].values)):
            cos_distances.append(scipy.spatial.distance.cosine(list(b), centroids[i]))
        df['distance_from_{}'.format(i)] = cos_distances
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    return df

def update(centroids, df):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids

def k_means_cluster_cos(x, y, k, x_upper, y_upper):
    '''
    Function to compute k-means clustering using cosine distance as distance metric.
    ----------
    Parameters
    ----------
    x: array
        array of x values
    y: array
        array of y values
    k: int
        number of clusters
    x_upper: int
        x upper-bound to pick random centroids from
    y_upper: int
        y upper-bound to pick random centroids from
    --------
    Returns
    --------
    df: DataFrame
        dataframe with each point assigned to a cluster and its cosine distance from the cluster center
    '''
    
    # initialization
    df = pd.DataFrame({'x': x,
                       'y': y})
    centroids = {i+1: [np.random.randint(0, x_upper), np.random.randint(0, y_upper)] for i in range(k)}
    print(centroids)
    
    # assignment
    df = assignment(df, centroids)
    
    # update
    old_centroids = copy.deepcopy(centroids)
    centroids = update(centroids, df)
    
    # Continue until all assigned categories don't change any more
    while True:
        closest_centroids = df['closest'].copy(deep=True)
        centroids = update(centroids, df)
        df = assignment(df, centroids)
        if closest_centroids.equals(df['closest']):
            break
    return df