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
    return chfp

def show_two_ims(
    im_1,
    im_2,
    titles=[None, None],
    interpixel_distances=[0.13, 0.13],
    cmap=[None, None]
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
    """
    p_1 = bebi103.image.imshow(
        im_1,
        frame_height=225,
        title=titles[0],
        cmap=cmap[0],
        #interpixel_distance=interpixel_distances[0],
        #length_units="µm",
    )
    p_2 = bebi103.image.imshow(
        im_2,
        frame_height=225,
        title=titles[1],
        cmap=cmap[1],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
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
):
    """Convenient function for showing three images side by side."""
    p_1 = bebi103.image.imshow(
        im_1,
        frame_height=225,
        title=titles[0],
        cmap=cmap[0],
        #interpixel_distance=interpixel_distances[0],
        #length_units="µm",
    )
    p_2 = bebi103.image.imshow(
        im_2,
        frame_height=225,
        title=titles[1],
        cmap=cmap[1],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
    )
    p_3 = bebi103.image.imshow(
        im_3,
        frame_height=225,
        title=titles[2],
        cmap=cmap[2],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
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
):
    """Convenient function for showing four images side by side."""
    p_1 = bebi103.image.imshow(
        im_1,
        frame_height=225,
        title=titles[0],
        cmap=cmap[0],
        #interpixel_distance=interpixel_distances[0],
        #length_units="µm",
    )
    p_2 = bebi103.image.imshow(
        im_2,
        frame_height=225,
        title=titles[1],
        cmap=cmap[1],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
    )
    p_3 = bebi103.image.imshow(
        im_3,
        frame_height=225,
        title=titles[2],
        cmap=cmap[2],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
    )
    p_4 = bebi103.image.imshow(
        im_4,
        frame_height=225,
        title=titles[3],
        cmap=cmap[3],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
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
):
    """Convenient function for showing five images side by side."""
    p_1 = bebi103.image.imshow(
        im_1,
        frame_height=225,
        title=titles[0],
        cmap=cmap[0],
        #interpixel_distance=interpixel_distances[0],
        #length_units="µm",
    )
    p_2 = bebi103.image.imshow(
        im_2,
        frame_height=225,
        title=titles[1],
        cmap=cmap[1],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
    )
    p_3 = bebi103.image.imshow(
        im_3,
        frame_height=225,
        title=titles[2],
        cmap=cmap[2],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
    )
    p_4 = bebi103.image.imshow(
        im_4,
        frame_height=225,
        title=titles[3],
        cmap=cmap[3],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
    )
    p_5 = bebi103.image.imshow(
        im_5,
        frame_height=225,
        title=titles[4],
        cmap=cmap[4],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
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
):
    """Convenient function for showing six images side by side."""
    p_1 = bebi103.image.imshow(
        im_1,
        frame_height=225,
        title=titles[0],
        cmap=cmap[0],
        #interpixel_distance=interpixel_distances[0],
        #length_units="µm",
    )
    p_2 = bebi103.image.imshow(
        im_2,
        frame_height=225,
        title=titles[1],
        cmap=cmap[1],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
    )
    p_3 = bebi103.image.imshow(
        im_3,
        frame_height=225,
        title=titles[2],
        cmap=cmap[2],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
    )
    p_4 = bebi103.image.imshow(
        im_4,
        frame_height=225,
        title=titles[3],
        cmap=cmap[3],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
    )
    p_5 = bebi103.image.imshow(
        im_5,
        frame_height=225,
        title=titles[4],
        cmap=cmap[4],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
    )
    p_6 = bebi103.image.imshow(
        im_6,
        frame_height=225,
        title=titles[5],
        cmap=cmap[5],
        #interpixel_distance=interpixel_distances[1],
        #length_units="µm",
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
    dapi_filt_gauss = skimage.filters.gaussian(dapi, 1.5)
    cfp_filt_gauss = skimage.filters.gaussian(cfp, 1.5)
    chfp_filt_gauss = skimage.filters.gaussian(chfp, 1.5)
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

def ws(image, min_size, intensity_image):
    '''
    Function to perform watershed and remove small objects. Will generate a dataframe and output total area
    of the watershed cells.
    
    ----------
    Parameters
    ----------
    image : array
        image should be filtered and segmented
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
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((25,25)),
                                labels=image)
    markers = ndi.label(local_maxi)[0]
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
