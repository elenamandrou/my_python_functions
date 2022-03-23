# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:25:20 2020

@author: Elena
"""
import matplotlib.pyplot as plt

import skimage.io

def image_vs_image(image1,image2, figsize = [12,4], axis_status = None, cmap = 'gray'):
    """ 
    Puts 2 images (np.ndarray) side by side so you can compare them. 
    Uses plt.imshow(). Can specify figsize=[,] and turn off axes
    
    """
    plt.figure(figsize = figsize)
    plt.subplot(1,2,1)
    plt.imshow(image1, cmap = cmap)
    if axis_status == 'off':
        plt.axis(axis_status)
    plt.subplot(1,2,2)
    plt.imshow(image2, cmap=cmap)
    if axis_status == 'off':
        plt.axis(axis_status)

#########################################################
def split_to(img,channel):
    """Takes a multi-channel img of (x,y,channel) shape and separates the channel provided"""
    img=img[:,:,int(channel)]
    return img
#########################################################
from skimage.filters import roberts, sobel, scharr, prewitt, farid
from skimage import feature, img_as_float

def best_edge(img,canny_sigma = 3):
    """To check roberts, sobel, scharr, prewitt, farid and canny edge detection filters. Can modify canny_sigma. Returns list of 6 images"""
    roberts0 = roberts(img)    
    sobel1 = sobel(img)
    scharr2 = scharr(img)
    prewitt3 = prewitt(img)
    farid4 = farid(img)
    canny5 = feature.canny(img,sigma=canny_sigma)
    edge_imgs = [roberts0,sobel1,scharr2,prewitt3,farid4,canny5]
    return edge_imgs
    
####################################################################
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma, unsupervised_wiener
from skimage import img_as_float, img_as_float32
from scipy import ndimage as nd
import cv2

def best_denoised(img):
    """
   To check NLM, gaussian, median, bilateral blur denoise filters. Returns list of 4 images
    """
    float_img = img_as_float(img)
    float32_img = img_as_float32(img)
    
    sigma_est = np.mean(estimate_sigma(float_img, multichannel = True))

    nlm0 = denoise_nl_means(float_img, h = 0.8 * sigma_est, fast_mode = False,patch_size = 5, patch_distance = 3, multichannel = True)
    
    gaussian1 = nd.gaussian_filter(float_img, sigma = 3)

    median2 = nd.median_filter(float_img, size = 3)
    
    bilateral_blur3 = cv2.bilateralFilter(float32_img, 9, 75, 75)
    
    denoised_imgs = [nlm0, gaussian1, median2, bilateral_blur3]
    return denoised_imgs
######################################################################
import numpy as np
import matplotlib.pyplot as plt 

def flimhist(donor, fret, histrange = (1,4)):

    plt.hist(donor.flat, range = histrange, bins = 100, color = "green")

    plt.hist(fret.flat, range = histrange, bins = 100, color = "brown")    
    
    plt.title("Donor", loc = "right")

    plt.title("FRET", loc = "left")

    plt.show()
    
def tau_mean(donor, fret, mintau = 3, maxtau = 7):
    thresholded_donor_lifetime = []
    for i in donor.flat:
        if mintau<i<maxtau:
            thresholded_donor_lifetime.append(i)
    donor_mean_lifetime = np.mean(thresholded_donor_lifetime)
    print(donor_mean_lifetime)

    thresholded_fret_lifetime=[]

    for z in fret.flat:
        if mintau<z<maxtau:
            thresholded_fret_lifetime.append(z)
        
    fret_mean_lifetime = np.mean(thresholded_fret_lifetime)
    print(fret_mean_lifetime)
    
def single_tau_mean(fluorophore, mintau = 3, maxtau = 7):                                            
    thresholded_fluorophore_lifetime = []
    for i in fluorophore.flat:
        if mintau<i<maxtau:
            thresholded_fluorophore_lifetime.append(i)
    fluorophore_mean_lifetime = np.mean(thresholded_fluorophore_lifetime)
    print(fluorophore_mean_lifetime)
    
##########################################################
def all_cmaps(img):

    cmaps = ["Accent", "Blues", "BrBG", "BuGn", "BuPu", "CMRmap", "Dark2", "GnBu", "Greens", "Greys", "OrRd", "Oranges", "PRGn", "Paired", "Pastel1", "Pastel2", "PiYG", "PuBu", "PuBuGn","PuOr", "PuRd", "Purples", "RdBu", "RdGy", "RdPu", "RdYlBu", "RdYlGn", "Reds", "Set1", "Set2", "Set3", "Spectral", "Wistia", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd", "afmhot", "autumn", "binary", "bone", "brg", "bwr", "cividis", "cool", "coolwarm", "copper", "cubehelix", "flag", "gist_earth", "gist_gray", "gist_heat", "gist_ncar", "gist_rainbow", "gist_stern", "gist_yarg", "gnuplot", "gnuplot2", "gray", "hot", "hsv", "inferno", "jet", "magma", "nipy_spectral", "ocean", "pink", "plasma", "prism", "rainbow", "seismic", "spring", "summer", "tab10", "tab20", "tab20b", "tab20c", "terrain", "twilight", "twilight_shifted", "viridis", "winter"]

    for i in cmaps:
        plt.imshow(img, cmap = "{0}".format(i))
        plt.title(i)
        plt.show()

############################################################
import glob
from skimage import io

def my_glob(path):
    """quick import"""
    img_list = []
    for file in glob.glob(path):
        img = img_as_float(io.imread(file))        
        img_list.append(img)
    return img_list
########################################################

def dataset_mean_tau(FLIMfit_dataset):
    """finds mean lifetime"""
    FLIMfit_dataset_taus = FLIMfit_dataset.loc[:, 'mean - tau_1']

    print("--> Taus (ps) in this dataset: ", FLIMfit_dataset_taus)

    mean_FLIMfit_dataset_taus = np.mean(FLIMfit_dataset_taus)

    print("--> Mean tau (ps) of taus in the dataset: ", mean_FLIMfit_dataset_taus)
    
########################################################

def cvclose():
    """quick cv2 close"""
    cv2.waitKey(0)          
    cv2.destroyAllWindows()
    
#############################################################

def OCV_split(img):
    """Splits rgb channel image"""
    r = img.copy()
    r[:, :, 0] = r[:, :, 1] = 0
    
    g = img.copy()
    g[:, :, 0] = g[:, :, 2] = 0
    
    b = img.copy()
    b[:, :, 1] = b[:, :, 2] = 0
    
    cv2.imshow("red", r)
    cv2.imshow("green", g)
    cv2.imshow("blue", b)
    cv2.imshow("img", img)
    
    cvclose()
    
    
##################################################################
from matplotlib import pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation

def my_swarm(data, order = [], y = 'mean - tau_1'):
    """Creates swarm plot from .csv FLIM data"""
    sns.set_theme(style = "ticks", color_codes = True)
    sns.catplot(x = "SampleName", y = y, kind = "swarm", data = data, order = order)
    
###################################################################

def stat_box(data, order = [], pairs = [], y = "mean - tau_1"):
    """Statistical annotation for .csv FLIM data"""
    ax=sns.boxplot(x = "SampleName", y = y, order = order, data = data, palette = "Dark2")

    add_stat_annotation(ax, data = data, x = "SampleName", y = y, order = order,box_pairs = pairs, test = 't-test_ind', comparisons_correction = None, text_format = 'star', loc = 'inside', verbose = 2)
####################################################################################

def active_fraction(df, sample_name, thresh):
    """Takes dataframe, SampleName and lifetime cutoff of interest"""
    sample_sum = df[df['SampleName'] == sample_name]['SampleName'].count()
        
    sample_active = df[(df['mean - tau_1'] > thresh)&(df['SampleName'] == sample_name)]['SampleName'].count()
    
    pct_sample_active = (sample_active*100)/sample_sum
    print("%s%%"%"{:.2f}".format(pct_sample_active))
