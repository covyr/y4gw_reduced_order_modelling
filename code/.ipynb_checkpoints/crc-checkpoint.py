##############################################################################

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
warnings.filterwarnings("ignore", "external/local_xla/xla/")
warnings.filterwarnings("ignore", "TensorRT")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import os
import json
import copy
import sys
import pickle
from pycbc.types import *
from typing import Union, Optional
import time

##############################################################################

crc972home = '/rds/homes/c/crc972/connor'
crc972wdir = '/rds/homes/c/crc972/connor/code'

# set rc_params for plotting

rc_params = {
    'axes.grid'             : False,
    'axes.labelsize'        : 24,
    'axes.linewidth'        : 1,
    'axes.titlesize'        : 24,
    'font.size'             : 24,
    'legend.fontsize'       : 16,
    'xtick.labelsize'       : 16,
    'ytick.labelsize'       : 16,
    # 'font.family'           :'sans-serif',
    'font.family'           :'serif',
    'text.latex.preamble'   : r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}',
    'text.usetex'           : False,
    'patch.force_edgecolor' : True,
    'figure.dpi'            : 256,
    'mathtext.fontset'      : 'cm'     
}

mpl.rcParams.update(rc_params)
# plt.rcParams.update(rc_params)

##############################################################################

# GLOBAL STUFF

def GET_TIME() -> np.ndarray:
    
    """
    gets the common time array used for every waveform (in geometric units: 
    T/M)
    """
    
    return np.load(f"{crc972home}/treasure/TIME.npy")

TIME = GET_TIME()


ITEMS = {
    'scenario' : {'NS', 'AS', 'P'},
    # 'mode'     : {'2,2', '4,4', '4,3'},
    'mode'     : {'2,2', '2,1', '3,3', '3,2', '4,4', '4,3'},
    'component': {'full', 'A', 'phi'}
}

def ASSERT(scenario: str = None, mode: str = None, component: str = None) -> None:
    
    """
    used to assert compatibility of frequently used, key variables
    """
    
    args = copy.deepcopy(locals())
    for arg in args:
        if args[arg]:
            assert args[arg] in ITEMS[arg]
    
    return

##############################################################################

def get_colours(n: int = 1):
    
    """
    returns n random colours
    """
    
    p = lambda: random.randint(0, 255)
    hex_colour  = lambda: "#{:02x}{:02x}{:02x}".format(p(), p(), p())
    hex_colours = []
    for i in range(n):
        hex_colours.append(hex_colour())
    
    return hex_colours[0] if n == 1 else hex_colours

COLOURS = {'2,2': '#51bca5', '2,1': '#547483', '3,3': '#0b3d8e', '3,2': '#a50f5f', '4,4': '#b5b367', '4,3': '#d3aad8'}
COLOURS_opp = {'2,2': '#ae435a', '2,1': '#ab8b7c', '3,3': '#f4c271', '3,2': '#5af0a0', '4,4': '#4a4c98', '4,3': '#2c5527'}

def GET_COLOURS():
    return COLOURS
def GET_COLOURS_opp():
    return COLOURS_opp

def comp_hex(hex: str):
    r, g, b = crc.hex_to_rgb(hex)
    rc, gc, bc = 255 - r, 255 - g, 255 - b
    return crc.rgb_to_hex((rc, gc, bc))

def colour_fam(colour_family):
    fig, ax = plt.subplots(1, 1, figsize = (12, 0.5))
    for i in range(len(colour_family)):
        plt.scatter(i, 1, s = 256, color = colour_family[i], marker = 's')
    if len(colour_family) < 16:
        ax.set_xticks(np.array([n for n in range(len(colour_family))]))
        ax.set_xticklabels(np.array(colour_family))
    else:
        ax.set_xticks(np.array([n for n in range(len(colour_family))])[::len(colour_family) // 16])
        ax.set_xticklabels(np.array(colour_family)[::len(colour_family) // 16])
    ax.set_yticks([])
    # plt.tight_layout()
    plt.show()
    return

hex_rgb = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'a': 10,
    'b': 11,
    'c': 12,
    'd': 13,
    'e': 14,
    'f': 15
}

rgb_hex = {value: key for key, value in hex_rgb.items()}

def hex_to_rgb(hex):
    hex = hex[1:]
    r = 16 * hex_rgb[hex[0]] + hex_rgb[hex[1]]
    g = 16 * hex_rgb[hex[2]] + hex_rgb[hex[3]]
    b = 16 * hex_rgb[hex[4]] + hex_rgb[hex[5]]
    return (r, g, b)

def rgb_to_hex(rgb):
    r, g, b = rgb
    hex_0 = r // 16
    hex_1 = ((r / 16) - hex_0) * 16
    hex_2 = g // 16
    hex_3 = ((g / 16) - hex_2) * 16
    hex_4 = b // 16
    hex_5 = ((b / 16) - hex_4) * 16
    return f"#{rgb_hex[hex_0]}{rgb_hex[hex_1]}{rgb_hex[hex_2]}{rgb_hex[hex_3]}{rgb_hex[hex_4]}{rgb_hex[hex_5]}"

def sort_rgb(rgbs_in):
    rgbs = np.copy(rgbs_in)
    oinds = [n for n in range(1, len(rgbs_in))]
    cinds = [0]
    for i in range(1, len(rgbs_in)):
        cdelt = np.array([(rgbs[cinds[i-1]][0] - r) ** 2 + (rgbs[cinds[i-1]][1] - g) ** 2 + (rgbs[cinds[i-1]][2] - b) ** 2 for r, g, b in rgbs])
        cdelt[cinds] += 3 * 255 ** 2
        cinds.append(int(np.argmin(cdelt)))
        np.delete(rgbs, int(np.argmin(cdelt)))
    return np.array(rgbs_in)[cinds]

def sort_hex(hexs):
    rgbs = [hex_to_rgb(hex) for hex in hexs]    
    sorted_rgbs = sort_rgb(rgbs)
    return [rgb_to_hex(rgb) for rgb in sorted_rgbs]
        
##############################################################################

def Normal(mean=0.5, std_dev=0.2, num_samples=5251):
    
    """
    make a large normal distribution to sample from
    """
    
    normal = np.random.normal(mean, std_dev, num_samples**2)
    normal = normal[np.where(normal >= 0)[0]]
    normal = normal[np.where(normal <= 1)[0]]
    
    return normal



def Uniform(min=0, max=1, num_samples=5251):
    
    """
    make a large uniform distribution to sample from [0, 1]
    """
    
    return np.random.uniform(min, max, num_samples**2)

def Uniform_Double(min=-1, max=1, num_samples=5251):

    """
    make a large uniform distribution to sample from [-1, 1]
    """
    
    return np.random.uniform(min, max, num_samples**2)



def Gamma(k: float = 7/4, theta: float = 1/5, num_samples: int = 5251) -> np.ndarray:
    
    """
    make a large gamma distribution to sample from
    """

    gamma = np.random.gamma(k, theta, num_samples**2)
    gamma = gamma[np.where(gamma >= 0)[0]]
    gamma = gamma[np.where(gamma <= 1)[0]]
    
    return gamma

##############################################################################

# Define functions to sample r, theta, phi of spins from distributions
# Want isotropic angles, and i settled on a gamma distribution for magnitude as per
# Citation R. Abbott et al 2021 ApJL 913 L7
# DOI 10.3847/2041-8213/abe949

def sample_r(scenario, n: int) -> np.ndarray:
    
    """
    sample spin magnitude from uniform distribution in:
        AS: [-1, 1]
        P : [ 0, 1]
    a bit about choice of distribution...
    """

    length = {0: 1000, 1: 1000, 2: 10000, 3: 65536, 4: 65536}
    num_samples = length[n]
    # num_samples = {0: 1000, 1: 1000, 2: 10000, 3: 65536, 4: 65536}[n] if n else 1

    if scenario == 'AS':
        return np.random.choice(Uniform_Double(), size=num_samples, replace=False)
    elif scenario == 'P':
        return np.random.choice(Uniform(), size=num_samples, replace=False)

def sample_theta(n: int) -> np.ndarray:
    
    """
    sample spin angle theta from uniform distribution in [-pi/2, +pi/2]
    uniform distribution -> isotropic spin directions
    """

    length = {0: 1000, 1: 1000, 2: 10000, 3: 65536, 4: 65536}
    num_samples = length[n]
    
    # return np.random.choice(Uniform(), size=num_samples, replace=False)*np.pi - np.pi/2
    return np.random.choice(Uniform(), size=num_samples, replace=False) * np.pi/2 - np.pi

def sample_phi(n: int) -> np.ndarray:
    
    """
    sample spin angle phi from uniform distribution in [-pi, +pi]
    uniform distribution -> isotropic spin directions
    """

    length = {0: 1000, 1: 1000, 2: 10000, 3: 65536, 4: 65536}
    num_samples = length[n]
    
    # return np.random.choice(Uniform(), size=num_samples, replace=False)*2*np.pi -np.pi
    return np.random.choice(Uniform(), size=num_samples, replace=False) * np.pi - 2 * np.pi

def sample_spin(scenario, n: int):

    """
    combination of the above 3 functions for convenience and conciseness
    """
    
    return sample_r(scenario, n), sample_theta(n), sample_phi(n)

##############################################################################

def matrix_basis(basis: dict) -> np.ndarray:
    
    """
    returns the 2d numpy array of the waveforms in a basis dictionary
    """
    
    return np.vstack(list(basis.values()))

##############################################################################

def dot(h1: np.ndarray, h2: np.ndarray, xs: np.ndarray) -> np.ndarray:
    
    """
    returns the dot product of 2 vectors {h1, h2} across {xs}
    """
    
    return np.trapz(np.conjugate(h1) * h2, xs)

def mag(h: np.ndarray, xs: np.ndarray) -> np.ndarray:

    """
    returns the magnitude of a vector
    """
    
    return np.sqrt(dot(h, h, xs))

def mag2(h: np.ndarray, xs: np.ndarray) -> np.ndarray:

    """
    returns the magnitude^2 of a vector
    """
    
    return dot(h, h, xs)

##############################################################################

def normalise(h: np.ndarray, xs: np.ndarray) -> np.ndarray:
    
    """
    returns e: the vector h divided by its magnitude (normalised)
    """

    return h / mag(h, xs)

##############################################################################

def project(h: np.ndarray, rb: Union[np.ndarray, dict], xs: np.ndarray) -> np.ndarray:
    
    """
    returns the projection of vector h onto basis (a set of basis vectors)
    """
    
    if type(rb) == dict:
        rb = matrix_basis(rb)
            
    if rb.ndim == 1:
        projection = dot(h, rb, xs) * rb
    if rb.ndim > 1:
        projection = np.sum([dot(h, rb[i], xs) * rb[i] for i in range(rb.shape[0])], axis = 0)
    
    return projection

##############################################################################

def test_orth(basis, xs):

    """
    test the orthogonality of a basis
    """
    
    orth = []
    for m in basis:
        for i in basis:
            if m != i:
                ip = dot(basis[m], basis[i], xs) # <ei, ej> = 0 (i != j) if ei and ej are orthogonal
                
                if ip != 0: # looking for least orthogonal combination, so can ignore the 'perfectly' orthogonal combinations
                    O = np.log10(np.abs(ip)) # looking at the order of magnitude of the deviation from zero is more convenient than reading heaps of zeroes
                    orth.append(O) # add deviation from zero to a list
    
    return(f"orth: 10^({np.max(orth):.3f})") # show the dot product between the least orthogonal vectors

##############################################################################

def test_norm(basis, xs):
    
    """
    test the normality of a basis
    """
    
    norm = []
    for m in basis:
        ip = dot(basis[m], basis[m], xs) # <ei, ei> = 1 if ei is a normalised vector
        
        if ip != 1: # looking for least normalised vector, so can ignore the 'perfectly' normalised vectors
            N = np.log10(np.abs(np.abs(ip)-1)) # looking at the orfer of magnitude of the deviation from one is more convenient than looking at heaps of zeroes or ones
            norm.append(N) # add deviation from one to a list
    
    return(f"norm: 10^({np.max(norm):.3f})") # show dot product between the least normalised vector and itself

##############################################################################

def match(h1: np.ndarray, h2: np.ndarray, M_sol:int = 50) -> float:
    
    """
    calculates the overlap of waveforms {h1, h2}
    takes mass = 50 solar masses as convention
    returns overlap
    """
    
    import lal, lalsimulation
    # lal.swig_redirect_standard_output_error(True)
    from pycbc import waveform as pywaveform
    from pycbc import filter as pyfilter
    from pycbc import detector as pydetector
    from pycbc import psd as pypsd
    
    dt = M_sol * lal.MTSUN_SI # dt in geometric time, t/M, is 1. so dt in seconds is 50 solar masses (consistency choice)
    
    h1_ts = timeseries.TimeSeries(h1.real, delta_t = dt)
    h2_ts = timeseries.TimeSeries(h2.real, delta_t = dt)
    t_len  = max(len(h1_ts.sample_times), len(h2_ts.sample_times))
    
    delta_f = 1 / h1_ts.duration
    f_len = t_len//2 + 1
    f_min = 20
    # f_max = 4096
    f_max = 2048
    
    psd_aLIGO = pypsd.aLIGODesignSensitivityP1200087(f_len, delta_f, f_min) 

    overlap, _ = pyfilter.match(
        h1_ts, h2_ts, psd = psd_aLIGO, 
        low_frequency_cutoff = f_min, high_frequency_cutoff = f_max, 
        subsample_interpolation = True
    )
    
    return float(overlap)

##############################################################################

def mismatch(h1: np.ndarray, h2: np.ndarray, M_sol:int = 50) -> float:

    """
    calculates the mismatch of waveforms {h1, h2}
    takes mass = 50 solar masses as convention
    returns mismatch
    """
    
    return float(1 - match(h1, h2, M_sol))
    
##############################################################################

def GaussQuad(func, a, b, n: int):
    
    """
    approximates the integral of a function (func) over [a, b] via Gaussian 
    Quadrature, using n points to approximate
    """
    
    coef = [0*n for n in range(n+1)] 
    coef[n] = 1 # numpy generates a legendre series, so coef selects the polynomial of degree n
    
    P = np.polynomial.legendre.Legendre(coef) # n-th degree legendre polynomial
    roots = np.polynomial.legendre.legroots(coef) # roots of n-th degree legendre polynomial
    Pdashcoef = np.polynomial.legendre.legder(coef) # coefficients of legendre series for derivative of n-th degree legendre polynomial
    Pdash = np.polynomial.legendre.Legendre(Pdashcoef) # derivative of n-th degree legendre polynomial

    weights = []
    for i in roots:
        w = 2 / ((1-(i**2))*(Pdash(i))**2)
        weights.append(w)
        
    Sum = 0 # section below sums over the weights and accounts for limits (a, b) != (-1, 1)
    for i in range(n):
        Sum += weights[i] * func(((b-a)/2) * roots[i] + (b+a)/2)
    I = ((b-a)/2) * Sum
    
    return I # returns integral approximation

##############################################################################
    
def sort_basis(basis: dict) -> dict:
    
    """
    sorts a basis dictionary by key
    """
    
    return {q: basis[q] for q in sorted(basis)}

##############################################################################

def MGS_old(V_ini: dict, xs: np.ndarray) -> dict: ### ! THIS BROKE SOMEWHERE ALONG THE WAY ! ###
    
    """
    ! THIS BROKE SOMEWHERE ALONG THE WAY !
    """
    
    V = {i: V_ini[i] for i in list(V_ini.keys())} # make copy of V
    indexes = list(V.keys())
    U = {indexes[0]: V[indexes[0]]} # unnormalised basis (initially empty)
    uindexes = [indexes[0]]
    del V[indexes[0]]
    del indexes[0]
    
    l = 0 # initialising k for indexing throughout while loop
    while len(V) > 0:
        l += 1
        k = indexes[0]
        
        u = U[uindexes[-1]] # sets the u vector to project onto as the (k-1)-th basis vector
        for i in V:
            V[i] = V[i] - project(V[i], U, xs) # updates the i-th vector in the working set
            
        U.update({k: normalise(V[k], xs)})
        
        uindexes.append(k)
        del V[k] # remove v0 from the working set
        del indexes[0]
                    
    
    for i in list(U.keys()):
        U[i] = normalise(U[i], xs)
        
    return U # returns an orthonormal basis of m vectors   



def MGS(V_in: dict, xs: np.ndarray) -> dict:

    """
    """
    
    ks = list(V_in.keys())
    M = len(ks)
    V = matrix_basis(copy.deepcopy(V_in))
        
    i = 0
    U = {ks[i]: normalise(V[0], xs)}
    V = V[1:, :]
    i += 1

    while i < M:
        
        for j in range(V.shape[0]):
            V[j] -= project(V[j], U, xs)
        
        U.update({ks[i]: normalise(V[0], xs)})
        V = V[1:, :]
        i += 1
        
    return U


        
def MGS_step(rb: dict, remaining_waveforms: dict, xs: np.ndarray) -> dict:
    
    """
    progresses a whole basis by one step in the MGS algorithm such that each 
    element becomes the orthogonal projection of the original element onto the
    last reduced basis element
    
    Args:
        rb (dict): reduced basis
        waveforms (dict): basis (rb removed)
        basis (dict): basis (original)
        xs (np.ndarray): 

    Returns:
        dict: 
    """
    
    keyswf = list(remaining_waveforms.keys())
    keysrb = list(rb.keys())
    new_rb_element = rb[keysrb[-1]]
    
    for i in keyswf:
        remaining_waveforms[i] -= project(remaining_waveforms[i], new_rb_element, xs)
    
    return remaining_waveforms
    
##############################################################################

# Define a functtion to calculate the greedy error

def RB_error(h: np.ndarray, U, xs: np.ndarray) -> float:
    
    """
    """
    
    diff = h - project(h, U, xs)
        
    return dot(diff, diff, xs)

##############################################################################

# Define function to make a REDUCED BASIS

def RB(V: dict, xs: np.ndarray, tolerance: float):

    """
    """
    vectors = copy.deepcopy(V)
    # plt.plot(matrix_basis(vectors))
    M = len(vectors)
    og_keys = list(vectors.keys())
    
    training_keys = list(vectors.keys()) # list of keys of vectors left in the training set
        
    # (l, m) modes with odd m have a zero-waveform for q=1 and is a waste of a RB element
    ind = np.argmax([dot(vectors[i], vectors[i], xs) for i in training_keys])
    
    key = training_keys[ind]
    rb = {key: normalise(vectors[key], xs)} # dict to add reduced basis vectors to (initialise with first vector provided)
        
    indexes = [ind]
    
    del vectors[key] # removing the first reduced basis vector from the training set
    del training_keys[ind] # removing the key of first reduced basis vector from training_keys list
    vectors = MGS_step(rb, vectors, xs)
    
    greedy_errors = [1] # append greedy error leading to basis vector selection here
    greedy_error = 1 # prescribe greedy vector as bigger than tolerance so the function will work
    m = 0 # initialising a counter
    
    while greedy_error > tolerance: # this loop selects vectors for the reduced basis until the greedy error comes beneath the tolerance
        m += 1
        
        if m < M: # this loop will take every vector left in the training set and select the one with the largest disagreement with the reduced basis up to here
                        
            sigmas = [dot(vectors[key], vectors[key], xs) for key in training_keys]
            
            greedy_error = np.max(np.array(sigmas)) # determine greedy error
            node = sigmas.index(greedy_error) # get index of the vector who gave the greedy error   
            greedy_errors.append(greedy_error) # add greedy error to list of errors which will be returned by the function (good for plotting etc)    
            
            key = training_keys[node]          
            # rb.update({key: normalise(V[key], xs)}) # add vector who gave the greedy error, to the RB ##############VVVVVVVVVVVVVVVVVVVvv
            rb.update({key: vectors[key]}) # add vector who gave the greedy error, to the RB ##############VVVVVVVVVVVVVVVVVVVvv
            rb = MGS(rb, xs)
            
            ind = og_keys.index(key)
            indexes.append(ind) # NEED TO GET INDEX OF VECTOR IN OG SET THAT THE GREEDY ERROR COMES FROM (FROM INCOMPLETE SET)
            
            
            del vectors[key] # remove new reduced basis vector from training set
            del training_keys[node] # remove key of new reduced basis vector from the training_keys list
            vectors = MGS_step(rb, vectors, xs)
            
            print(f"{m+1}/{M} vectors in rb, greedy error: 10^({np.log10(greedy_error):.3f})            ", end='\r')
            
        elif m >= M: # if all vectors exhausted
            print(f"out of vectors: tolerance of {tolerance} not satisfied                         ")
            print(test_orth(rb, xs)) # test orthogonality
            print(test_norm(rb, xs)) # test normality
            
            return rb, errors, indexes
        
    
    print(f"tolerance met: {len(rb)}/{M} vectors in reduced basis                     ")
    print(test_orth(rb, xs)) # test orthogonality
    print(test_norm(rb, xs)) # test normality
        
    return rb, greedy_errors, indexes



def new_rb_element(h_m: np.ndarray, rb: dict, xs: np.ndarray):

    """
    """
    
    return normalise(h_m - project(h_m, rb, xs), xs)


    
def RB_old(V: dict, xs: np.ndarray, tolerance: float):
    
    """
    """
    
    vectors = {i: V[i] for i in set(list(V.keys()))} # make a copy of V, so to leave V unedited
    
    training_keys = list(vectors.keys()) # list of keys of vectors left in the training set
    
    ini = np.argmax([dot(vectors[i], vectors[i], xs) for i in list(vectors.keys())])
    # ini = 0
    key = training_keys[ini]
    indexes = [ini]
    
    rb = {key: normalise(V[key], xs)} # dict to add reduced basis vectors to (initialise with first vector provided)
    rb_keys = [key] # add key of first reduced basis vector to rb_keys list
    
    
    del vectors[key] # removing the first reduced basis vector from the training set
    del training_keys[ini] # removing the key of first reduced basis vector from training_keys list
    
    errors = [1] # append greedy error leading to basis vector selection here
    sigma_max = 1 # prescribe greedy vector as bigger than tolerance so the function will work
    i = 0 # initialising a counter
    
    while sigma_max >= tolerance: # this loop selects vectors for the reduced basis until the greedy error comes beneath the tolerance
        i += 1
        
        if i < len(V): # this loop will take every vector left in the training set and select the one with the largest disagreement with the reduced basis up to here
            sigmas = [RB_error(vectors[v], rb, xs) for v in list(vectors.keys())] # make list to add greedy errors to            
            sigma_max = np.max(np.array(sigmas)) # determine greedy error
            sigma_max_index = sigmas.index(sigma_max) # get index of the vector who gave the greedy error   
            errors.append(sigma_max) # add greedy error to list of errors which will be returned by the function (good for plotting etc)
            
            # rb.update({training_keys[sigma_max_index]: vectors[training_keys[sigma_max_index]]}) # add vector who gave the greedy error, to the RB
            rb.update({training_keys[sigma_max_index]: normalise(vectors[training_keys[sigma_max_index]], xs)}) # add vector who gave the greedy error, to the RB
            rb_keys.append(training_keys[sigma_max_index]) # add key of this new reduced basis vector to the rb_keys list
            
            ind = list(V.keys()).index(training_keys[sigma_max_index])
            indexes.append(ind) # NEED TO GET INDEX OF VECTOR IN OG SET THAT THE GREEDY ERROR COMES FROM (FROM INCOMPLETE SET)
            
            del vectors[training_keys[sigma_max_index]] # remove new reduced basis vector from training set
            del training_keys[sigma_max_index] # remove key of new reduced basis vector from the training_keys list
    
            rb = MGS(rb, xs) # perform modified Gram-Schmidt process on the reduced basis
            print(f"{i+1}/{len(V)} vectors in rb, greedy error: 10^({np.log10(sigma_max):.3f})       ", end='\r')
            
        elif i >= len(V): # if all vectors exhausted
            print(f"out of vectors: tolerance of {tolerance} not satisfied                         ")
            print(test_orth(rb, xs)) # test orthogonality
            print(test_norm(rb, xs)) # test normality
            
            return rb, errors
    
    print(f"tolerance met: {len(rb)}/{len(V)} vectors in reduced basis                     ")
    print(test_orth(rb, xs)) # test orthogonality
    print(test_norm(rb, xs)) # test normality
        
    return rb, errors, indexes
    
##############################################################################

# Define function to PLOT (REDUCED) BASIS VECTORS IN ONE SUBPLOT

def plot_rb(rb: dict, xs: np.ndarray, Legendre: bool = False):
    # rb            : reduced basis
    # xs            : time
    # normalisation : used for plotting Legendre polynomials
    
    """
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    
    if Legendre:
        # ax.set_title(f'reduced basis vectors $e_i$ against $x$ (vectors have been divided by $sqrt(L+1/2)$ for visualisation purposes)');
        for _, i in enumerate(rb):
            ax.plot(xs, rb[i] / np.sqrt(i+1/2), color = get_colours())
            
    elif not Legendre:
        ax.set_title(fr"reduced basis elements ${{e_i(t)}}_{{i=1}}^m$")
        for _, i in enumerate(rb):
            ax.plot(xs, rb[i], color = get_colours(), label = fr"$q = {i:.3f}$")
            
    fig.tight_layout()
    plt.show()
    
    return 



def plot_basis(rb: dict, xs: np.ndarray, Legendre: bool = False, component: str = 'h'):
    # rb            : reduced basis
    # xs            : time
    # normalisation : used for plotting Legendre polynomials
    
    """
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    
    if Legendre:
        # ax.set_title(f'reduced basis vectors $e_i$ against $x$ (vectors have been divided by $sqrt(L+1/2)$ for visualisation purposes)');
        for _, i in enumerate(rb):
            ax.plot(xs, rb[i] / np.sqrt(i+1/2), color = get_colours())
            
    elif not Legendre:
        ax.set_title(fr"waveforms ${{{component}(t; \Lambda_i)}}_{{i=1}}^m$")
        for _, i in enumerate(rb):
            ax.plot(xs, rb[i], color = get_colours(), label = fr"$q = {i:.3f}$")

    ax.legend()
    fig.tight_layout()
    plt.show()
    
    return 

# Define function to PLOT (REDUCED) BASIS VECTORS IN SEPARATE SUBPLOTS

def subplot_rb(rb: dict, xs: np.ndarray, normalisation: bool = False):
    # rb            : reduced basis
    # xs            : time
    # normalisation : used for plotting Legendre polynomials
    
    """
    """
    
    fig, axs = plt.subplots(nrows=len(rb), ncols=1, figsize=(10, 2*len(rb)))
    
    for n, i in enumerate(rb):
        if normalisation == True:
            axs[n].plot(xs, rb[i] / np.sqrt(i+1/2))
            # axs[n].set_title(f'reduced basis vector $e_{{{n}}}$ ($v_{{L={i}}}$) against $x$ (vectors have been divided by $sqrt(L+1/2)$ for visualisation purposes)');
            
        elif normalisation == False:
            axs[n].plot(xs, rb[i])
            # axs[n].set_title(f'reduced basis vector $e_{{{n}}}$ ($v_{{L={i}}}$) against $x$');
            
    fig.tight_layout()
    return

# Define function to PLOT GREEDY ERRORS AS A FUNCTION OF SIZE OF REDUCED BASIS

def plot_greedy(errors, tolerance, param, saving: bool = False):
    # errors    : greedy errors
    # tolerance : tolerance
    # param     : which waveform ie 'A' = amplitude, 'phi' = phase
    # saving    : True = save figure : False = don't save figure
    
    """
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
    
    n = [i for i in range(1, len(errors)+1)]
    
    ax.axhline(y=np.log10(tolerance), linestyle='dashed', label=r'$\epsilon$')
    ax.plot(n, np.log10(errors))
    ax.scatter(n, np.log10(errors), label=r'$\sigma_{m}$')
    ax.set_xlim(0, n[-1]+1)
    
    yticks = []
    yticklabels = []
    for tick in ax.get_yticks():
        yticks.append(tick)
    for tick in ax.get_yticklabels():
        exponent = tick.get_text().replace('$\\mathdefault{', '').replace('}$', '')
        yticklabels.append(f'$10^{{{exponent}}}$')
    ax.set_yticks(yticks, yticklabels)
    # ax.set_xticks(n[::2])
    ax.set_ylabel(r'$\sigma_{m}$')
    ax.set_xlabel(r'$m$')
    ax.legend()
    fig.tight_layout()
    
    if saving:
        plt.savefig(f"{crc972home}/figs/prelim/greedy_errors_{param}", bbox_inches='tight')
    return

# Define function to PLOT GREEDY ERRORS  FOR BOTH RB AND EI AS A FUNCTION OF SIZE OF REDUCED BASIS

def plot_greedy_both(errors, EIerrors, tolerance: float, component, saving: bool = False):
    # errors    : greedy errors from REDUCED BASIS
    # EIerrors  : greedy errors from EMPIRICAL INTERPOLATION
    # tolerance : tolerance
    # component : which waveform ie 'A' = amplitude, 'phi' = phase
    # saving    : True = save figure : False = don't save figure
    
    """
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
    
    n = [i for i in range(1, len(errors)+1)]
    
    ax.axhline(y=np.log10(tolerance), linestyle='dashed', label=r'$\epsilon$')
    c1, c2 = get_colours(2)
    ax.plot(n, np.log10(errors), color = c1)
    ax.plot(n, np.log10(EIerrors), color = c2) # ADDED WHILE WRITING PRELIM REPORT
    ax.scatter(n, np.log10(errors), color = c1, label = r'$\sigma_{m}$')
    ax.scatter(n, np.log10(EIerrors), color = c2, label = r'$\sigma_{\mathcal{I},m}$')
    ax.set_xlim(0, n[-1]+1)
    
    yticks = []
    yticklabels = []
    for tick in ax.get_yticks():
        yticks.append(tick)
    for tick in ax.get_yticklabels():
        exponent = tick.get_text().replace('$\\mathdefault{', '').replace('}$', '')
        yticklabels.append(f'$10^{{{exponent}}}$')
    ax.set_yticks(yticks, yticklabels)
    # ax.set_xticks(n[::2]);
    ax.set_ylabel(r'$\sigma_{m}$')
    ax.set_xlabel(r'$m$')
    title = '\phi' if component == 'phi' else component
    ax.set_title(f'${title}$')
    ax.legend()
    fig.tight_layout()
    
    if saving:
        plt.savefig(f"{crc972home}/figs/prelim/greedy_errors_both_{component}", bbox_inches='tight')
    return

##############################################################################

# Define function for EMPIRICAL INTERPOLATION PROCESS

def EI(rb_in: Union[dict, np.ndarray], xs: np.ndarray):
    
    """
    """

    if type(rb_in) == dict:
        rb = matrix_basis(copy.deepcopy(rb_in))
    else:
        rb = copy.deepcopy(rb_in)
    
    m, L = rb.shape
    
    # indexes = [np.argmin(np.absolute(rb[0]))] # changing this to argmin made mismatches go down by 1e4
    # indexes = [0]
    indexes = [np.argmax(np.absolute(rb[0]))] # changing this to argmin made mismatches go down by 1e4
    
    V     = np.zeros((m, m))
    V_inv = 1 / rb[0, indexes[0]]
    
    # B  = np.zeros((m, L)) # Bj(t) built from [(j-1)x(j-1)] V
    B = rb[0] * V_inv

    # B_log = []
    # B_log.append(copy.deepcopy(B))
            
    for j in range(1, m):
        print(f"B is {j}x{j} / {m}x{m}           ", end='\r')
        
        # find r, the difference from the interpolated ei and the known full ei
        if B.ndim == 1:
            I = B * rb[1, indexes[0]]
        elif B.ndim > 1:
            I = np.sum([B[jay] * rb[j, indexes[jay]] for jay in range(j)], axis = 0)
            
        r = I - rb[j] # still i-th RB element, but j = i any
        # r = EI_h(rb[j], B, indexes) - rb[j] # still i-th RB element, but j = i any
        r[indexes] *= 0
                
        # indexes.append(np.argmax(r))
        indexes.append(np.argmax(np.absolute(r)))

        V = make_V_matrix(rb, indexes)
        
        # find inverse of updated V matrix
        V_inv = np.linalg.inv(V[:j+1, :j+1])
        # V_inv = np.linalg.inv(V)
                
        # # find new B_m with every empirical node found, to, to build the V matrix  
        B = make_B_matrix(V_inv, rb, indexes)
        
        # B_log.append(copy.deepcopy(B))
        
    print(f"B matrix complete                    ")
        
    return indexes, B#, B_log



def make_V_matrix(rb: np.ndarray, indexes: list) -> np.ndarray:

    """
    """

    m = len(indexes)
    V = rb[:m, indexes]
    # print(V.shape)
        
    return V.T



def make_B_matrix(V_inv: np.ndarray, rb: np.ndarray, indexes: list) -> np.ndarray:

    """
    """
    
    m, L = len(indexes), rb.shape[1]
    rb = copy.deepcopy(rb)[:m, :]
    B = (rb.T @ V_inv)
        
    return B.T

##############################################################################

# Define function to empirically interpolate a waveform

def EI_h(h: np.ndarray, B: np.ndarray, indexes: list):
    # h       : waveform at empirical time nodes
    # B       : 2d array of B[j](t)
    # indexes : indexes of empirical time nodes
    
    """
    """
    h = h[np.newaxis, :] if len(h) == len(indexes) else h[np.newaxis, indexes]
    # h = h[np.newaxis, indexes] if len(h) != len(indexes) else h[np.newaxis, :]
    B = B[np.newaxis, :] if B.ndim == 1 else B
    I = h @ B
    I = np.squeeze(I.T)
        
    # return np.sum([B[j] * h[j] for j, _ in enumerate(indexes)], axis = 0)
    return I

##############################################################################

# Define function to calculate EI errors

def EI_error(h, B, indexes, xs):
    # h       : waveform array
    # B       : 2d array of B[j](t)
    # indexes : indexes of empirical time nodes
    # xs      : time
    
    """
    """
    
    diff = EI_h(h, B, indexes) - h
    
    return dot(diff, diff, xs)

##############################################################################

# Define function to calculate greedy EI errors

def greedy_EI_errors(rb_in: Union[dict, np.ndarray], B_log: list, indexes: list, xs: np.ndarray) -> list:
    # rb      : reduced basis
    # B_log   : 3d array that gives the B matrix for rb size 0 to m
    # indexes : indexes of the empirical time nodes : Tj = xs[indexes[j]]
    # xs      : time
    
    """
    """
    
    if type(rb_in) == dict:
        rb = matrix_basis(rb_in)
    else:
        rb = rb_in

    M = rb.shape[0]
    m = len(indexes)
    greedy_EI_ind = [np.argmax([EI_error(rb[i], B_log[0], indexes[:1], xs) for i in range(M)])]
    greedy_EI_errors = [1]
    
    for j in range(1, m): # want m greedy errors
        
        greedy = np.array([EI_error(rb[i], B_log[j], indexes[:j+1], xs) for i in range(M)]) ### this is going wrong with list B_log ###
        greedy[greedy_EI_ind] = 0
        ind = np.argmax(greedy)
        
        greedy_EI_ind.append(ind)
        greedy_EI_errors.append(greedy[ind])
        
    return greedy_EI_errors

##############################################################################

# Define function to TEST LIKENESS OF WAVEFORM AND EMPIRICAL INTERPOLANT

def plot_mismatches(x: np.ndarray, bins: int):

    """
    """
    
    c = get_colours()
    fig, axs = plt.subplots(1, 2, figsize = (10, 5))
    logbins = np.logspace(np.log10(np.min(x)),np.log10(np.max(x)),bins+1)
    axs[0].hist(x, bins = logbins, color = c)
    axs[1].plot(np.log10(x), color = c)
    axs[0].set_xscale('log')

    # xticks = []
    # xticklabels = []
    # for tick in axs[0].get_xticks():
    #     xticks.append(tick)
    # for tick in axs[0].get_xticklabels():
    #     exponent = tick.get_text().replace('$\\mathdefault{', '').replace('}$', '')
    #     xticklabels.append(f'$10^{{{exponent}}}$')
    # axs[0].set_xticks(xticks, xticklabels)
    # axs[0].set_xlabel(r'$mismatch$')
    # axs[0].set_ylabel(r'$$')
    
    yticks = []
    yticklabels = []
    for tick in axs[1].get_yticks():
        yticks.append(tick)
    for tick in axs[1].get_yticklabels():
        exponent = tick.get_text().replace('$\\mathdefault{', '').replace('}$', '')
        yticklabels.append(f'$10^{{{exponent}}}$')
    axs[1].set_yticks(yticks, yticklabels)
    axs[1].set_ylabel(r'$mismatch$')
    axs[1].set_xlabel(r'$M$')
    
    fig.tight_layout()
    plt.show()
    
    return



def test_EI_new(basis: Union[dict, np.ndarray], B: np.ndarray, indexes: list, xs: np.ndarray) -> None:
    # rb_in   : reduced basis
    # B       : B[j, i](t)
    # indexes : indexes of the empirical time nodes : Tj = xs[indexes[j]]
    
    """
    """
    
    if type(basis) == dict:
        basis = matrix_basis(basis)

    M = basis.shape[0]
    m, L = B.shape
    
    mismatches = []
    
    for i in range(M):
        d = EI_h(basis[i], B, indexes) - basis[i]
        
        mismatches.append((dot(d, d, xs)) / (mag2(basis[i], xs)))
        # mismatches.append(dot(d, d, xs))
    
    mismatches = np.array(mismatches)
    
    # plot_mismatches(mismatches, 20)
    
    worst_mismatch = np.max(mismatches)
    i = np.argmax(mismatches)
    I = EI_h(basis[i], B, indexes)
    mismatch_out = mismatch(basis[i], I)
    print(f'worst mismatch: ~10^({np.log10(mismatch_out)})')
    
    return 

##############################################################################

# Define function to CHECK NODES FROM EI

def check_nodes(nodes):
    
    """
    """
        
    fig, ax = plt.subplots(1, 1, figsize=(12, 0.5))
    ax.scatter(nodes, np.zeros(len(nodes)), marker='x', color='k')
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()
    # fig.tight_layout()
    
    return

##############################################################################

# Define function to INTERPOLATE

def interpolate(h: np.ndarray, time_original: np.ndarray, common_time: np.ndarray = TIME) -> np.ndarray:
    # h             : waveform
    # time_original : time domain given by the SEOBNRv5HM generation
    # time_final    : common time domain dictated by user : t/M in [-5000, 250] here  
    
    """
    """
    
    import scipy
    imp_real = scipy.interpolate.InterpolatedUnivariateSpline(time_original, h.real, w=None, bbox=[None, None], k=5, ext=0, check_finite=False)
    imp_imag = scipy.interpolate.InterpolatedUnivariateSpline(time_original, h.imag, w=None, bbox=[None, None], k=5, ext=0, check_finite=False)
    h_interpolated = imp_real(common_time) + ( 1j * imp_imag(common_time) )
        
    return h_interpolated

##############################################################################

# Define function to generate waveforms

def build_directories(scenario: str, n: int) -> None:
    
    """
    builds directories for waveforms
    """
    
    ASSERT(scenario)
    
    directories_to_make = []
    
    spin_path = f"{crc972home}/datasets_new/{scenario}"
    if not os.path.isdir(spin_path):
        directories_to_make.append(spin_path)
    
    set_path = f"{spin_path}/{n}"
    if not os.path.isdir(set_path):
        directories_to_make.append(set_path)

    params_path = f"{set_path}/params"
    if not os.path.isdir(params_path):
        directories_to_make.append(params_path)

    for mode in ITEMS['mode']:
        mode_path = f"{set_path}/{mode.replace(',', '_')}"
        if not os.path.isdir(mode_path):
            directories_to_make.append(mode_path)
    
        waveforms_path = f"{mode_path}/waveforms"
        if not os.path.isdir(waveforms_path):
            directories_to_make.append(waveforms_path)

        h_path = f"{waveforms_path}/full"
        if not os.path.isdir(h_path):
            directories_to_make.append(h_path)

        bits_path = f"{mode_path}/bits"
        if not os.path.isdir(bits_path):
            directories_to_make.append(bits_path)
    
    for dir in directories_to_make:
        os.mkdir(dir)
        
    return



def make_params(scenario: str, n: int, wf_input: dict) -> None:
    
    """
    make the parameters for each waveform and save them
    """

    ASSERT(scenario)
    
    # formatting the parameter arrays depending on scenarios: AS (aligned spin), P (nonaligned spin, precessing), NS (nonspinning)
    if scenario == 'NS':
        input_params = {
            'q'     : wf_input['q']
        }
    elif scenario == 'AS':
        input_params = {
            'q'     : wf_input['q'], 
            'chi_1z': wf_input['chi_1z'], 
            'chi_2z': wf_input['chi_2z']
        }
    elif scenario == 'P':
        input_params = {
            'q'     : wf_input['q'], 
            'chi_1z': wf_input['chi_1z'], 'chi_1x': wf_input['chi_1x'], 'chi_1y': wf_input['chi_1y'], 
            'chi_2z': wf_input['chi_2z'], 'chi_2x': wf_input['chi_2x'], 'chi_2y': wf_input['chi_2y']
        }
    
    # for each waveform, save its parameters
    for i in range(len(wf_input['q'])):
        print(f"{i + 1}/{len(wf_input['q'])}", end='\r')
        params = {param: input_params[param][i] for param in list(input_params.keys())}
        with open(f"{crc972home}/datasets_new/{scenario}/{n}/params/params_{i}.json", 'w') as outfile:
            json.dump(params, outfile)

    print(f"params complete")
            
    return



def generate_waveforms_SEOBNRv5_to_save(scenario: str, n: int, mode: str):
    
    """
    """
    
    ASSERT(scenario, mode)
    
    import pyseobnr

    model       = "SEOBNRv5PHM" if scenario == 'P' else "SEOBNRv5HM"
    params_path = f"{crc972home}/datasets_new/{scenario}/{n}/params/"
    M           = len(os.listdir(params_path))
    omega0      = 0.015

    stats = {}

    i = 0
    while i < M - 1: 
        print(f'{i+1}/{M}     ', end='\r')
        params_i_path = f"{crc972home}/datasets_new/{scenario}/{n}/params/params_{i}.json"
        
        with open(f"{params_i_path}", "r") as infile:
            params_i = json.load(infile)
            
            if scenario == 'NS':
                chi_1, chi_2 = 0, 0
            if scenario == 'AS':
                chi_1, chi_2 = params_i['chi_1z'], params_i['chi_2z']
            if scenario == 'P':
                chi_1 = (params_i['chi_1x'], params_i['chi_1y'], params_i['chi_1z'])
                chi_2 = (params_i['chi_2x'], params_i['chi_2y'], params_i['chi_2z'])

            t_i = time.time()
            t, modes, data  = pyseobnr.generate_waveform.generate_modes_opt(
                params_i['q'], chi_1, chi_2, 0.015,
                omega_ref   = 0.015,
                approximant = model,
                settings    = None,
                debug       = True
            )
            dt = time.time() - t_i

            wf = interpolate(modes[mode], t)

            if np.argmin(np.abs(wf)[:5000+1]) != 0:
                print(f"BAD PARAM: {params_i}")
                i += 1
                if i >= M:
                    break
                continue
            if np.argmax(np.abs(wf)[:2500+1]) != 2500:
                print(f"BAD PARAM: {params_i}")
                i += 1
                if i >= M:
                    break
                continue
            if np.abs(np.abs(wf)[10] - np.abs(wf)[0]) > np.abs(np.abs(wf)[110] - np.abs(wf)[100]):
                print(f"BAD PARAM: {params_i}")
                i += 1
                if i >= M:
                    break
                continue

            stats.update({params_i['q']: dt})
            
            np.save(
                f"{crc972home}/datasets_new/{scenario}/{n}/{mode.replace(',', '_')}/waveforms/full/h_{i}.npy", wf)

            i += 1
            
    print(f"waveforms complete")

    with open(f"{crc972home}/stats/{scenario}/{n}/sthts", 'w') as outfile:
        json.dump(stats, outfile)
    
    return



def generate_waveforms_SEOBNRv5_NS(n: int, mode: str) -> dict:
    
    """
    """
    
    ASSERT(None, mode)
    
    import pyseobnr

    params_path = f"{crc972home}/datasets_new/NS/{n}/params/"
    M           = len(os.listdir(params_path))
    omega0      = 0.015

    waveforms = {}
    stats = {}
    
    for i in range(M):
        print(f'{i+1}/{M}     ', end='\r')
        params_i_path = f"{crc972home}/datasets_new/NS/{n}/params/params_{i}.json"
        
        with open(f"{params_i_path}", "r") as infile:
            params_i = json.load(infile)

            t_i = time.time()
            t, modes, data  = pyseobnr.generate_waveform.generate_modes_opt(
                params_i['q'], 0, 0, 0.015,
                omega_ref   = 0.015,
                approximant = "SEOBNRv5HM",
                settings    = None,
                debug       = True
            )
            dt = time.time() - t_i
            stats.update({params_i['q']: dt})
            
            waveforms.update({
                i: {
                    'wf': interpolate(modes[mode], t), 
                    'params': {
                        'q': params_i['q']
                    }
                }
            })
    
    print(f"waveforms complete")

    with open(f"{crc972home}/stats/NS/{n}/sthts", 'w') as outfile:
        json.dump(stats, outfile)
    
    return waveforms

def generate_waveforms_SEOBNRv5_AS(n: int, mode: str) -> dict:
    
    """
    """
    
    ASSERT(None, mode)
    
    import pyseobnr

    params_path = f"{crc972home}/datasets_new/AS/{n}/params/"
    M           = len(os.listdir(params_path))
    omega0      = 0.015

    waveforms = {}
    stats = {}

    i = 0
    while i < M - 1:
        print(f'{i+1}/{M}     ', end='\r')
        params_i_path = f"{crc972home}/datasets_new/AS/{n}/params/params_{i}.json"
        
        with open(f"{params_i_path}", "r") as infile:
            params_i = json.load(infile)

            t_i = time.time()
            t, modes, data  = pyseobnr.generate_waveform.generate_modes_opt(
                params_i['q'], params_i['chi_1z'], params_i['chi_2z'], 0.015,
                omega_ref   = 0.015,
                approximant = "SEOBNRv5HM",
                settings    = None,
                debug       = True
            )
            dt = time.time() - t_i

            wf = interpolate(modes[mode], t)

            if np.argmin(np.abs(wf)[:5000+1]) != 0:
                i += 1
                if i >= M:
                    break
                continue
            if np.argmax(np.abs(wf)[:2500+1]) != 2500:
                i += 1
                if i >= M:
                    break
                continue
            if np.abs(np.abs(wf)[10] - np.abs(wf)[0]) > np.abs(np.abs(wf)[110] - np.abs(wf)[100]):
                i += 1
                if i >= M:
                    break
                continue
                
            stats.update({params_i['q']: dt})
            
            waveforms.update({
                i: {
                    'wf': wf, 
                    'params': {
                        'q'     : params_i['q'],
                        'chi_1z': params_i['chi_1z'],
                        'chi_2z': params_i['chi_2z']
                    }
                }
            })

            i += 1
            
    print(f"waveforms complete")

    with open(f"{crc972home}/stats/AS/{n}/sthts", 'w') as outfile:
        json.dump(stats, outfile)
    
    return waveforms

##############################################################################

# Define function to load saved waveforms

# def load_waveforms(scenario: str, n: int, mode: str, component: str = 'full') -> dict:
    
#     """
#     """
    
#     ASSERT(scenario, mode, component)
    
#     waveforms = {}
#     types      = {'waveforms', 'params'}
#     datapath = f"{crc972home}/datasets_new/{scenario}/{n}"

#     if scenario == 'NS':
#         wf_input = {
#             'q'     : ()
#         }
#     elif scenario == 'AS':
#         wf_input = {
#             'q'     : (), 
#             'chi_1z': (), 
#             'chi_2z': ()
#         }
#     elif scenario == 'P':
#         wf_input = {
#             'q'     : (), 
#             'chi_1z': (), 'chi_1x': (), 'chi_1y': (),
#             'chi_2z': (), 'chi_2x': (), 'chi_2y': (), 
#         }
    
#     dir_params = f"{datapath}/params"
#     dir_mode = f"{datapath}/{mode.replace(',', '_')}"
#     dir_waveforms = f"{dir_mode}/waveforms/{component}"
#     component_tag = 'h' if component == 'full' else component

#     for file in os.listdir(f'{dir_params}/'):
#         if not file.endswith('.ipynb_checkpoints'):
#             with open(f'{dir_params}/{file}') as infile:
                
#                 i = int(file.replace('params_', '').replace('.json', ''))
#                 params_i = json.load(infile)
#                 waveforms.update({i: {
#                     'wf'    : (),
#                     'params': {}
#                 }})
#                 waveforms[i]['params'] = params_i

#     for file in os.listdir(f'{dir_waveforms}/'):
#         if not file.endswith('.ipynb'):
            
#             i = int(file.replace(f'{component_tag}_', '').replace('.npy', ''))
#             wf_i = np.load(f"{dir_waveforms}/{file}")
#             waveforms[i]['wf'] = wf_i

#     final_waveforms = sort_basis(waveforms)
    
#     return final_waveforms



def load_waveforms(scenario: str, n: int, mode: str, component: str = 'full') -> dict:
    
    """
    """
    
    ASSERT(scenario, mode, component)
    
    waveforms = {}
    # types      = {'waveforms', 'params'}
    datapath = f"{crc972home}/datasets_new/{scenario}/{n}"
    
    dir_params = f"{datapath}/params"
    dir_mode = f"{datapath}/{mode.replace(',', '_')}"
    dir_waveforms = f"{dir_mode}/waveforms/{component}"
    component_tag = 'h' if component == 'full' else component
    
    # for file in os.listdir(f'{dir_params}/'):
    #     if not file.endswith('.ipynb_checkpoints'):
    #         with open(f'{dir_params}/{file}') as infile:
                
    #             i = int(file.replace('params_', '').replace('.json', ''))
    #             params_i = json.load(infile)
    #             waveforms.update({i: {
    #                 'wf'    : (),
    #                 'params': {}
    #             }})
    #             waveforms[i]['params'] = params_i

    for file in os.listdir(f'{dir_waveforms}/'):
        if not file.endswith('.ipynb'):
            
            i = int(file.replace(f"{component_tag}_", '').replace('.npy', ''))
            wf_i = np.load(f"{dir_waveforms}/{file}")
            params_i = ()
            with open(f"{dir_params}/params_{i}.json") as infile:
                params_i = json.load(infile)
            
            waveforms.update({i: {
                'wf'    : wf_i,
                'params': params_i
            }})

    final_waveforms = sort_basis(waveforms)
    
    return final_waveforms

##############################################################################

# Define function to add A, phi to waveform dictionaries

def add_A_phi(waveforms: dict) -> dict:
    
    """
    """
    
    waveforms_out = {}
    
    for i in list(waveforms.keys()):
        waveforms_out.update({i : {}})
        waveforms_out[i].update({
            'wf'    : {},
            'params': {}
        }) 
        waveforms_out[i]['wf'].update({
            'full' : waveforms[i]['wf'], # h
            'A'    : np.abs(waveforms[i]['wf']), # amplitude
            'phi'  : -np.unwrap(np.angle(waveforms[i]['wf'])) # phase
        }) 
        waveforms_out[i]['params'].update(waveforms[i]['params'])
    
    return waveforms_out

##############################################################################

# Define function to reformat waveforms dictionary to be compatible with RB, EI etc

def reformat(waveforms: dict):
    
    """
    """
    
    keys = list(waveforms.keys())
    full = {waveforms[i]['params']['q']: waveforms[i]['wf']['full'] for i in keys}
    A    = {waveforms[i]['params']['q']: waveforms[i]['wf']['A']    for i in keys}
    phi  = {waveforms[i]['params']['q']: waveforms[i]['wf']['phi']  for i in keys}
                
    return full, A, phi

##############################################################################

# Define function to save essential bits

def save_bits(scenario: str, mode: str, n: int, component: str, param_indexes: list, time_indexes: list, B: np.ndarray) -> None:

    """
    """

    ASSERT(scenario, mode, component)
    
    datapath     = f"{crc972home}/datasets_new/{scenario}/{n}/{mode.replace(',', '_')}/bits/{component}"
    treasurepath = f"{crc972home}/treasure/{scenario}/{mode.replace(',', '_')}/bits/{component}"
    mk_nested_dirs(datapath)
    mk_nested_dirs(treasurepath)
    
    np.save(f"{datapath}/param_indexes.npy", param_indexes)
    np.save(f"{datapath}/time_indexes.npy",  time_indexes)
    np.save(f"{datapath}/B.npy",             B)
    # np.save(f'{datapath}/B_log.npy',         B_log)

    if n == 3:
        np.save(f"{treasurepath}/param_indexes.npy", param_indexes)
        np.save(f"{treasurepath}/time_indexes.npy",  time_indexes)
        np.save(f"{treasurepath}/B.npy",             B)
        # np.save(f'{treasurepath}/B_log.npy',         B_log)
    
    return

def save_stats(scenario: str, n: int, mode: str, component: str, stats: dict, name: str) -> None:

    """
    """

    ASSERT(scenario, mode)
    
    statspath = f"{crc972home}/stats/{scenario}/{n}/{mode.replace(',', '_')}/{component}"
    mk_nested_dirs(statspath)
    
    with open(f"{statspath}/stats_{name}_{component}", 'w') as outfile:
        json.dump(stats, outfile)
    
    return

# Define function to load essential bits

def load_bits(scenario: str, mode: str, n: int, component: str):
    # mode      : ie '2,2' etc
    # component : which waveform ie 'A' = amplitude, 'phi' = phase
    # n         : which dataset
    # scenario  : 'NS', 'AS', 'P'
    
    """
    """

    ASSERT(scenario, mode, component)

    # directory to load from
    datapath      = f"{crc972home}/datasets_new/{scenario}/{n}/{mode.replace(',', '_')}/bits/{component}"

    # saving bits
    param_indexes = np.load(f"{datapath}/param_indexes.npy", allow_pickle=False)
    time_indexes  = np.load(f"{datapath}/time_indexes.npy",  allow_pickle=False)
    B             = np.load(f"{datapath}/B.npy",             allow_pickle=False)
    # B_log         = np.load(f"{datapath}/B_log.npy',         allow_pickle=False)
    
    return param_indexes, time_indexes, B

def load_stats(scenario: str, n: int, mode: str, component: str, name: str) -> dict:

    """
    """

    ASSERT(scenario, mode)
    
    statspath = f"{crc972home}/stats/{scenario}/{n}/{mode.replace(',', '_')}/{component}"

    with open(f"{statspath}/stats_{name}_{component}", 'r') as infile:
        stats = json.load(infile)
    
    return stats

##############################################################################

# Define function to make treasure directories (stuff i don't want to delete; will be deleting a lot of waveforms)

def mk_nested_dirs(directory_path: str) -> None:
    
    """
    create nested dictionaries that don't exist
    """
    
    # split the directory path into individual directory names
    directory_path = directory_path.replace(f"{crc972home}", '')
    directories = directory_path.split(os.path.sep)[1:]
    path_so_far = f"{crc972home}"
    
    # iterate over each layer
    for directory in directories:
        path_so_far = os.path.join(path_so_far, directory)
        
        if not os.path.exists(path_so_far):
            print(f"creating {path_so_far}")
            os.makedirs(path_so_far)
    
    return

##############################################################################

# Make RB, do EI

def p1peline(waveforms: dict, scenario: str, n: int, mode: str, component: str, tolerance: float = 1e-8, plotting: bool = False, saving: bool = True) -> None:
    
    """
    this function takes a set of waveforms and a series of other associated
    parameters and builds a reduced basis for them, builds the B matrix
    (yielding the empirical time nodes), and has options for plotting and
    saving key bits of data
    """

    ASSERT(scenario, mode)

    print(f"P1PELINE: {scenario} | {n} | {mode} | {tolerance} | {component}")
    
    # INITIALISING
    _ = ['full', 'A', 'phi']
    # wf = reformat(waveforms)[_.index(component)]
    wf = waveforms
    
    stats = {} 

    # REDUCED BASIS
    print(f"RB: constructing reduced basis\n...")
    t_i = time.time()
    rb, sigma_m, param_indexes = RB(wf, TIME, tolerance)

    stats.update({'rb_time': time.time() - t_i})
    print(f"...\nRB: reduced basis complete    ")
    print(f"RB: time elapsed: {stats['rb_time']:.3f}")

    if plotting:
        print(f'RB: plotting reduced basis')
        
        colours_random_in = sort_hex(get_colours(len(rb))) # for RB: elements -> random colours -> THIS IS OK -> SUPPOSED TO APPEAR MESSY ALMOST
        colours_random = {}
        for i, q in enumerate(sorted(rb.keys())):
            colours_random.update({q: colours_random_in[i]})
        colours_greedy = np.load(f"{crc972home}/treasure/colours_greedy.npy")
        complot = r'\phi' if component == 'phi' else fr"{component}"
        Lambda = r'\Lambda'
        
        # plot all rb elements
        fig, ax = plt.subplots(1, 1, figsize = (10, 6))
        for i, q in enumerate(rb):
            ax.plot(TIME, rb[q], color = colours_random[q], zorder = len(rb) - i)
        ax.set_xlabel(fr"$t (t/M)$")
        ax.set_ylabel(fr"${complot}_{{{mode.replace(',', '')}}}^{{\text{{{scenario}}}}}$: $e_i(t)$")
        fig.tight_layout()
        plt.savefig(f"{crc972home}/figures/{scenario}/{n}/{mode.replace(',', '_')}/{component}/rb_elements")
        plt.show()

        # plot greedy errors for RB
        fig, ax = plt.subplots(1, 1, figsize = (6, 6))
        n_dots = 19
        m = len(rb)
        dm = m // n_dots if m > n_dots else 1
        ax.axhline(tolerance, linestyle = '--', color = colours_greedy[0], label = r"$\epsilon$", zorder = 0)
        ax.plot([n for n in range(len(sigma_m))], sigma_m, color = colours_greedy[2], zorder = 1)
        ax.scatter([n for n in range(len(sigma_m))][::dm], sigma_m[::dm], color = colours_greedy[2], label = r"$\sigma_m^{\text{RB}}$", zorder  = 1)
        plt.yscale('log')
        plt.legend()
        ax.set_xlabel(r"$m$")
        ax.set_ylabel(r"$\sigma_m^{\text{RB}}$")
        fig.tight_layout()
        plt.savefig(f"{crc972home}/figures/{scenario}/{n}/{mode.replace(',', '_')}/{component}/greedy_errors_RB")
        plt.show()

        # plot first 6 corresponding waveforms 
        fig, ax = plt.subplots(1, 1, figsize = (10, 6))
        for i, q in enumerate(sorted(list(rb.keys())[:6])):
            ax.plot(TIME, copy.deepcopy(wf)[q], color = colours_random[q], 
                    label = fr"${complot}_{{{mode.replace(',', '')}}}^{{\text{{{scenario}}}}}(t; \vec{{\Lambda}}_{list(rb.keys()).index(q)})$") # fr"$q = {q:.3f}$"
        plt.legend()
        ax.set_xlabel(fr"$t (t/M)$")
        ax.set_ylabel(fr"${complot}_{{{mode.replace(',', '')}}}^{{\text{{{scenario}}}}}(t)$")
        fig.tight_layout()
        plt.savefig(f"{crc972home}/figures/{scenario}/{n}/{mode.replace(',', '_')}/{component}/6_key_waveforms")
        plt.show()
        
    # EMPIRICAL INTERPOLATION
    print(f"\nEI: constructing B matrix")
    wfEI, rb = matrix_basis(wf), matrix_basis(rb)
    t_i = time.time()
    time_indexes, B = EI(rb, TIME)
    stats.update({'ei_time': time.time() - t_i})
    print(f"EI: B matrix complete")
    print(f"EI: time elapsed: {stats['ei_time']:.3f}")

    if plotting:
        B_log = [make_B_matrix(np.linalg.inv(make_V_matrix(rb, time_indexes)[:i+1, :i+1]), rb[:i+1, :], time_indexes[:i+1]) for i in range(len(time_indexes))]
        sigma_Im = greedy_EI_errors(wfEI, B_log, time_indexes, TIME)

        # plot greedy errors for RB & EI
        fig, ax = plt.subplots(1, 1, figsize = (6, 6))
        ax.axhline(tolerance, linestyle = '--', color = colours_greedy[0], label = r"$\epsilon$", zorder = 0)
        ax.plot([n for n in range(len(sigma_m))], sigma_m, color = colours_greedy[2], zorder = 1)
        ax.scatter([n for n in range(len(sigma_m))][::dm], sigma_m[::dm], color = colours_greedy[2], label = r"$\sigma_m^{\text{RB}}$", zorder  = 1)
        ax.plot([n for n in range(len(sigma_Im))][1:], sigma_Im[1:], color = colours_greedy[1], zorder = 2)
        ax.scatter([n for n in range(len(sigma_Im))][1::dm], sigma_Im[1::dm], color = colours_greedy[1], label = r"$\sigma_m^{\text{EI}}$", zorder = 2)
        plt.yscale('log')
        plt.legend()
        ax.set_xlabel(r"$m$")
        ax.set_ylabel(r"$\sigma_m^x$")
        # ax.set_title("greedy errors")
        fig.tight_layout()
        plt.savefig(f"{crc972home}/figures/{scenario}/{n}/{mode.replace(',', '_')}/{component}/greedy_errors_EI")
        plt.show()
    
    # CLEANUP    
    if saving:
        print(f'saving {mode} {component}')
        save_bits(scenario, mode, n, component, param_indexes, time_indexes, B)
        save_stats(scenario, n, mode, component, stats, 'p1peline')
    
    return


##############################################################################

# Define function to make dataset from RB, EI, for ANN construction

def load_dataset(scenario: str, n: int, mode: str, component: str, time_indexes: list, waveforms: Union[dict, None] = None):
    
    """
    """
    
    ASSERT(scenario, mode, component)

    if not waveforms:
        waveforms = add_A_phi(load_waveforms(scenario, n, mode))
    
    # if scenario == 'NS':
    #     x_data = np.array([list(waveforms[i]['params'].values()) for i in waveforms])
    # elif scenario != 'NS':
    #     x_data = np.array([list(waveforms[i]['params'].values()) for i in waveforms])
    x_data = np.array([list(waveforms[i]['params'].values()) for i in waveforms])
    y_data = np.array([waveforms[i]['wf'][component] for i in waveforms])[:, time_indexes]
    
    # return np.array(x_data).real, np.array(y_data).real # added ".real" as a temp debugging solution
    # return x_data.real, y_data.real
    return x_data, y_data

# Define function to split dataset into (x, y) (train, test)

def train_test_split(x_data: np.ndarray, y_data: np.ndarray, scenario: str, fraction_train: float = 0.8, shuffle: bool = True):
    
    """
    """

    ASSERT(scenario)
    
    # Condition x_data for spin vs spinless
    # if scenario == 'NS':
    #     empty  = np.zeros((x_data.shape[0], 1))
    #     x_data = empty + x_data[:, :1]
    # # elif scenario == 
    # print(f"TESTSPLIT{x_data.shape}, {y_data.shape}")
    # Use copy to preserve original data
    dataset    = np.concatenate((np.copy(x_data), np.copy(y_data)), axis = -1)
    n_samples  = np.shape(dataset)[0]
    fraction   = 0.8
    n_train    = int(fraction_train * n_samples)
        
    if shuffle:
        np.random.shuffle(dataset)
    
    training   = dataset[:n_train]
    testing    = dataset[n_train:]
    
    train_x = training[:, :x_data.shape[1]]
    test_x  = testing[:, :x_data.shape[1]]
    train_y = training[:, x_data.shape[1]:]
    test_y  = testing[:, x_data.shape[1]:]
    
    return train_x, test_x, train_y, test_y

##############################################################################

# Make ANN'

def p2peline_A(waveforms: dict, scenario: str, n: int, mode: str, saving: bool = True, plotting: bool = False) -> None: ### !!! ###
    # use nodes from p1peline and fiducial waveform values at the nodes to construct an ANN to guess these values
    # construct model
    # save model
    
    """
    TESTING TESTING TESTING
    """

    ASSERT(scenario, mode)
    
    import tensorflow as tf
    import keras
    from sklearn import preprocessing
    from hyperas.distributions import choice, uniform
            
    # get bits for training set
    _, time_indexes, B = load_bits(scenario, mode, n, 'A')
    T = TIME[time_indexes]
    
    # make training dataset
    print(f"loading dataset...                  ", end='\r')
    # x_data, y_data = np.array(list(waveforms.keys()))[:, np.newaxis], matrix_basis(waveforms)[:, time_indexes]
    # if scenario == 'NS':
    #     x_data = np.array([list(waveforms[i]['params'].values()) for i in waveforms.keys()])
    # elif scenario != 'NS':
    #     x_data = np.array([list(waveforms[i]['params'].values()) for i in waveforms.keys()])
    x_data = np.array([list(waveforms[i]['params'].values()) for i in waveforms.keys()])
    y_data = np.array([waveforms[i]['wf']['A'] for i in waveforms])[:, time_indexes]

    print(f"\n{x_data.shape}, {y_data.shape}\n") ### !!! ###
    
    yscaler = preprocessing.MinMaxScaler() # init data preprocessor
    yscaler = yscaler.fit(y_data) # calibrate data preprocessor
    
    print(f"saving scaler...                    ", end='\r')
    with open(f"{crc972home}/treasure/{scenario}/{mode.replace(',', '_')}/scaler{mode.replace(',', '_')}_A.pickle", 'wb') as outfile: 
        pickle.dump(yscaler, outfile) # save data preprocessor
    with open(f"{crc972home}/models/scaler{scenario}{n}_{mode.replace(',', '_')}_A.pickle", 'wb') as outfile: 
        pickle.dump(yscaler, outfile) # save data preprocessor
    y_data = yscaler.transform(y_data) # scale data
    print(f"splitting dataset...                  ", end='\r')
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, scenario) # split data into train & test data
    
    # model params
    HP = {
        'Ne'            : 4000,
        'lr0'           : 3e-2,
        'lr_min'        : 1e-10,
        'lrop_patience' : 128,
        'lrop_factor'   : 0.7,
        'es_patience'   : 256,
        'batch_size'    : 128
    }
    
    # Adam optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=HP['lr0'], epsilon=1e-6, name='Adam')

    # stop model learning if no improvement for extended period
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=HP['es_patience'], verbose=1, restore_best_weights=True)

    # reduce learning rate on plateau
    lrop_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=HP['lrop_factor'], patience=HP['lrop_patience'], min_lr=HP['lr_min'], verbose=1)
        
    # construct ANN
    # want to minimise mean squared error (mse)    
    print(f"building neural network...                ", end='\r')
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(x_data.shape[1] if scenario != 'NS' else 1,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(y_data.shape[1], activation='linear', bias_initializer=tf.keras.initializers.glorot_uniform)
    ])    

    # compile model
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    # train ANN
    print(f"training neural network...                ", end='\n')
    t_i = time.time()
    history = model.fit(
        train_x, train_y, validation_data = (test_x, test_y), 
        epochs = HP['Ne'], batch_size = HP['batch_size'], shuffle = True, callbacks = [
            early_stopping,
            lrop_callback,
        ])
    stats = {'ann_time': time.time() - t_i}
    save_stats(scenario, n, mode, 'A', stats, 'p2peline')
    
    # model summary
    model.summary()
    
    # save model
    if saving:
        print(f"saving neural network model...              ", end='\r')
        model.save(f"{crc972home}/models/model{scenario}{n}_{mode.replace(',', '_')}_A.keras")
        if n == 3:
            model.save(f"{crc972home}/treasure/{scenario}/{mode.replace(',', '_')}/model{scenario}{n}_{mode.replace(',', '_')}_A.keras")

    # plotting
    if plotting:
        # colours_ann = get_colours(3)
        colours_ann = np.load(f"{crc972home}/treasure/colours_greedy.npy")
        # np.save(f"{crc972home}/treasure/{scenario}/colours_ann", np.array(colours_ann))
        fig, axs = plt.subplots(1, 1, figsize = (10, 6))
        plt.loglog(history.history['loss']    , label = r"$\mathcal{C}_{\text{ost}}$ (Training)", color = colours_ann[0])
        plt.loglog(history.history['val_loss'], label = r"$\mathcal{C}_{\text{ost}}$ (Validation)", color = colours_ann[2])
        plt.loglog(history.history['lr']      , label = r"Learning Rate (LR)", color = colours_ann[1])
        plt.xlabel(r"Epoch")
        plt.ylabel(fr"$A_{{{mode.replace(',', '')}}}^{{\text{{{scenario}}}}}$: Costs & LR")
        plt.legend()
        plt.grid(True)
        fig.tight_layout()
        plt.savefig(f"{crc972home}/figures/{scenario}/{n}/{mode.replace(',', '_')}/A/ann_A_training")
        plt.show()
    
    return



def p2peline_phi(waveforms: dict, scenario: str, n: int, mode: str, saving: bool = True, plotting: bool = False) -> None: ### !!! ###
    # use nodes from p1peline and fiducial waveform values at the nodes to construct an ANN to guess these values
    # construct model
    # save model
    
    """
    TESTING TESTING TESTING
    """

    ASSERT(scenario, mode)
    
    import tensorflow as tf
    import keras
    from sklearn import preprocessing
    from hyperas.distributions import choice, uniform
        
    # get bits for training set
    _, time_indexes, B = load_bits(scenario, mode, n, 'phi')
    T = TIME[time_indexes]
    
    # make training dataset
    print(f"loading dataset...                  ", end='\r')
    # x_data, y_data = np.array(list(waveforms.keys()))[:, np.newaxis], matrix_basis(waveforms)[:, time_indexes]
    if scenario == 'NS':
        x_data = np.array([list(waveforms[i]['params'].values()) for i in waveforms.keys()])
    elif scenario != 'NS':
        x_data = np.array([list(waveforms[i]['params'].values()) for i in waveforms.keys()])
    y_data = np.array([waveforms[i]['wf']['phi'] for i in waveforms])[:, time_indexes]

    print(f"\n{x_data.shape}, {y_data.shape}\n") ### !!! ###
    
    yscaler = preprocessing.MinMaxScaler() # init data preprocessor
    yscaler = yscaler.fit(y_data) # calibrate data preprocessor
    print(f"saving scaler...                    ", end='\r')
    with open(f"{crc972home}/treasure/{scenario}/{mode.replace(',', '_')}/scaler{mode.replace(',', '_')}_phi.pickle", 'wb') as outfile: 
        pickle.dump(yscaler, outfile) # save data preprocessor
    with open(f"{crc972home}/models/scaler{scenario}{n}_{mode.replace(',', '_')}_phi.pickle", 'wb') as outfile: 
        pickle.dump(yscaler, outfile) # save data preprocessor
    y_data = yscaler.transform(y_data) # scale data
    print(f"splitting dataset...                  ", end='\r')
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, scenario) # split data into train & test data
    
    # model params
    HP = {
        'Ne'            : 10000, # 300
        'lr0'           : 5e-5, # 1.3e-2
        'lr_min'        : 1e-10, # 1e-10
        'lrop_patience' : 32, # 8
        'lrop_factor'   : 0.9, # 0.6
        'es_patience'   : 256, # 64
        'batch_size'    : 512 # 16
    }
        
    # Adam optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=HP['lr0'], epsilon=1e-6, name='Adam')

    # stop model learning if no improvement for extended period
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=HP['es_patience'], verbose=1, restore_best_weights=True)

    # reduce learning rate on plateau
    lrop_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=HP['lrop_factor'], patience=HP['lrop_patience'], min_lr=HP['lr_min'], verbose=1)

    # learning rate scheduler
    n_burnin, n_1, n_2, n_3, n_4 = 64, 128, 256, 512, 1024
    def schedule(epoch):
        if epoch >= 0 and epoch < n_burnin:
            lr = 5e-2
        elif epoch >= n_burnin and epoch < n_1:
            lr = 1e-2
        elif epoch >= n_1 and epoch < n_2:
            lr = 8e-3
        elif epoch >= n_2 and epoch < n_3:
            lr = 5e-3
        elif epoch >= n_3 and epoch < n_4:
            lr = 4e-3
        elif epoch >= n_4:
            lr = 3e-3
        return lr
        
    def schedule2(epoch, lr):
        if epoch >= 0 and epoch < n_burnin:
            return 5e-2
        elif epoch >= n_burnin:
            return lr
        
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule2, verbose = 0)
        
    # construct ANN
    # want to minimise mean squared error (mse)     
    print(f"building neural network...                ", end='\r')
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='sigmoid', input_shape=(x_data.shape[1] if scenario != 'NS' else 1,)),
        tf.keras.layers.Dense(y_data.shape[1], activation='linear', bias_initializer=tf.keras.initializers.glorot_uniform)
    ])
    
    # compile model
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])

    # train ANN
    print(f"training neural network...                ", end='\n')
    t_i = time.time()
    history = model.fit(
        train_x, train_y, validation_data = (test_x, test_y), 
        epochs = HP['Ne'], batch_size = HP['batch_size'], shuffle = True, callbacks = [
            early_stopping,
            lr_scheduler,
            lrop_callback,
            
        ])
    stats = {'ann_time': time.time() - t_i}
    save_stats(scenario, n, mode, 'phi', stats, 'p2peline')
    
    # model summary
    model.summary()
    
    # save model
    if saving:
        print(f"saving neural network model...              ", end='\r')
        model.save(f"{crc972home}/models/model{scenario}{n}_{mode.replace(',', '_')}_phi.keras")

    # plotting
    if plotting:
        # colours_ann = get_colours(3)
        colours_ann = np.load(f"{crc972home}/treasure/colours_greedy.npy")
        # np.save(f"{crc972home}/treasure/{scenario}/colours_ann", np.array(colours_ann))
        fig, axs = plt.subplots(1, 1, figsize = (10, 6))
        plt.loglog(history.history['loss']    , label = r"$\mathcal{C}_{\text{ost}}$ (Training)", color = colours_ann[0])
        plt.loglog(history.history['val_loss'], label = r"$\mathcal{C}_{\text{ost}}$ (Validation)", color = colours_ann[2])
        plt.loglog(history.history['lr']      , label = r"Learning Rate (LR)", color = colours_ann[1])
        plt.xlabel(r"Epoch")
        plt.ylabel(fr"$\phi_{{{mode.replace(',', '')}}}^{{\text{{{scenario}}}}}$: Costs & LR")
        plt.legend()
        plt.grid(True)
        fig.tight_layout()
        plt.savefig(f"{crc972home}/figures/{scenario}/{n}/{mode.replace(',', '_')}/phi/ann_phi_training")
        plt.show()
    
    return

##############################################################################

def load_params(scenario: str, n: int) -> dict:
    
    params_dir = f"{crc972home}/datasets_new/{scenario}/{n}/params"
    params = {}
    for i, file in enumerate(os.listdir(params_dir)):
        if file.endswith('.json'):
            with open(f"{params_dir}/{file}", "r") as infile:
                
                j = file.replace('.json', '').replace('params_', '')
                params_j = json.load(infile)
                
                for param in params_j:
                    if i == 0:
                        params[param] = []
                    params[param].append(params_j[param])
    
    params = {param: np.array(params[param]) for param in list(params.keys())}
        
    return params

##############################################################################

# Test pipeline: UNDER CONSTRUCTION

def p3peline(scenario: str, n: int, mode: str, plotting: bool = True, mismatches: bool = True) -> dict:
    
    """
    """

    ASSERT(scenario, mode)
    
    import tensorflow as tf
    import keras
    from sklearn import preprocessing

    n_test = 1
    
    _, time_indexes_A  , B_A   = load_bits(scenario, mode, n, 'A')
    _, time_indexes_phi, B_phi = load_bits(scenario, mode, n, 'phi')
    T_A, T_phi                 = TIME[time_indexes_A], TIME[time_indexes_phi]
    
    with open(f"{crc972home}/models/scaler{scenario}{n}_{mode.replace(',', '_')}_A.pickle", 'rb') as infile:
        yscaler_A = pickle.load(infile)
    with open(f"{crc972home}/models/scaler{scenario}{n}_{mode.replace(',', '_')}_phi.pickle", 'rb') as infile:
        yscaler_phi = pickle.load(infile)

    model_A   = tf.keras.saving.load_model(f"{crc972home}/models/model{scenario}{n}_{mode.replace(',', '_')}_A.keras")
    model_phi = tf.keras.saving.load_model(f"{crc972home}/models/model{scenario}{n}_{mode.replace(',', '_')}_phi.keras")

    waveforms_test              = add_A_phi(load_waveforms(scenario, n_test, mode))
    full_test, A_test, phi_test = reformat(waveforms_test)
    
    x_data     = np.array([list(waveforms_test[i]['params'].values()) for i in waveforms_test.keys()])
    y_data_A   = np.array([waveforms_test[i]['wf']['A']   for i in waveforms_test])[:, time_indexes_A]
    y_data_phi = np.array([waveforms_test[i]['wf']['phi'] for i in waveforms_test])[:, time_indexes_phi]

    print(f"x: {x_data.shape}, A  : {y_data_A.shape}")
    print(f"x: {x_data.shape}, phi: {y_data_phi.shape}")
        
    predict_A   = yscaler_A.inverse_transform(model_A.predict(x_data))
    predict_phi = yscaler_phi.inverse_transform(model_phi.predict(x_data))

    print(predict_A.shape, predict_phi.shape)
    print(time_indexes_A.shape, time_indexes_phi.shape)
    
    reconstructed_A = np.zeros((len(A_test), len(TIME)))
    reconstructed_phi = np.zeros_like(reconstructed_A)
    reconstructed_full = np.zeros_like(reconstructed_phi, dtype = np.complex128)
    
    for i, q in zip([i for i in range(len(A_test))], [float(q) for q in A_test]):
        
        h_A                    = predict_A[i].copy()
        Ih_A                   = EI_h(h_A, B_A, time_indexes_A)
        reconstructed_A[i]    += Ih_A
    
        h_phi                  = predict_phi[i].copy()
        Ih_phi                 = EI_h(h_phi, B_phi, time_indexes_phi)
        reconstructed_phi[i]  += Ih_phi
    
        reconstructed_full[i] += Ih_A * np.exp(-1j * Ih_phi)
    
    print(reconstructed_full.shape)

    mse_A, mse_phi = (np.square(predict_A - y_data_A)).mean(axis=1), (np.square(predict_phi - y_data_phi)).mean(axis=1)
    ind_A, ind_phi = np.where(np.max(mse_A) == mse_A)[0][0], np.where(np.max(mse_phi) == mse_phi)[0][0]
    mm_A, mm_phi   = mismatch(reconstructed_A[ind_A], A_test[list(A_test.keys())[ind_A]]), mismatch(reconstructed_phi[ind_phi], phi_test[list(phi_test.keys())[ind_phi]])
    # print(f'A   ) ind: {ind_A:.3f} | mse: {mse_A[ind_A]:.3f} | mismatch: 10^({np.log10(mm_A):.3f})')
    # print(f'phi ) ind: {ind_phi:.3f} | mse: {mse_phi[ind_phi]:.3f} | mismatch: 10^({np.log10(mm_phi):.3f})')

    diff = [reconstructed_full[i] - full_test[list(full_test.keys())[i]] for i in range(reconstructed_full.shape[0])]
    if mismatches:
        full_mismatches = np.array([mismatch(reconstructed_full[i], full_test[list(full_test.keys())[i]]) for i in range(reconstructed_full.shape[0])])
        np.save(f"{crc972home}/stats/{scenario}/{n}/{mode.replace(',', '_')}/full_mismatches", full_mismatches)
    elif not mismatches:
        full_mismatches = np.array([dot(diff[i], diff[i], TIME) for i in range(reconstructed_full.shape[0])])
    mm_full, ind_full = np.max(full_mismatches), np.argmax(full_mismatches)
    # print(f'full) ind: {ind_full:.3f} | mismatch: 10^({np.log10(np.abs(mm_full)):.3f})')

    if plotting:
        
        # coolours
        # colours_rec = get_colours(29)
        colours_rec = np.load(f"{crc972home}/treasure/NS/colours_rec.npy")
        # np.save(f"{crc972home}/treasure/NS/colours_rec", colours_rec)
        
        k = list(full_test.keys())[ind_full]
        xlabel = fr"$t (t/M)$"
        fid_mod = r"SEOBNRv5HM" if scenario != 'P' else r"SEOBNRv5PHM"

        # plot mismatches
        c = get_colours()
        fig, ax = plt.subplots(1, 1, figsize = (5, 6))
        bins = 7
        logbins = np.logspace(np.log10(np.min(full_mismatches)),np.log10(np.max(full_mismatches)),bins+1)
        ax.hist(full_mismatches, bins = logbins, color = c)
        ax.set_xscale('log')
        ax.set_xlabel(r"$1 - \mathcal{O}(h_{\text{surrogate}}, h_{\text{fiducial}})$")
        ax.set_ylabel(r"Count")
        # fig.tight_layout()
        plt.savefig(f"{crc972home}/figures/{scenario}/{n}/{mode.replace(',', '_')}/mismatch_hist")
        plt.show()

        # plot A corresponding to worst h reconstruction
        print(f'A   ) ind: {ind_full:.3f}| mismatch: 10^({np.log10(mismatch(A_test[k], reconstructed_A[ind_full])):.3f})')
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(TIME, A_test[k], ### !!! ###
                color = colours_rec[0], label = fr"{fid_mod}", zorder = 0)
        ax.plot(TIME, reconstructed_A[ind_full], ### !!! ###
                color = COLOURS[mode], label = fr"Surrogate ({scenario})", linestyle = "dashed", zorder = 1)
        ax.scatter(TIME[time_indexes_A], predict_A[ind_full],
                   color = colours_rec[-1], label = r"Empirical Time Nodes", marker = 'x', zorder = 2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(fr"$A_{{{mode.replace(',', '')}}}^{{\text{{{scenario}}}}}(t)$")
        plt.legend()
        fig.tight_layout()
        plt.savefig(f"{crc972home}/figures/{scenario}/{n}/{mode.replace(',', '_')}/A_worst_h")
        plt.show()

        # plot phi corresponding to worst h reconstruction
        print(f'phi ) ind: {ind_full:.3f}| mismatch: 10^({np.log10(mismatch(phi_test[k], reconstructed_phi[ind_full])):.3f})')
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(TIME, phi_test[k], ### !!! ###
                color = colours_rec[0], label = fr"{fid_mod}", zorder=0)
        ax.plot(TIME, reconstructed_phi[ind_full], ### !!! ### 
                color = COLOURS[mode], label = fr"Surrogate ({scenario})", linestyle = "dashed", zorder=1)
        ax.scatter(TIME[time_indexes_phi], predict_phi[ind_full],
                   color = colours_rec[-1], label = r"Empirical Time Nodes", marker = 'x', zorder = 2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(fr"$\phi_{{{mode.replace(',', '')}}}^{{\text{{{scenario}}}}}(t)$")
        plt.legend()
        fig.tight_layout()
        plt.savefig(f"{crc972home}/figures/{scenario}/{n}/{mode.replace(',', '_')}/phi_worst_h")
        plt.show()

        # plot worst h reconstruction
        print(f'full) ind: {ind_full:.3f}| mismatch: 10^({np.log10(mismatch(full_test[k], reconstructed_full[ind_full])):.3f})')
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(TIME, full_test[k].real,
                color = colours_rec[0], label = fr"{fid_mod}", zorder=0)
        ax.plot(TIME, reconstructed_full[ind_full].real, 
                color = COLOURS[mode], label = fr"Surrogate ({scenario})", linestyle = "dashed", zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(fr"$h_{{{mode.replace(',', '')}}}^{{\text{{{scenario}}}}}(t)$")
        plt.legend()
        fig.tight_layout()
        plt.savefig(f"{crc972home}/figures/{scenario}/{n}/{mode.replace(',', '_')}/worst_h")
        plt.show()
        
        return

    print(f'full) ind: {ind_full:.3f}| mismatch: 10^({np.log10(mm_full):.3f})')
    
    return



def choose_wf(mismatches: np.ndarray):
    sorted_mismatches = np.sort(mismatches)
    sorted_mismatches_to_plot = sorted_mismatches[::(sorted_mismatches.shape[0] // 62)]
    if sorted_mismatches_to_plot.shape[0] < 64:
        sorted_mismatches_to_plot = np.append(sorted_mismatches_to_plot, np.max(sorted_mismatches))
    # print(sorted_mismatches_to_plot.shape, np.max(sorted_mismatches_to_plot), np.max(sorted_mismatches_mode))
    indexes = [list(mismatches).index(mm) for mm in sorted_mismatches_to_plot]
    # print(indexes)
    return indexes

def mismatch_mass_range_per_wf(hsur, hfid): # do 20 solar masses to 300 solar masses
    M_space = np.array(list(range(20, 301)))
    mismatch_line = [mismatch(hsur, hfid, M) for M in M_space]
    return mismatch_line

def m4ssline(scenario: str, n: int, mode: str) -> dict:
    
    """
    """

    ASSERT(scenario, mode)
    
    import tensorflow as tf
    import keras
    from sklearn import preprocessing

    mismatches_50M = np.load(f"../stats/{scenario}/{n}/{mode.replace(',', '_')}/full_mismatches.npy")
    wf_indexes_NA  = choose_wf(mismatches_50M)

    n_test = 1
    
    _, time_indexes_A  , B_A   = load_bits(scenario, mode, n, 'A')
    _, time_indexes_phi, B_phi = load_bits(scenario, mode, n, 'phi')
    T_A, T_phi                 = TIME[time_indexes_A], TIME[time_indexes_phi]
    
    with open(f"{crc972home}/models/scaler{scenario}{n}_{mode.replace(',', '_')}_A.pickle", 'rb') as infile:
        yscaler_A = pickle.load(infile)
    with open(f"{crc972home}/models/scaler{scenario}{n}_{mode.replace(',', '_')}_phi.pickle", 'rb') as infile:
        yscaler_phi = pickle.load(infile)

    model_A   = tf.keras.saving.load_model(f"{crc972home}/models/model{scenario}{n}_{mode.replace(',', '_')}_A.keras")
    model_phi = tf.keras.saving.load_model(f"{crc972home}/models/model{scenario}{n}_{mode.replace(',', '_')}_phi.keras")

    waveforms_test_in           = add_A_phi(load_waveforms(scenario, n_test, mode))
    wf_indexes                  = [list(waveforms_test_in.keys())[i] for i in wf_indexes_NA]
    waveforms_test              = {ind: waveforms_test_in[ind] for ind in wf_indexes}
    full_test, A_test, phi_test = reformat(waveforms_test)
    
    x_data     = np.array([list(waveforms_test[i]['params'].values()) for i in waveforms_test.keys()])
    y_data_A   = np.array([waveforms_test[i]['wf']['A']   for i in waveforms_test])[:, time_indexes_A]
    y_data_phi = np.array([waveforms_test[i]['wf']['phi'] for i in waveforms_test])[:, time_indexes_phi]

    print(f"x: {x_data.shape}, A  : {y_data_A.shape}")
    print(f"x: {x_data.shape}, phi: {y_data_phi.shape}")
        
    predict_A   = yscaler_A.inverse_transform(model_A.predict(x_data))
    predict_phi = yscaler_phi.inverse_transform(model_phi.predict(x_data))

    print(predict_A.shape, predict_phi.shape)
    print(time_indexes_A.shape, time_indexes_phi.shape)
    
    reconstructed_A = np.zeros((len(A_test), len(TIME)))
    reconstructed_phi = np.zeros_like(reconstructed_A)
    reconstructed_full = np.zeros_like(reconstructed_phi, dtype = np.complex128)
    
    for i, q in zip([i for i in range(len(A_test))], [float(q) for q in A_test]):
        
        h_A                    = predict_A[i].copy()
        Ih_A                   = EI_h(h_A, B_A, time_indexes_A)
        reconstructed_A[i]    += Ih_A
    
        h_phi                  = predict_phi[i].copy()
        Ih_phi                 = EI_h(h_phi, B_phi, time_indexes_phi)
        reconstructed_phi[i]  += Ih_phi
    
        reconstructed_full[i] += Ih_A * np.exp(-1j * Ih_phi)
    
    print(reconstructed_full.shape)

    # have hsur & hfid to plot
    # now need to get the mismatch lines
    mismatch_lines = {}
    for i, key in enumerate(list(waveforms_test.keys())):
        mismatch_line = mismatch_mass_range_per_wf(reconstructed_full[i], waveforms_test[key]['wf']['full'])
        mismatch_lines.update({key: mismatch_line})

    with open(f"../stats/{scenario}/mismatch_channel_{mode.replace(',', '_')}", "w") as outfile:
        json.dump(mismatch_lines, outfile)
    
    return mismatch_lines



def p3peline_noANN(scenario: str, n: int, mode: str, plotting: bool = True) -> None:
    
    """
    """

    ASSERT(scenario, mode)
    
    import tensorflow as tf
    import keras
    from sklearn import preprocessing

    n_test = 1
        
    _, time_indexes_A  , B_A   = load_bits(scenario, mode, n, 'A')
    _, time_indexes_phi, B_phi = load_bits(scenario, mode, n, 'phi')
    T_A, T_phi                 = TIME[time_indexes_A], TIME[time_indexes_phi]
    
    waveforms_test              = add_A_phi(load_waveforms(scenario, n_test, mode))
    full_test, A_test, phi_test = reformat(waveforms_test)
    
    y_data_A = matrix_basis(A_test)[:, time_indexes_A]
    y_data_phi = matrix_basis(phi_test)[:, time_indexes_phi]

    # y_data_A = np.array([waveforms_test[i]['wf']['A'] for i in waveforms_test])[:, time_indexes_A]
    # y_data_phi = np.array([waveforms_test[i]['wf']['phi'] for i in waveforms_test])[:, time_indexes_phi]
    
    predict_A = copy.deepcopy(y_data_A)
    predict_phi = copy.deepcopy(y_data_phi)
    
    reconstructed_A = np.zeros((len(A_test), len(TIME)))
    reconstructed_phi = np.zeros_like(reconstructed_A)
    reconstructed_full = np.zeros_like(reconstructed_phi, dtype=np.complex128)
    
    for i, q in zip([i for i in range(len(A_test))], [float(q) for q in A_test]):
        
        h_A                    = predict_A[i]
        Ih_A                   = EI_h(h_A, B_A, time_indexes_A)
        reconstructed_A[i]    += Ih_A
    
        h_phi                  = predict_phi[i]
        Ih_phi                 = EI_h(h_phi, B_phi, time_indexes_phi)
        reconstructed_phi[i]  += Ih_phi
    
        reconstructed_full[i] += Ih_A * np.exp(-1j * Ih_phi)

    diff = [reconstructed_full[i] - full_test[list(full_test.keys())[i]] for i in range(reconstructed_full.shape[0])]
    full_mismatches = [dot(diff[i], diff[i], TIME) for i in range(reconstructed_full.shape[0])]
    mm_full, ind_full = np.max(full_mismatches), np.argmax(full_mismatches)
    mm_full = mismatch(reconstructed_full[ind_full], full_test[list(full_test.keys())[ind_full]])
    
    if plotting:

        c = get_colours(3)
        k = list(full_test.keys())[ind_full]
        xlabel = fr"$(t/M)$"

        # print(f'A   ) ind: {ind_full:.3f}| mismatch: 10^({np.log10(mismatch(A_test[k], reconstructed_A[ind_full])):.3f})')
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(TIME, A_test[k], ### !!! ###
                color = c[0], label = "Original", zorder = 0)
        ax.plot(TIME, reconstructed_A[ind_full], ### !!! ###
                color = c[1], label = "Reconstruction", linestyle = "dashed", zorder = 1)
        ax.scatter(TIME[time_indexes_A], predict_A[ind_full],
                   color = c[2], label = "emp nodes", marker = 'x', zorder = 2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(fr"$A_{{{mode.replace(',', '')}}}(t)$")
        plt.legend()
        fig.tight_layout()
        plt.show()
        
        print(f'phi ) ind: {ind_full:.3f}| mismatch: 10^({np.log10(mismatch(phi_test[k], reconstructed_phi[ind_full])):.3f})')

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(TIME, phi_test[k], ### !!! ###
                color = c[0], label = "Original", zorder=0)
        ax.plot(TIME, reconstructed_phi[ind_full], ### !!! ### 
                color = c[1], label = "Reconstruction", linestyle = "dashed", zorder=1)
        ax.scatter(TIME[time_indexes_phi], predict_phi[ind_full],
                   color = c[2], label = "emp nodes", marker = 'x', zorder = 2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(fr"$\phi_{{{mode.replace(',', '')}}}(t)$")
        plt.legend()
        fig.tight_layout()
        plt.show()
        
        print(f'full) ind: {ind_full:.3f}| mismatch: 10^({np.log10(mismatch(full_test[k], reconstructed_full[ind_full])):.3f})')
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(TIME, full_test[k].real, 
                color = c[0], label = "Original", zorder=0)
        ax.plot(TIME, reconstructed_full[ind_full].real, 
                color = c[1], label = "Reconstruction", linestyle = "dashed", zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(fr"$h_{{{mode.replace(',', '')}}}(t)$")
        plt.legend()
        fig.tight_layout()
        plt.show()

        return

    print(f'full) ind: {ind_full:.3f}| mismatch: 10^({np.log10(mm_full):.3f})')
    
    return



#############################################################################

def zip_wf(scenario: str, mode: str, n: int) -> None:
    
    """
    """

    ASSERT(scenario, mode)
    
    waveforms = load_waveforms(MODE, N)[MODE]
    waveforms_tz = {str(i): waveforms[i]['wf'] for i in list(waveforms.keys())}

    np.savez(f"{crc972home}/datasets_new/{scenario}/{n}/{mode.replace(',', '_')}/waveforms.npz", **waveforms_tz)
    
    return

##############################################################################

# Define function to get models and scalers quickly

def get_mods(scenario: str, mode: str):
    
    """
    load ANN models -> to predict waveform values at the empirical time nodes
    """
    
    ASSERT(scenario, mode)
    
    mods_path = f"{crc972home}/treasure/{tag}/{mode.replace(',', '_')}"
    
    return model_A, model_phi, scaler_A, scaler_phi

##############################################################################

# Define function to find x!

def factorial(x):
    
    """
    """
    
    output = 1
    if x > 0:
        for n in range(1, x+1):
            output *= n
    return output

##############################################################################

# Define function to find Wigner-d matrices

def Wigner_d(l, m_, m, beta):
    
    """
    """
    
    prefactor = np.sqrt( factorial(l+m) * factorial(l-m) * factorial(l+m_) * factorial(l-m_) )
    sum = np.zeros_like(beta)
    for k in K:
      sum += ( (-1)**(k+m_-m) / (factorial(k) * factorial(l+m-k) * factorial(l-m_-k) * factorial(m_-m+k)) ) * ( np.sin(-beta/2)**((2*k)+m_-m) ) * ( np.cos(-beta/2)**((2*l)-(2*k)-m_+m) )
    
    return prefactor * sum

##############################################################################

# Define function to perform L-J transformations:

def LJ(H_in, l, m, gamma, beta, alpha):
    # H_in : waveform in, need to sum over the l's
    # gamma : Euler angle gamma(t)
    # beta : Euler angle beta(t)
    # alpha : Euler angle alpha : const
    """
    """
    
    H_out = np.zeros_like(H_in)
    for m_ in range(-l, l+1):
        H_out += np.exp(1j*m_*gamma) * Wigner_d(l, m_, m, beta) * np.exp(1j * m * alpha) * H_in # need to sum over l's in H_in
        
    return H_out

##############################################################################























