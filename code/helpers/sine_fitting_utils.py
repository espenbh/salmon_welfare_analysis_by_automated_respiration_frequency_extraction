# The functions below uses RANSAC and Levenberg-Marquardt to fit a sinusoid to a noisy time series signal.

import numpy as np
import math
import random
from scipy.optimize import curve_fit

# Define the sine function to fit
def sine(x, freq, amp, phase, offset, a):
        return (np.sin(x * freq*(2*np.pi) + phase) * amp + offset) + a*x

# Find the necessary number of iterations to find an outlier free sample of size n with eta0 percent probability given an outlier ratio of eps.
def stop_crit(eta0, eps, n):
        return math.log(1-eta0, 10)/math.log(1-math.pow(eps,n), 10)

# fit a straight line to the time series in order to estimate offset and standard deviation of the sample.
def fit_line_to_data(x, y):
        xm = np.sum(x)/x.shape[0]
        ym = np.sum(y)/y.shape[0]
        a = np.sum(np.multiply((x-xm),(y-ym)))/np.sum(np.power(x-xm, 2))
        b = ym-a*xm
        sd = np.sqrt(np.sum(np.power(y-(a*x+b), 2))/y.shape[0])
        return a, b, sd

# thresh_gain: The RANSAC outlier threshold is calculated as thresh_gain multiplied by the standard deviation betwenn the sample set and a straight line fitted to the sample.
# thresh_gain_ret: The threshold multipliers for the inliers that are returned from the function. Increase this to increase the inlier count.

def fit_sine(x, y, freq0, eta0 = 0.9999, thresh_gain = 1, thresh_gain_ret = 0.7, debug = False):
    # Initialize sinusoid
    al, _, sdl = fit_line_to_data(x, y)
    freq0 = freq0           # Frequency of sine in Hz
    amp0 = np.sqrt(2)*sdl   # Amplitude of sine in pxl
    offset0 = np.mean(y) # Offset of sine in pxl
    a0 = al              # linear weight of signal in pxl/sec


    # Initialize sine parameters and storage
    thresh = sdl*thresh_gain
    b_score = 0
    ns = 5
    b_inl = []
    num_it = 0
    k = 100
    b_popt = []
    b_pcov = []

    while num_it < k:
        # Do small, random permutations of initial values at each iteration to avoid local minima
        phase0 = random.uniform(-np.pi, np.pi)
        p0=[freq0*random.uniform(0.95, 1.05), amp0*random.uniform(0.7, 1.3), phase0, offset0*random.uniform(0.95, 1.05), a0*random.uniform(0.95, 1.05)]

        # Draw random minimum sample
        idx = random.sample(range(0, x.shape[0]), ns)
        x_sam = x[idx]
        y_sam = y[idx]

        try:
            # Fit sine, calculate covariance
            popt, pcov = curve_fit(sine, x_sam, y_sam, p0=p0, maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            num_it = num_it + 1
        except:
            continue
        
        # Calculate score
        y_sine = sine(x, *list(popt))
        scores = np.abs(y_sine-y)
        inl = scores < thresh

        # Save parameres if best score
        if sum(inl) > b_score:
                b_inl = inl
                b_score = sum(inl)
                b_popt = popt
                b_pcov = pcov
                if sum(inl) < len(y) and sum(inl) > 0:
                    k = stop_crit(eta0, sum(inl)/len(inl), ns)

    try:
        # Use all inliers to refine model
        x_sam = x[b_inl]
        y_sam = y[b_inl]
        popt, pcov = curve_fit(sine, x_sam, y_sam, p0=p0, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        y_sine = sine(x, *list(popt))

        scores = np.abs(y_sine-y)
        b_inl = scores < thresh
        _ , _, sdl = fit_line_to_data(x[b_inl], y[b_inl])
        b_inl = scores < sdl*thresh_gain_ret
    except:
        # If the LM method is not able to fit the full inlier sample, use the minimal sample results
        if debug: print('Could not refine model by all inliers')
        popt, pcov = b_popt, b_pcov
        perr = np.sqrt(np.diag(pcov))
        y_sine = sine(x, *list(popt))

        scores = np.abs(y_sine-y) 
        b_inl = scores < thresh
        _ , _, sdl = fit_line_to_data(x[b_inl], y[b_inl])
        b_inl = scores < sdl*thresh_gain_ret

    return y_sine, popt, perr, b_inl