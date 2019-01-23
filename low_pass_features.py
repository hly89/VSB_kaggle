from fft_dwt import *
import pandas as pd
#import pyarrow.parquet as pq
import numpy as np
from numpy.fft import *
from scipy import fftpack
import pywt


def lp_features_extraction(sig, thred):
    # raw signal to low-pass signal with threshold 1e2/1e3
    lp = sig.apply(lambda x: low_pass(x, thred), axis=1)
    lp_mean = lp.apply(np.mean, axis=1)
    lp_std = lp.apply(np.std, axis=1)
    lp_max = lp.apply(max, axis=1)
    lp_max_idx = lp.apply(np.argmax, axis=1)
    lp_features = pd.concat([lp_max_idx, lp_max, lp_mean, lp_std], axis=1)
    lp_features.columns=['lp_max_idx', 'lp_max', 'lp_mean', 'lp_std']

    
    return lp_features