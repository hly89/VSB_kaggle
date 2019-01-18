from fft_dwt import *
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from numpy.fft import *
from scipy import fftpack
import pywt

train_df = pq.read_pandas("../train.parquet").to_pandas()
train_df = train_df.T
meta_tr = pd.read_csv("../metadata_train.csv")
# raw signal to low-pass signal with threshold 1e2
lp = train_df.apply(lambda x: low_pass(x, 1e2), axis=1)

lp_mean = lp.apply(np.mean, axis=1)
lp_std = lp.apply(np.std, axis=1)
lp_max = lp.apply(max, axis=1)
lp_max_idx = lp.apply(np.argmax, axis=1)
