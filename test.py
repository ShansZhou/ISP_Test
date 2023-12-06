import rawpy
import numpy as np
import ISP_algorithms as isp

## load Raw

raw_img_path = 'data\Indoor1_2592x1536_12bit_RGGB.raw'

raw = np.fromfile(raw_img_path, dtype=np.uint16).reshape(2592, 1536)

# Cropping
img_cropped = isp.Cropping(raw, 1024, 720)












