import rawpy
import numpy as np
import matplotlib.pyplot as plt
import ISP_algorithms as isp

## load Raw

raw_img_path = 'data\ColorChecker_2592x1536_12bits_RGGB.raw'

raw = np.fromfile(raw_img_path, dtype=np.uint16).reshape(1536,2592)
plt.subplot(2,3,1)
plt.imshow(raw)
plt.xlabel('raw')

# demosaic 
demosaic_img = isp.Demosaic(raw)
plt.subplot(2,3,2)
plt.imshow(demosaic_img)
plt.xlabel('demosaic')

# color space convertion
yuv_img = isp.CSconversion(demosaic_img)
plt.subplot(2,3,3)
plt.imshow(yuv_img)
plt.xlabel('yuv_img')

# Color correction matrix
ccm_img = isp.CCmatrix(yuv_img)
plt.subplot(2,3,4)
plt.imshow(ccm_img)
plt.xlabel('CCM')

# RGB Conversion
rgb_img = isp.RGBconversion(yuv_img)
plt.subplot(2,3,5)
plt.imshow(rgb_img)
plt.xlabel('RGB')

# auto white balance
awb_img = isp.AWBalance(rgb_img)
plt.subplot(2,3,6)
plt.imshow(awb_img)
plt.xlabel('AWB')


plt.show()
# saving raw as img
# plt.imshow(raw)
plt.imsave("./results/rgb_img.png", rgb_img)







