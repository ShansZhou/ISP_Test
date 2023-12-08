import numpy as np
from scipy.signal import correlate2d


# Cropping
def Cropping(raw, height, witdth):
    pass


# Dead pixel correction
def DPcorrection(raw):
    pass


# Black level correction
def BLcorrection(raw):
    pass

# Lens shading correction
def LScorrection(raw):
    pass


# Bayer noise reduction
def BNReduction(raw):
    pass

# Auto White Balance
def AWBalance(raw):
    rgb_avg = raw.mean(axis=0)

    red_gain = rgb_avg[1]/ rgb_avg[0]
    blue_gain = rgb_avg[1] / rgb_avg[2]

    raw[0] = raw[0]* red_gain
    raw[2] = raw[2]* blue_gain

    return raw
    

# CFA demosaicing
def Demosaic(raw):

    # initial r,g,b channels wrt dictionary
    channels = dict(
        (channel, np.zeros(raw.shape, dtype=bool)) for channel in 'rgb'
    )

    # create bayer layer respectively for each channel with particular pattern
    for channel, (y_channel, x_channel) in zip('rggb', [(0,0),(0,1),(1,0),(1,1)]):
        channels[channel][y_channel::2, x_channel::2] = True

    ######## Malvar He Cutler ########
    raw = np.float32(raw)
    # initial mask of r,g,b
    mask_r, mask_g, mask_b = channels['r'],channels['g'],channels['b']
    # initial a demosaic container with same dememtion of raw image
    demos_out = np.empty((raw.shape[0], raw.shape[1],3))

    # 5x5 filiter
    # g_channel at r_channel & b_channel location,
    g_at_r_and_b = (
        np.float32(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 2, 0, 0],
                [-1, 2, 4, 2, -1],
                [0, 0, 2, 0, 0],
                [0, 0, -1, 0, 0],
            ]
        )
        * 0.125
    )

    # r_channel at green in r_channel row & b_channel column --
    # b_channel at green in b_channel row & r_channel column
    r_at_gr_and_b_at_gb = (
        np.float32(
            [
                [0, 0, 0.5, 0, 0],
                [0, -1, 0, -1, 0],
                [-1, 4, 5, 4, -1],
                [0, -1, 0, -1, 0],
                [0, 0, 0.5, 0, 0],
            ]
        )
        * 0.125
    )

    # r_channel at green in b_channel row & r_channel column --
    # b_channel at green in r_channel row & b_channel column
    r_at_gb_and_b_at_gr = np.transpose(r_at_gr_and_b_at_gb)

    # r_channel at blue in b_channel row & b_channel column --
    # b_channel at red in r_channel row & r_channel column
    r_at_b_and_b_at_r = (
        np.float32(
            [
                [0, 0, -1.5, 0, 0],
                [0, 2, 0, 2, 0],
                [-1.5, 0, 6, 0, -1.5],
                [0, 2, 0, 2, 0],
                [0, 0, -1.5, 0, 0],
            ]
        )
        * 0.125
    )

    r_channel = raw * mask_r
    g_channel = raw * mask_g
    b_channel = raw * mask_b

    # Creating g_channel channel first after applying g_at_r_and_b filter
    g_channel = np.where(
        np.logical_or(mask_r == 1, mask_b == 1),
        correlate2d(raw, g_at_r_and_b, mode="same", boundary="symm"),
        g_channel,
    )

    # Applying other linear filters
    rb_at_g_rbbr = correlate2d(
        raw, r_at_gr_and_b_at_gb, mode="same", boundary="symm"
    )
    rb_at_g_brrb = correlate2d(
        raw, r_at_gb_and_b_at_gr, mode="same", boundary="symm"
    )
    rb_at_gr_bbrr = correlate2d(
        raw, r_at_b_and_b_at_r, mode="same", boundary="symm"
    )
        
    # Extracting Red rows.
    r_rows = np.transpose(np.any(mask_r == 1, axis=1)[np.newaxis]) * np.ones(
        r_channel.shape, dtype=np.float32
    )

    # Extracting Red columns.
    r_col = np.any(mask_r == 1, axis=0)[np.newaxis] * np.ones(
        r_channel.shape, dtype=np.float32
    )

    # Extracting Blue rows.
    b_rows = np.transpose(np.any(mask_b == 1, axis=1)[np.newaxis]) * np.ones(
        b_channel.shape, dtype=np.float32
    )

    # Extracting Blue columns
    b_col = np.any(mask_b == 1, axis=0)[np.newaxis] * np.ones(
        b_channel.shape, dtype=np.float32
    )
    r_channel = np.where(
        np.logical_and(r_rows == 1, b_col == 1), rb_at_g_rbbr, r_channel
    )
    r_channel = np.where(
        np.logical_and(b_rows == 1, r_col == 1), rb_at_g_brrb, r_channel
    )

    # Similarly for B channel we have to update pixels at
    # [r_channel rows and b_channel cols]
    # & at [b_channel rows and r_channel cols] 3 pixels need
    # to be updated near one given b_channel
    b_channel = np.where(
        np.logical_and(b_rows == 1, r_col == 1), rb_at_g_rbbr, b_channel
    )
    b_channel = np.where(
        np.logical_and(r_rows == 1, b_col == 1), rb_at_g_brrb, b_channel
    )

    # Final r_channel & b_channel channels
    r_channel = np.where(
        np.logical_and(b_rows == 1, b_col == 1), rb_at_gr_bbrr, r_channel
    )
    b_channel = np.where(
        np.logical_and(r_rows == 1, r_col == 1), rb_at_gr_bbrr, b_channel
    )

    demos_out[:, :, 0] = r_channel
    demos_out[:, :, 1] = g_channel
    demos_out[:, :, 2] = b_channel

    # Clipping the pixels values within the bit range
    demos_out = np.clip(demos_out, 0, 2**12 - 1)
    demos_out = np.uint16(demos_out)

    return demos_out


# Color correction matrix
def CCmatrix(raw):
    row_1 = np.array([1.660, -0.527, -0.133])
    row_2 = np.array([-0.408, 1.563, -0.082])
    row_3 = np.array([-0.055, -1.641, 2.695])

    ccm_mat = np.float32([row_1,row_2,row_3])
    
    # normalize nbit to 0-1 img
    raw = np.float32(raw)/ (2**12 - 1)

    # convert to nx3
    img1 = raw.reshape(raw.shape[0]*raw.shape[1],3)

    # keeping imatest convention of colum sum to 1 mat. O*A => A = ccm
    out = np.matmul(img1, ccm_mat.transpose())

    # clipping after ccm is must to eliminate neg values
    out = np.float32(np.clip(out, 0, 1))

    # convert back
    out = out.reshape(raw.shape).astype(raw.dtype)
    out = np.uint16(out * (2**12 - 1))

    return out

# Auto-Exposure
def AExposure(raw):
    pass


# Color space conversion
def CSconversion(raw):

    bit_depth = 12
    # BT. 709
    rgb2yuv_mat = np.array([[ 47,  157,  16],
                            [-26,  -86, 112],
                            [112, -102, -10]])
    
    # make nx3 2d matrix of image
    mat_2d = raw.reshape((raw.shape[0] * raw.shape[1], 3))

    # convert to 3xn for matrix multiplication
    mat2d_t = mat_2d.transpose()

    # convert to YUV
    yuv_2d = np.matmul(rgb2yuv_mat, mat2d_t)

    # convert image with its provided bit_depth
    yuv_2d = np.float64(yuv_2d) / (2**8)
    yuv_2d = np.where(yuv_2d >= 0, np.floor(yuv_2d + 0.5), np.ceil(yuv_2d - 0.5))

    # black-level/DC offset added to YUV values
    yuv_2d[0, :] = 2 ** (bit_depth / 2) + yuv_2d[0, :]
    yuv_2d[1, :] = 2 ** (bit_depth - 1) + yuv_2d[1, :]
    yuv_2d[2, :] = 2 ** (bit_depth - 1) + yuv_2d[2, :]

    # reshape the image back
    yuv2d_t = yuv_2d.transpose()

    yuv2d_t = np.clip(yuv2d_t, 0, (2**bit_depth) - 1)

    # Modules after CSC need 8-bit YUV so converting it into 8-bit after Normalizing.
    yuv2d_t = yuv2d_t / (2 ** (bit_depth - 8))
    yuv2d_t = np.where(
        yuv2d_t >= 0, np.floor(yuv2d_t + 0.5), np.ceil(yuv2d_t - 0.5)
    )

    yuv2d_t = np.clip(yuv2d_t, 0, 255)

    raw = yuv2d_t.reshape(raw.shape).astype(np.uint8)

    return raw

def RGBconversion(yuv_img):
    # make nx3 2d matrix of image
    mat_2d = yuv_img.reshape(
        (yuv_img.shape[0] * yuv_img.shape[1], 3)
    )

    # convert to 3xn for matrix multiplication
    mat2d_t = mat_2d.transpose()

    # subract the offsets
    mat2d_t = mat2d_t - np.array([[16, 128, 128]]).transpose()

    # for BT. 709
    yuv2rgb_mat = np.array([[74, 0, 114], [74, -13, -34], [74, 135, 0]])

    # convert to RGB
    rgb_2d = np.matmul(yuv2rgb_mat, mat2d_t)
    rgb_2d = rgb_2d >> 6

    # reshape the image back
    rgb2d_t = rgb_2d.transpose()
    yuv_img = rgb2d_t.reshape(yuv_img.shape).astype(np.float32)

    # clip the resultant img as it can have neg rgb values for small Y'
    yuv_img = np.float32(np.clip(yuv_img, 0, 255))

    # convert the image to [0-255]
    yuv_img = np.uint8(yuv_img)
    return yuv_img