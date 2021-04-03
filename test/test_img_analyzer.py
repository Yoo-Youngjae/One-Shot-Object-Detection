from PIL import Image
import numpy as np
import cv2
from scipy.misc import imread
import torch
from matplotlib import pyplot as plt
def prep_im_for_blob(im, target_size):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    # changed to use pytorch models
    im /= 255.  # Convert range to [0,1]
    # normalization for pytroch pretrained models.
    # https://pytorch.org/docs/stable/torchvision/models.html
    pixel_means = [0.485, 0.456, 0.406]
    pixel_stdens = [0.229, 0.224, 0.225]

    # normalize manual
    im -= pixel_means  # Minus mean
    im /= pixel_stdens  # divide by stddev

    # im = im[:, :, ::-1]
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    # im = imresize(im, im_scale)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

def crop(image, purpose, size):


    cut_image = image[int(purpose[1]):int(purpose[3]),int(purpose[0]):int(purpose[2]),:]


    height, width = cut_image.shape[0:2]

    max_hw   = max(height, width)
    cty, ctx = [height // 2, width // 2]

    cropped_image  = np.zeros((max_hw, max_hw, 3), dtype=cut_image.dtype)

    x0, x1 = max(0, ctx - max_hw // 2), min(ctx + max_hw // 2, width)
    y0, y1 = max(0, cty - max_hw // 2), min(cty + max_hw // 2, height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = max_hw // 2, max_hw // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = cut_image[y0:y1, x0:x1, :]


    return cv2.resize(cropped_image, (size,size), interpolation=cv2.INTER_LINEAR)

if __name__ == '__main__':
    version = 'custom'
    if version == 'coco':
        im = imread('/home/yjyoo/PycharmProjects/data/coco/images/val2017/000000397133.jpg')
        query_im = imread('/home/yjyoo/PycharmProjects/data/coco/images/val2017/000000007816.jpg')

    else:
        im = imread('./scene.jpeg')
        _im = cv2.resize(im, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        plt.imshow(_im)
        plt.show()
        print(np.array(im).shape)           # (480, 640, 3)
        query_im = imread('./query.jpeg')
        query_im = cv2.resize(query_im, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        print(np.array(query_im).shape)     # (480, 640, 3)

    im, im_scale = prep_im_for_blob(im, target_size=600)
    im = torch.tensor(im)
    im = torch.unsqueeze(im, 0)
    im = im.transpose(1, 3)
    im = im.transpose(2, 3)



    # query_im = crop(query_im, [505.54, 53.01, 543.08, 164.09], size=128)
    query_im, query_im_scale = prep_im_for_blob(query_im, target_size=128)
    query_im = torch.tensor(query_im)
    query_im = torch.unsqueeze(query_im, 0)
    query_im = query_im.transpose(1, 3)
    query_im = query_im.transpose(2, 3)

    print(im.shape)         # torch.Size([1, 3, 600, 899])
    print(im_scale)         # 1.405152224824356
    print(query_im.shape)   # torch.Size([1, 3, 128, 128])
    print(query_im_scale)   # 1.0





# 640, 425
