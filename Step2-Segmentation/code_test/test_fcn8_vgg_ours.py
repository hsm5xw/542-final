# Copyright(c) 2017 Yuhao Tang, Cheng Ouyang, Haoyu Yu, Hong Moon All Rights Reserved.
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# ================================================================================================================

#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import fcn8_vgg_ours as fcn8_vgg
import utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops

img1 = scp.misc.imread("./test_data/tabby_cat.png")          # read image (tabby_cat)


fake_mask = scp.misc.imread("./test_data/tabby_cat.png",'L') # ‘L’ (8-bit pixels, black and white) # &&
msk_layer = np.expand_dims(fake_mask, axis = 2)
print(msk_layer.shape,img1.shape)
input_img = np.append(img1, msk_layer, 2)                    
print("shape",input_img.shape)

with tf.Session() as sess:                                  # open session
    images = tf.placeholder("float")
    feed_dict = {images: input_img}                         # (feed_dict is just an argument for sess.run() function)  # &&
    batch_images = tf.expand_dims(images, 0)                # (to build vgg_fcn)

    vgg_fcn = fcn8_vgg.FCN8VGG()                            # build the vgg_fcn ("VGG OBJECT") for finetuning (FCN8 VGG here)
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True)             # "" batch_images to build vgg_fcn ("VGG OBJECT")

    print('Finished building Network.')

    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")

    logging.info("Start Initializing Variabels.")

    init = tf.global_variables_initializer()
    sess.run(init)

    print('Running the Network')                            # call sess.run() function 
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up]               # ---- bring pred and pred_up elements (tensors) from "VGG OBJECT"
                                                            #               (one for downsampled output image) + (the other for upsampled output image)
    down, up = sess.run(tensors, feed_dict=feed_dict)       # ---- feed_dict defined above


    down_color = utils.color_image(down[0])                 # see utils.py file
    up_color = utils.color_image(up[0])

    scp.misc.imsave('fcn8_downsampled.png', down_color)     # save downsampled image 
    scp.misc.imsave('fcn8_upsampled.png', up_color)         # save upsampled image
