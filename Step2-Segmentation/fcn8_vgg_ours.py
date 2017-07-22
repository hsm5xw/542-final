# Copyright(c) 2017 Yuhao Tang, Cheng Ouyang, Haoyu Yu, Hong Moon All Rights Reserved.
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# ================================================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]   # mean pixel values taken from the VGG authors


class FCN8VGG:

    def __init__(self, vgg16_npy_path=None):                                        # class constructor
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "vgg16.npy")                                  # .npy file = VGG weights
            vgg16_npy_path = path
            logging.info("Load npy file from '%s'.", vgg16_npy_path)
            # load the vgg16 parameters
        if not os.path.isfile(vgg16_npy_path):
            logging.error(("File '%s' not found. Download it from "
                           "ftp://mi.eng.cam.ac.uk/pub/mttt2/"
                           "models/vgg16.npy"), vgg16_npy_path)
            sys.exit(1)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()          # data_dict = load VGG weights
        # data_dict structure:
        # keys: 'conv5_1', 'fc6', 'conv5_3', 'conv5_2', 'fc8', 'fc7', 'conv4_1',
        #  'conv4_2', 'conv4_3', 'conv3_3', 'conv3_2', 'conv3_1', 'conv1_1', 'conv1_2', 
        #  'conv2_2', 'conv2_1'] 
        # each value is a list, [0] is the W, [1] is the bias

        self.wd = 5e-4                                                              # weight decay (see paper section 4.3)
        print("npy file loaded")

    def build(self, rgbm, train=False, num_classes=2, random_init_fc8=False,        # build the vgg_fcn ("VGG OBJECT") for "FINE-TUNING" (FCN8 VGG here)
              debug=False):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        @ rgb: image batch tensor
                    Image in rgb shap. Scaled to Intervall [0, 255]
        @ train: bool
                    Whether to build train or inference graph
        @ num_classes: int
                    How many classes should be predicted (by fc8)
        @ random_init_fc8 : bool
                    Whether to initialize fc8 layer randomly.
                    Finetuning is required in this case.
        @ debug: bool
                    Whether to print additional Debug Information.
        """
        # Convert RGB to BGR

        with tf.name_scope('Processing'):

            red, green, blue, mask = tf.split(rgbm, 4, 3)               # split RGB
            #red, green, blue = tf.split(rgb, 3, 3)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            # concatenate the fourth dimension, which is the color layer
            # the first dimension is the image dimension (different images)
            bgrm = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
                mask
            ], 3)
            # now bgr is: [batch, in_height, in_width, in_channels]

            if debug:
                # print all input dimension size, summarize = 4
                bgrm = tf.Print(bgrm, [tf.shape(bgrm)],
                               message='Shape of input image: ',
                               summarize=4, first_n=1)

                                                                        # ============== VGG architecture... (see VGG architecture diagram) ===================

        self.conv1_1 = self._conv_layer(bgrm, "conv1_1")                # 2conv-1pool  (1)
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")          # 2conv-1pool  (2)
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")          # 3conv-1pool  (3)
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")          # 3conv-1pool  (4)
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")          # 3conv-1pool  (5)
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)


        self.fc6 = self._fc_layer(self.pool5, "fc6")                    # fc layer (6)
        if train:                                                       #    ""     train or inference?
            self.fc6 = tf.nn.dropout(self.fc6, 0.5)

        self.fc7 = self._fc_layer(self.fc6, "fc7")                      # fc layer (7)
        if train:                                                       #    ""     train or inference?
            self.fc7 = tf.nn.dropout(self.fc7, 0.5)



        if random_init_fc8:
            self.score_fr = self._score_layer(self.fc7, "score_fr",     #  ignore this! This value defaults to false. So this is never executed.
                                              num_classes)
        else:
            self.score_fr = self._fc_layer(self.fc7, "score_fr",        #  fc layer (8)
                                           num_classes=num_classes,
                                           relu=False)

        self.pred = tf.argmax(self.score_fr, dimension=3)                   # prediction ???? 
                                                                            #       (one for downsampled output image)



        # ============================================================      # (skip architecture)
        self.upscore2 = self._upscore_layer(self.score_fr,                  # upscore layer (a.k.a deconvolution layer)
                                            shape=tf.shape(self.pool4),     #       upsample "score_fr" layer to match dimensions with pool4 layer !!
                                            num_classes=num_classes,        #       * note: stride 2
                                            debug=debug, name='upscore2',
                                            ksize=4, stride=2)
        self.score_pool4 = self._score_layer(self.pool4, "score_pool4",     #       create score_pool4 layer
                                             num_classes=num_classes)
        self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)           #       FUSE pool4 layer ! ===  (see Figure 3 on paper)



        # ============================================================      # (skip architecture)
        self.upscore4 = self._upscore_layer(self.fuse_pool4,                #       upscore layer (a.k.a deconvolution layer)
                                            shape=tf.shape(self.pool3),     #       upsample "fuse_pool4" layer to match dimensions with pool3 layer !!
                                            num_classes=num_classes,        #       * note: stride 2
                                            debug=debug, name='upscore4',
                                            ksize=4, stride=2)
        self.score_pool3 = self._score_layer(self.pool3, "score_pool3",     #       create score_pool3 layer
                                             num_classes=num_classes)
        self.fuse_pool3 = tf.add(self.upscore4, self.score_pool3)           #       FUSE pool3 layer ! ===  (see Figure 3 on paper)



        # ============================================================      # final upsampling to match dimension with bgr (normalized batch_images)
        self.upscore32 = self._upscore_layer(self.fuse_pool3,               #       upscore layer (a.k.a deconvolution layer)
                                             shape=tf.shape(bgrm),          #       * note: stride 8 (8x upsampled)
                                             num_classes=num_classes,
                                             debug=debug, name='upscore32',
                                             ksize=16, stride=8)

        self.pred_up = tf.argmax(self.upscore32, dimension=3)               # plug in final upscore layer -> pred upsampled 
                                                                            #        (another for upsampled output image)
        if debug:
            tf.Print(self.pred_up, [tf.shape(self.pred_up)],
                               message='Shape of output image: ',
                               summarize=4, first_n=1)


    # max pool layer
    def _max_pool(self, bottom, name, debug):

        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            # tf.Print returns same tensor as input pool
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    # conv layer 
    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            
            if (name == "conv1_1"):                                         # && NEW
                filt_rgb = self.get_conv_filter(name) 
                filt_mask = tf.Variable(tf.truncated_normal([3, 3, 1, 64], dtype=tf.float32,
                                                     stddev=1e-1))
                filt = tf.concat([filt_rgb,filt_mask],2)
            else:
                filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME') # 'SAME' convolution
                                                                            # apply the corresponding filter ([1,1,1,1] are strides...)
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)                                         # relu
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu                 # return relu

    # fc layer (fully connected layer)
    def _fc_layer(self, bottom, name, num_classes=None,
                  relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
                                                                            # f_shape = [ksize, ksize, num_classes, in_features]

            if name == 'fc6':                                               # filter (kernel) for fc6 layer
                filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])

            elif name == 'score_fr':                                        # filter (kernel) for score_fr layer: 1x1 convolution
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                                  num_classes=num_classes)
            else:                                                           # filter (kernel): 1x1 convolutioin (for fc7 ?)
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])

            self._add_wd_and_summary(filt, self.wd, "fc_wlosses")

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME') # convolution
            conv_biases = self.get_bias(name, num_classes=num_classes)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)                                     # relu activation
            _activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias

    def _score_layer(self, bottom, name, num_classes):
        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]

            # He initialization Sheme
            if name == "score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "score_pool4":
                stddev = 0.001
            elif name == "score_pool3":
                stddev = 0.0001

            # Apply convolution
            w_decay = self.wd
            weights = self._variable_with_weight_decay(shape, stddev, w_decay,
                                                       decoder=True)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')  # convolution

            # Apply bias
            conv_biases = self._bias_variable([num_classes], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            _activation_summary(bias)

            return bias

    def _upscore_layer(self, bottom, shape,             # (a.k.a) deconvolution layer - Able to output fine results
                       num_classes, name, debug,        #                             - (transposed convolution with stride)
                       ksize=4, stride=2):              #                             -  upsample images by stride times (default value 2)
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1    # insert (upscale_factor-1) zeros between each successive pixels
                w = ((in_shape[2] - 1) * stride) + 1    # insert (upscale_factor-1) zeros between each successive pixels
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.stack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape)                           # get filter for deconvolution layer (with specified f_shape)
            self._add_wd_and_summary(weights, self.wd, "fc_wlosses")
                                                                                # (transposed convolution with stride)            
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,      # it is NOT just conv2d, but is conv2d_TRANSPOSE !!!
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

        _activation_summary(deconv)
        return deconv

    def get_deconv_filter(self, f_shape):                    # called from Upscore layer !
        width = f_shape[0]
        heigh = f_shape[0]

        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)

        bilinear = np.zeros([f_shape[0], f_shape[1]])       # f_shape is input to the function (2-D array filled with zeros)

        for x in range(width):                              # iterate through and fill in values in 2-D array
            for y in range(heigh):                          
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="up_filter", initializer=init,
                              shape=weights.shape)
        return var

    def get_conv_filter(self, name):                                    # called from Conv layer (to get conv filter) !
        init = tf.constant_initializer(value=self.data_dict[name][0],   # data_dict = load VGG weights
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape                           # @name = Ex) "conv1_1"
                                                                        # load VGG weights with corresponding names

        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))

        var = tf.get_variable(name="filter", initializer=init, shape=shape) # varaible initialized to VGG weights

        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        _variable_summaries(var)
        return var

    def get_bias(self, name, num_classes=None):

        bias_wights = self.data_dict[name][1]           # data_dict = load VGG weights
        shape = self.data_dict[name][1].shape

        if name == 'fc8':
            bias_wights = self._bias_reshape(bias_wights, shape[0],     # reshapes the original weights to be used in a fully-convolutional layer 
                                             num_classes)               # which produces NUM_NEW_CLASSES.
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape) # variable initialized to VGG weights
        _variable_summaries(var)
        return var

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        _variable_summaries(var)
        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`

        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)

        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements

            avg_idx = start_idx//n_averaged_elements    # which index to store at avg_bweight

            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])

        return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):            # called from get_fc_weight_reshape(), which was called from fc layer
        """ Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. 

        This reshapes the original weights
        to be used in a fully-convolutional layer which produces "NUM_NEW CLASSES". 
        To archive this the "AVERAGE" (mean) of n adjacent classes is taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
          @ fweight: "ORIGINAL" weights
          @ shape: shape of the "DESIRED" fully-convolutional layer
          @ num_new: number of new classes


        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)

        # // is “floor” division (rounds down to nearest whole number)
        # avg_fweight is a new tensor with num_new output channels

        n_averaged_elements = num_orig//num_new 
        avg_fweight = np.zeros(shape)   # to return

        for i in range(0, num_orig, n_averaged_elements):   # for-loop with step-size of n_averaged_elements
            start_idx = i
            end_idx = start_idx + n_averaged_elements

            avg_idx = start_idx//n_averaged_elements    # which index to store at avg_fweight
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean( fweight[:, :, :, start_idx:end_idx], axis=3)

        return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd, decoder=False):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def _add_wd_and_summary(self, var, wd, collection_name=None):
        if collection_name is None:
            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def _bias_variable(self, shape, constant=0.0):                  # bias variable initialized to constant 0.0 (called from score layer)
        initializer = tf.constant_initializer(constant)
        var = tf.get_variable(name='biases', shape=shape,
                              initializer=initializer)
        _variable_summaries(var)
        return var

    def get_fc_weight_reshape(self, name, shape, num_classes=None): # get weights for fc layer (called from fc layer !!) 
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)

        weights = self.data_dict[name][0]   # load VGG weights with corresponding names
        weights = weights.reshape(shape)    # reshape tensor with specified dimensions

        if num_classes is not None:                                 # (only executed when num_classes is not None !)
            weights = self._summary_reshape(weights, shape,         # reshapes the original weights to be used in a fully-convolutional layer  
                                            num_new=num_classes)    # which produces NUM_NEW_CLASSES.
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var

# ------------------------- end of class functions for FCN8VGG -----------------------------------------------------------

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.info("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)

def pixel_wise_cross_entropy(logits, labels, num_classes):  # && NEW
    # self.cross_entropy = tf.reduce_mean(          
    #   tf.nn.softmax_cross_entropy_with_logits(
    #        labels=tf.reshape(self.label, [-1, 21]), logits=tf.reshape(self.deconv1, [-1,21])))
    # print(logits.get_shape().as_list())

    # logits = tf.reshape(logits, (-1, num_classes))
    # print(logits.get_shape().as_list())
    print(labels.get_shape().as_list())

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.to_int32(labels/255), logits=logits)) 
    # weight decay
    lambda_ = 5**(-4)
    #l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_weight_loss')
    cross_entropy += lambda_*l2_loss
    return cross_entropy

