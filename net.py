import sys
import os

import chainer
import cupy
from chainer import function
from chainer import initializers
from chainer import utils
from chainer.utils import type_check
import chainer.functions as F
import chainer.links as L
from chainer import link
from chainer.initializers import GlorotNormal, HeNormal
from chainer.functions.connection import convolution_2d
from chainer import variable
from chainer import Variable
from chainer import cuda
import numpy as np

from chainer import initializer

class ComplexConv2D(link.Link):
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                nobias=False, initialW=None, initial_bias=None, **kwargs):
        super(ComplexConv2D, self).__init__()
        
        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.out_channels = out_channels
        self.in_channels = in_channels

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_channels is not None:
                self._initialize_params(in_channels)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_channels)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        ## CAREFUL!!!!
        W_shape = (self.out_channels, in_channels/2, kh, kw)
        self.W.initialize(W_shape)

    def __call__(self, x):
        """Applies the convolution layer.
        Args:
            x (~chainer.Variable): Input image.
        Returns:
            ~chainer.Variable: Output of the convolution.
        """
        if self.W.data is None:
            self._initialize_params(x.shape[1])

        wid = self.W.shape[0]/2
        W = F.split_axis(self.W, 2, axis=0)
        b = F.split_axis(self.b, 2, axis=0)
        x_real = x[:,:x.shape[1]/2,:,:]
        x_imag = x[:,x.shape[1]/2:,:,:]

        xr_Wr = convolution_2d.convolution_2d(
            x_real, W[0], b[0], self.stride, self.pad) 
        xi_Wr = convolution_2d.convolution_2d(
            x_imag, W[0], b[0], self.stride, self.pad) 
        xr_Wi = convolution_2d.convolution_2d(
            x_real, W[1], b[1], self.stride, self.pad) 
        xi_Wi = convolution_2d.convolution_2d(
            x_imag, W[1], b[1], self.stride, self.pad) 

        r = xr_Wr - xi_Wi
        i = xr_Wi + xi_Wr
        return F.concat([r,i], axis=1)



class ComplexNN(chainer.Chain):
    def __init__(self, output_dim, init_weights=False, filter_height=1):
        super(ComplexNN, self).__init__()
        self.output_dim = output_dim
        self.filter_height = filter_height
        with self.init_scope():
            if init_weights:
                # assert False, "Not Implemented Complex Initialization"
                self.conv1 = ComplexConv2D(None,  96, (1, 3), 1, (0, 1), initialW=ComplexInitial('glorot'))
                self.conv2 = ComplexConv2D(None, 256, (filter_height, 3), 1, (0, 1), initialW=ComplexInitial('glorot'))
                self.conv3 = ComplexConv2D(None, 384, (1, 3), 1, (0, 1), initialW=ComplexInitial('glorot'))
                self.conv4 = ComplexConv2D(None, 384, (1, 3), 1, (0, 1), initialW=ComplexInitial('glorot'))
                self.conv5 = ComplexConv2D(None, 256, (1, 3), 1, (0, 1), initialW=ComplexInitial('glorot'))
                self.fc6 = L.Linear(None, 4096, initialW=HeNormal())
                self.fc7 = L.Linear(None, 4096, initialW=HeNormal())
                self.fc8 = L.Linear(None, output_dim, initialW=HeNormal())
            else:
                self.conv1 = ComplexConv2D(None,  96, (1, 3), 1, (0, 1))
                self.conv2 = ComplexConv2D(None, 256, (filter_height, 3), 1, (0, 1))
                self.conv3 = ComplexConv2D(None, 384, (1, 3), 1, (0, 1))
                self.conv4 = ComplexConv2D(None, 384, (1, 3), 1, (0, 1))
                self.conv5 = ComplexConv2D(None, 256, (1, 3), 1, (0, 1))
                self.fc6 = L.Linear(None, 4096)
                self.fc7 = L.Linear(None, 4096)
                self.fc8 = L.Linear(None, output_dim)

    def __call__(self, x):
        x_t = F.transpose(x, (0,2,1,3))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x_t))), (1,3), stride=1)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), (1,3), stride=1)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), (1,3), stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        return h

class Alex(chainer.Chain):
    def __init__(self, output_dim, init_weights=False, filter_height=1):
        super(Alex, self).__init__()
        self.output_dim = output_dim
        self.filter_height = filter_height
        with self.init_scope():
            if init_weights:
                self.conv1 = L.Convolution2D(None,  96, (1, 3), 1, (0, 1), initialW=ComplexInitial('glorot'))
                self.conv2 = L.Convolution2D(None, 256, (filter_height, 3), 1, (0, 1), initialW=ComplexInitial('glorot'))
                self.conv3 = L.Convolution2D(None, 384, (1, 3), 1, (0, 1), initialW=ComplexInitial('glorot'))
                self.conv4 = L.Convolution2D(None, 384, (1, 3), 1, (0, 1), initialW=ComplexInitial('glorot'))
                self.conv5 = L.Convolution2D(None, 256, (1, 3), 1, (0, 1), initialW=ComplexInitial('glorot'))
                self.fc6 = L.Linear(None, 4096, initialW=HeNormal())
                self.fc7 = L.Linear(None, 4096, initialW=HeNormal())
                self.fc8 = L.Linear(None, output_dim, initialW=HeNormal())
            else:
                self.conv1 = L.Convolution2D(None,  96, (1, 3), 1, (0, 1))
                self.conv2 = L.Convolution2D(None, 256, (filter_height, 3), 1, (0, 1))
                self.conv3 = L.Convolution2D(None, 384, (1, 3), 1, (0, 1))
                self.conv4 = L.Convolution2D(None, 384, (1, 3), 1, (0, 1))
                self.conv5 = L.Convolution2D(None, 256, (1, 3), 1, (0, 1))
                self.fc6 = L.Linear(None, 4096)
                self.fc7 = L.Linear(None, 4096)
                self.fc8 = L.Linear(None, output_dim)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), (1,3), stride=1)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), (1,3), stride=1)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), (1,3), stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        return h

## from chainer - copied (chainer.link convolution_2d.py)
def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x



class ComplexInitial(initializer.Initializer):


    def __init__(self, scale=1.0, dtype=None, criterion='glorot'):
        self.scale = scale
        self.criterion = criterion
        super(ComplexInitial, self).__init__(dtype)


    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        fan_in, fan_out = initializer.get_fans(array.shape)

        if self.criterion == 'glorot':
            s = np.sqrt(1. / (fan_in + fan_out))
        elif self.criterion == 'he':
            s = np.sqrt(1. / fan_in)
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)
        
        
        xp = cuda.get_array_module(array)
        if xp is not np:
            # Only CuPy supports dtype option
            if self.dtype == np.float32 or self.dtype == np.float16:
                # float16 is not supported in cuRAND
                args['dtype'] = np.float32

        [a,b,c,d] = array.shape
        #### Rayleigh distribution
        x_ran = xp.random.uniform(0,1,[a/2,b,c,d])
        modulus = s * xp.sqrt(-2*xp.log(x_ran))

        phase = xp.random.uniform(low=-np.pi, high=np.pi, size=[a/2,b,c,d])

        weight_real = modulus * xp.cos(phase)
        weight_imag = modulus * xp.sin(phase)
        weight = xp.concatenate([weight_real, weight_imag])

        array[...] = weight
        
       