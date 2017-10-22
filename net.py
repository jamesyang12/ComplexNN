import sys
import os

import chainer
import cupy
from chainer import function
from chainer import initializers
from chainer import utils
from chainer.utils import type_check
from chainer.utils import argument
from chainer import configuration
from chainer import functions
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


class ComplexBatchNorm(link.Link):

    """Batch normalization layer on outputs of linear or convolution functions.
    This link wraps the :func:`~chainer.functions.batch_normalization` and
    :func:`~chainer.functions.fixed_batch_normalization` functions.
    It runs in three modes: training mode, fine-tuning mode, and testing mode.
    In training mode, it normalizes the input by *batch statistics*. It also
    maintains approximated population statistics by moving averages, which can
    be used for instant evaluation in testing mode.
    In fine-tuning mode, it accumulates the input to compute *population
    statistics*. In order to correctly compute the population statistics, a
    user must use this mode to feed mini-batches running through whole training
    dataset.
    In testing mode, it uses pre-computed population statistics to normalize
    the input variable. The population statistics is approximated if it is
    computed by training mode, or accurate if it is correctly computed by
    fine-tuning mode.
    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_
    .. seealso::
       :func:`~chainer.functions.batch_normalization`,
       :func:`~chainer.functions.fixed_batch_normalization`
    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        avg_mean (numpy.ndarray or cupy.ndarray): Population mean.
        avg_var (numpy.ndarray or cupy.ndarray): Population variance.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
        ~BatchNormalization.eps (float): Epsilon value for numerical stability.
            This value is added to the batch variances.
    """

    def __init__(self, size, decay=0.9, eps=2e-5, dtype=np.float32,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None):
        super(ComplexBatchNorm, self).__init__()
        self.avg_mean = np.zeros(size, dtype=dtype)
        self.register_persistent('avg_mean')
        self.avg_var = np.zeros(size, dtype=dtype)
        self.register_persistent('avg_var')
        self.N = 0
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps


        self.moving_Vrr = Variable(cupy.full(size/2, 1./np.sqrt(2), dtype=dtype))
        self.moving_Vii = Variable(cupy.full(size/2, 1./np.sqrt(2), dtype=dtype))
        self.moving_Vri = Variable(cupy.zeros(size/2, dtype=dtype))
        self.moving_real_mean = Variable(cupy.zeros(size/2, dtype=dtype))
        self.moving_imag_mean = Variable(cupy.zeros(size/2, dtype=dtype))
      
        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma_rr = 1./np.sqrt(2)
                    initial_gamma_ri = 0
                initial_gamma_rr = initializers._get_initializer(initial_gamma_rr)
                initial_gamma_rr.dtype = dtype
                initial_gamma_ii = initializers._get_initializer(initial_gamma_rr)
                initial_gamma_ii.dtype = dtype
                initial_gamma_ri = initializers._get_initializer(initial_gamma_ri)
                initial_gamma_ri.dtype = dtype
                self.gamma_rr = variable.Parameter(initial_gamma_rr, size/2)
                self.gamma_ii = variable.Parameter(initial_gamma_ii, size/2)
                self.gamma_ri = variable.Parameter(initial_gamma_ri, size/2)

            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                initial_beta = initializers._get_initializer(initial_beta)
                initial_beta.dtype = dtype
                self.beta = variable.Parameter(initial_beta, size)

    def __call__(self, x, **kwargs):
        """__call__(self, x, finetune=False)
        Invokes the forward propagation of BatchNormalization.
        In training mode, the BatchNormalization computes moving averages of
        mean and variance for evaluation during training, and normalizes the
        input using batch statistics.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', False)``.
           See :func:`chainer.using_config`.
        Args:
            x (Variable): Input variable.
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.
        """
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
            'Use chainer.using_config')
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))



        # if hasattr(self, 'gamma_rr'):
        #     gamma_rr = self.gamma_rr
        #     gamma_ii = self.gamma_ii
        #     gamma_ri = self.gamma_ri
        # else:
        #     assert False, "yikes, no gamms"
        #     with cuda.get_device_from_id(self._device_id):
        #         gamma = variable.Variable(self.xp.ones(
        #             self.avg_mean.shape, dtype=x.dtype))
        # if hasattr(self, 'beta'):
        #     beta = self.beta
        # else:
        #     with cuda.get_device_from_id(self._device_id):
        #         beta = variable.Variable(self.xp.zeros(
        #             self.avg_mean.shape, dtype=x.dtype))




        # if configuration.config.train:
        if True:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay 

            # Brad says, "OK" -- LOL
            mu_real = F.mean(x[:,x.shape[1]/2:,:,:], axis=(0,2,3))
            mu_imag = F.mean(x[:,:x.shape[1]/2,:,:], axis=(0,2,3))
            
            #### Do the moving average stuff~~~~~
            with chainer.no_backprop_mode(): 
                self.moving_real_mean = self.moving_real_mean*self.decay + mu_real*(1-self.decay)
                self.moving_imag_mean = self.moving_imag_mean*self.decay + mu_imag*(1-self.decay)
            mu_real = self.moving_real_mean
            mu_imag = self.moving_imag_mean


            x_real = x[:,:x.shape[1]/2,:,:]
            x_imag = x[:,x.shape[1]/2:,:,:]

            # x_real_center = x[:,:x.shape[1]/2,:,:] - mu_real.data
            # b_x_real, b_mu = F.broadcast(x[:,:x.shape[1]/2,:,:], mu_real.reshape(1,mu_real.shape[0],1,1))
            b_mu_real = F.broadcast_to(F.reshape(mu_real, (1, mu_real.shape[0],1,1)), (x.shape[0], x.shape[1]/2, x.shape[2], x.shape[3]))
            x_real_center = x_real - b_mu_real

            # b_x_imag, b_mu = F.broadcast(x[:,x.shape[1]/2:,:,:], mu_imag.reshape(1,mu_imag.shape[0],1,1))
            b_mu_imag = F.broadcast_to(F.reshape(mu_imag, (1, mu_imag.shape[0],1,1)), (x.shape[0], x.shape[1]/2, x.shape[2], x.shape[3]))
            x_imag_center = x_imag - b_mu_imag
            # x_imag_center = x[:,x.shape[1]/2:,:,:] - mu_imag.data


            Vrr = F.mean(x_real_center**2, axis=(0,2,3)) + self.eps
            Vii = F.mean(x_imag_center**2, axis=(0,2,3)) + self.eps
            Vri = F.mean(x_real_center * x_imag_center, axis=(0,2,3)) + self.eps

          
            #### Do the moving average stuff~~~~~  
            with chainer.no_backprop_mode(): 
                self.moving_Vrr = self.moving_Vrr*self.decay + Vrr*(1-self.decay)
                self.moving_Vii = self.moving_Vii*self.decay + Vii*(1-self.decay)
                self.moving_Vri = self.moving_Vri*self.decay + Vri*(1-self.decay)
            Vrr = self.moving_Vrr
            Vii = self.moving_Vii
            Vri = self.moving_Vri
      

            tau = Vrr + Vii
            # delta = (Vrr * Vii) - (Vri ** 2) = Determinant. Guaranteed >= 0 because SPD
            delta = (Vrr * Vii) - (Vri ** 2)

            s = F.sqrt(delta) # Determinant of square root matrix
            t = F.sqrt(tau + 2 * s)
            inverse_st = 1.0 / (s * t)

            Wrr = (Vii + s) * inverse_st
            Wii = (Vrr + s) * inverse_st
            Wri = -Vri * inverse_st

            # Wrr_b = F.broadcast_to(Wrr.reshape(1,Wrr.shape[0], 1,1), x_real_center.shape)
            # Wii_b = F.broadcast_to(Wii.reshape(1,Wii.shape[0], 1,1), x_real_center.shape)
            # Wri_b = F.broadcast_to(Wri.reshape(1,Wri.shape[0], 1,1), x_real_center.shape)

            Wrr_b = F.broadcast_to(F.reshape(Wrr, (1,Wrr.shape[0], 1,1)), x_real_center.shape)
            Wii_b = F.broadcast_to(F.reshape(Wii, (1,Wii.shape[0], 1,1)), x_real_center.shape)
            Wri_b = F.broadcast_to(F.reshape(Wri, (1,Wri.shape[0], 1,1)), x_real_center.shape)

            x_tilda_real = Wrr_b.data*x_real_center + Wri_b.data*x_imag_center
            x_tilda_imag = Wri_b.data*x_real_center + Wii_b.data*x_imag_center

            # gamma_rr = F.broadcast_to(self.gamma_rr.reshape(1,self.gamma_rr.shape[0], 1,1), x_tilda_real.shape)
            # gamma_ri = F.broadcast_to(self.gamma_ri.reshape(1,self.gamma_ri.shape[0], 1,1), x_tilda_real.shape)
            # gamma_ii  = F.broadcast_to(self.gamma_ii.reshape(1,self.gamma_ii.shape[0], 1,1), x_tilda_real.shape)

            gamma_rr_b = F.broadcast_to(F.reshape(self.gamma_rr, (1,self.gamma_rr.shape[0], 1,1)), x_tilda_real.shape)
            gamma_ri_b = F.broadcast_to(F.reshape(self.gamma_ri, (1,self.gamma_ri.shape[0], 1,1)), x_tilda_real.shape)
            gamma_ii_b = F.broadcast_to(F.reshape(self.gamma_ii, (1,self.gamma_ii.shape[0], 1,1)), x_tilda_real.shape)

            x_final_real = gamma_rr_b * x_tilda_real + gamma_ri_b * x_tilda_imag
            x_final_imag = gamma_ri_b * x_tilda_real + gamma_ii_b * x_tilda_imag
            x_final_real = x_final_real + F.broadcast_to(F.reshape(self.beta[:self.beta.shape[0]/2], (1,self.beta.shape[0]/2, 1,1)),
                                                             (x_final_real.shape))
            x_final_imag = x_final_imag + F.broadcast_to(F.reshape(self.beta[self.beta.shape[0]/2:], (1,self.beta.shape[0]/2, 1,1)),
                                                             (x_final_imag.shape))

            # print self.gamma_rr.debug_print()
            # print x_final_real.debug_print()

            return F.concat([x_final_real,x_final_imag], axis=1)


            # keep to know what format exactly is being returned
            # ret = functions.batch_normalization(
                # x, gamma, beta, eps=self.eps, running_mean=self.avg_mean,
                # running_var=self.avg_var, decay=decay)
            # print type(ret), ret.shape
        else:
            # Use running average statistics or fine-tuned statistics.
            assert False
            mean = variable.Variable(self.avg_mean)
            var = variable.Variable(self.avg_var)
            ret = functions.fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps)
        return ret

    def start_finetuning(self):
        """Resets the population count for collecting population statistics.
        This method can be skipped if it is the first time to use the
        fine-tuning mode. Otherwise, this method should be called before
        starting the fine-tuning mode again.
        """
        self.N = 0


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

class ComplexSmallNN(chainer.Chain):
    def __init__(self, output_dim, init_weights=False, filter_height=1):
        super(ComplexSmallNN, self).__init__()
        self.output_dim = output_dim
        self.filter_height = filter_height
        with self.init_scope():
            if init_weights:
                # assert False, "Not Implemented Complex Initialization"
                self.conv1 = ComplexConv2D(None,  96, (1, 3), 1, (0, 1), initialW=ComplexInitial())
                self.conv2 = ComplexConv2D(None, 256, (filter_height, 3), 1, (0, 1), initialW=ComplexInitial())
                self.fc6 = L.Linear(None, 4096, initialW=HeNormal())
                self.fc7 = L.Linear(None, output_dim, initialW=HeNormal())
                self.bn1 = ComplexBatchNorm(96)
                self.bn2 = ComplexBatchNorm(256)
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
        # h = F.max_pooling_2d(F.local_response_normalization(
            # F.relu(self.conv1(x_t))), (1,3), stride=1)
        h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x_t))), (1,3), stride=1)
        # h = F.max_pooling_2d(F.relu(self.conv1(x_t)), (1,3), stride=1)
        # print self.bn1.gamma_rr.data[:5]
        # print self.bn1.gamma_rr.debug_print()
        # print self.conv1.W.debug_print()
        h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), (1,3), stride=1)
        # h = F.max_pooling_2d(F.relu(self.conv2(h)), (1,3), stride=1)
        h = F.dropout(F.relu(self.fc6(h)))
        h = self.fc7(h)
        return h



class ComplexNN(chainer.Chain):
    def __init__(self, output_dim, init_weights=False, filter_height=1):
        super(ComplexNN, self).__init__()
        self.output_dim = output_dim
        self.filter_height = filter_height
        with self.init_scope():
            if init_weights:
                # assert False, "Not Implemented Complex Initialization"
                self.conv1 = ComplexConv2D(None,  96, (1, 3), 1, (0, 1), initialW=ComplexInitial())
                self.conv2 = ComplexConv2D(None, 256, (filter_height, 3), 1, (0, 1), initialW=ComplexInitial())
                self.conv3 = ComplexConv2D(None, 384, (1, 3), 1, (0, 1), initialW=ComplexInitial())
                self.conv4 = ComplexConv2D(None, 384, (1, 3), 1, (0, 1), initialW=ComplexInitial())
                self.conv5 = ComplexConv2D(None, 256, (1, 3), 1, (0, 1), initialW=ComplexInitial())
                self.fc6 = L.Linear(None, 4096, initialW=HeNormal())
                self.fc7 = L.Linear(None, 4096, initialW=HeNormal())
                self.fc8 = L.Linear(None, output_dim, initialW=HeNormal())
                self.bn1 = ComplexBatchNorm(96)
                self.bn2 = ComplexBatchNorm(96)
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
        print self.conv1.W.debug_print()
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
                self.conv1 = L.Convolution2D(None,  96, (1, 3), 1, (0, 1), initialW=GlorotNormal())
                self.conv2 = L.Convolution2D(None, 256, (filter_height, 3), 1, (0, 1), initialW=GlorotNormal())
                self.conv3 = L.Convolution2D(None, 384, (1, 3), 1, (0, 1), initialW=GlorotNormal())
                self.conv4 = L.Convolution2D(None, 384, (1, 3), 1, (0, 1), initialW=GlorotNormal())
                self.conv5 = L.Convolution2D(None, 256, (1, 3), 1, (0, 1), initialW=GlorotNormal())
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
        
       