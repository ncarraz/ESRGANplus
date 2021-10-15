from keras import backend as K
from keras.engine.topology import Layer
import os
import tensorflow as tf

# Keras configuration directives

def SetActiveGPU(number=0):
    """
    Set visibility of GPUs to the Tensorflow engine.

    :param number: scalar or list of GPU indices
                   e.g. 0 for the 1st GPU, or [0,2] for the 1st and 3rd GPU
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not isinstance(number,list): number=[number]
    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(map(str,number))
    print ('Visible GPU(s):', os.environ["CUDA_VISIBLE_DEVICES"])

def GPUMemoryCap(fraction=1):
    """
    Limit the amount of GPU memory that can be used by an active kernel.

    :param fraction: in [0, 1], 1 = the entire available GPU memory.
    """
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    K.set_session(K.tf.Session(config=config))


# Metrics and losses
    
def plcc_tf(x, y):
    """PLCC metric"""
    xc = x - K.mean(x)
    yc = y - K.mean(y)
    return K.mean(xc*yc) / (K.std(x)*K.std(y) + K.epsilon())

def earth_mover_loss(y_true, y_pred):
    """
    Earth Mover's Distance loss.

    Reproduced from https://github.com/titu1994/neural-image-assessment/blob/master/train_inception_resnet.py
    """
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

def make_loss(loss, **params_defa):
    def custom_loss(*args, **kwargs):
        kwargs.update(params_defa)
        return loss(*args, **kwargs)
    return custom_loss

