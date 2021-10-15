from . import generic, tensor_ops, image_utils
from . import generators, model_helper, applications
import sys
from keras import backend as K
if sys.version_info[0] < 3:
    if K.backend()=='tensorflow':
        K.set_image_dim_ordering("tf")

# remove tensorflow warning
import logging
class WarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        tf_warning = 'retry (from tensorflow.contrib.learn.python.learn.datasets.base)' in msg
        return not tf_warning           
logger = logging.getLogger('tensorflow')
logger.addFilter(WarningFilter())

# if too many warnings from scikit-image 
import warnings
warnings.filterwarnings("ignore")

