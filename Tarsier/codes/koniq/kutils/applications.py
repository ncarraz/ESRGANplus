from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.utils import plot_model
from keras.layers import *

from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.nasnet import NASNetMobile

source_module = {
                 InceptionV3:       keras.applications.inception_v3,
                 DenseNet201:       keras.applications.densenet,
                 ResNet50:          keras.applications.resnet50,       
                 InceptionResNetV2: keras.applications.inception_resnet_v2,
                 VGG16:             keras.applications.vgg16,
                 NASNetMobile:      keras.applications.nasnet
                }

from .model_helper import *

# correspondences between CNN model name and pre-processing function
process_input = {
                 InceptionV3:       keras.applications.inception_v3.preprocess_input,
                 DenseNet201:       keras.applications.densenet.preprocess_input,
                 ResNet50:          keras.applications.resnet50.preprocess_input,
                 InceptionResNetV2: keras.applications.inception_resnet_v2.preprocess_input,
                 VGG16:             keras.applications.vgg16.preprocess_input,
                 NASNetMobile:      keras.applications.nasnet.preprocess_input
                }

def fc_layers(input_layer,
              name               = 'pred',
              fc_sizes           = [2048, 1024, 256, 1],
              dropout_rates      = [0.25, 0.25, 0.5, 0],
              batch_norm         = False,
              l2_norm_inputs     = False,
              kernel_regularizer = None,
              out_activation     = 'linear'):
    """
    Add a standard fully-connected (fc) chain of layers (functional Keras interface)
    with dropouts on top of an input layer. Optionally batch normalize, add regularizers
    and an output activation.

    e.g. default would look like dense(2048) > dropout

    :param input_layer: input layer to the chain
    :param name: prefix to each layer in the chain
    :param fc_sizes: list of number of neurons in each fc-layer
    :param dropout_rates: list of dropout rates for each fc-layer
    :param batch_norm: 0 (False) = no batch normalization (BN),
                       1 = do BN for all, 2 = do for all except the last
    :param l2_norm_inputs: normalize the `input_layer` with L2_norm
    :param kernel_regularizer: optional regularizer for each fc-layer
    :param out_activation: activation added to the last fc-layer
    :return: output layer of the chain
    """
    x = input_layer
    if l2_norm_inputs:
        x = Lambda(lambda x: K.tf.nn.l2_normalize(x, 1))(input_layer)

    assert len(fc_sizes) == len(dropout_rates),\
           'Each FC layer should have a corresponding dropout rate'

    for i in range(len(fc_sizes)):
        if i < len(fc_sizes)-1:
            act = 'relu'
            layer_type = 'fc%d' % i
        else:
            act  = out_activation
            layer_type = 'out'
        x = Dense(fc_sizes[i], activation=act, 
                  name='%s_%s' % (name, layer_type),
                  kernel_regularizer = kernel_regularizer,
                  kernel_initializer='he_normal')(x)
        if batch_norm == 1 or (batch_norm == 2 and i<len(fc_sizes)-1): 
            x = BatchNormalization(name='%s_bn%d' % (name, i))(x)
        if dropout_rates[i] > 0:
            x = Dropout(dropout_rates[i], 
                        name='%s_do%d' % (name, i))(x)            
    return x

def conv2d_bn(x, filters, num_row, num_col, padding='same',
              strides=(1, 1), name=None):
    """
    Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.

    Source: InceptionV3 Keras code
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(filters, (num_row, num_col),
               strides=strides, padding=padding,
               use_bias=False, name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def inception_block(x, size=768, name=''):
    channel_axis = 3
    
    branch1x1 = conv2d_bn(x, size, 1, 1, name=name+'branch_1x1')

    branch3x3 = conv2d_bn(x, size, 1, 1, name=name+'3x3_1x1')
    branch3x3_1 = conv2d_bn(branch3x3, size/2, 1, 3, name=name+'3x3_1x3')
    branch3x3_2 = conv2d_bn(branch3x3, size/2, 3, 1, name=name+'3x3_3x1')
    branch3x3 = concatenate(
        [branch3x3_1, branch3x3_2],
        axis=channel_axis,
        name=name+'branch_3x3')

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), 
                                   padding='same', 
                                   name=name+'avg_pool_2d')(x)
    branch_pool = conv2d_bn(branch_pool, size, 1, 1, 
                            name=name+'branch_pool')
    
    y = concatenate(
        [branch1x1, branch3x3, branch_pool],
        axis=channel_axis,
        name=name+'mixed_final')
    return y

def model_inception_multigap(input_shape=(224, 224, 3), return_sizes=False,
                             indexes=range(11), name = ''):
    """
    Build InceptionV3 multi-GAP model, that extracts narrow MLSP features.
    Relies on `get_inception_gaps`.

    :param input_shape: shape of the input images
    :param return_sizes: return the sizes of each layer: (model, gap_sizes)
    :param indexes: indices to use from the usual GAPs
    :param name: name of the model
    :return: model or (model, gap_sizes)
    """
    print ('Loading InceptionV3 multi-gap with input_shape:', input_shape)

    model_base = InceptionV3(weights     = 'imagenet', 
                             include_top = False, 
                             input_shape = input_shape)
    print ('Creating multi-GAP model')
    
    gap_name = name + '_' if name else ''

    feature_layers = [model_base.get_layer('mixed%d' % i) 
                      for i in indexes]
    gaps = [GlobalAveragePooling2D(name=gap_name+"gap%d" % i)(l.output)
            for i, l in zip(indexes, feature_layers)]
    concat_gaps = Concatenate(name=gap_name+'concat_gaps')(gaps)

    model = Model(inputs  = model_base.input,
                  outputs = concat_gaps)
    if name:
        model.name = name
    
    if return_sizes:
        gap_sizes = [np.int32(g.get_shape()[1]) for g in gaps]
        return (model, gap_sizes)
    else:
        return model


def model_inceptionresnet_multigap(input_shape=(224, 224, 3),
                                   return_sizes=False):
    """
    Build InceptionResNetV2 multi-GAP model, that extracts narrow MLSP features.

    :param input_shape: shape of the input images
    :param return_sizes: return the sizes of each layer: (model, gap_sizes)
    :return: model or (model, gap_sizes)
    """
    print ('Loading InceptionResNetV2 multi-gap with input_shape:', input_shape)

    model_base = InceptionResNetV2(weights='imagenet',
                                   include_top=False,
                                   input_shape=input_shape)
    print ('Creating multi-GAP model')
    
    feature_layers = [l for l in model_base.layers if 'mixed' in l.name]
    gaps = [GlobalAveragePooling2D(name="gap%d" % i)(l.output)
            for i, l in enumerate(feature_layers)]
    concat_gaps = Concatenate(name='concatenated_gaps')(gaps)

    model = Model(inputs=model_base.input, outputs=concat_gaps)

    if return_sizes:
        gap_sizes = [np.int32(g.get_shape()[1]) for g in gaps]
        return (model, gap_sizes)
    else:
        return model
    
def model_inception_pooled(input_shape=(None, None, 3), indexes=range(11),
                           pool_size=(5, 5), name='', return_sizes=False):
    """
    Returns the wide MLSP features, spatially pooled, from InceptionV3.
    Similar to `model_inception_multigap`.

    :param input_shape: shape of the input images
    :param indexes: indices to use from the usual GAPs
    :param pool_size: spatial extend of the MLSP features
    :param name: name of the model
    :param return_sizes: return the sizes of each layer: (model, pool_sizes)
    :return: model or (model, pool_sizes)
    """
    print ('Loading InceptionV3 multi-pooled with input_shape:', input_shape)
    model_base = InceptionV3(weights     = 'imagenet', 
                             include_top = False, 
                             input_shape = input_shape)
    print ('Creating multi-pooled model')
    
    ImageResizer = Lambda(lambda x: K.tf.image.resize_area(x, pool_size),
                          name='feature_resizer')

    feature_layers = [model_base.get_layer('mixed%d' % i) for i in indexes]
    pools = [ImageResizer(l.output) for l in feature_layers]
    conc_pools = Concatenate(name='conc_pools', axis=3)(pools)

    model = Model(inputs  = model_base.input, 
                  outputs = conc_pools)
    if name: model.name = name

    if return_sizes:
        pool_sizes = [[np.int32(x) for x in f.get_shape()[1:]] for f in pools]
        return model, pool_sizes
    else:
        return model
    
def model_inceptionresnet_pooled(input_shape=(None, None, 3), pool_size=(5, 5),
                                 name='', return_sizes=False):
    """
    Returns the wide MLSP features, spatially pooled, from InceptionResNetV2.

    :param input_shape: shape of the input images
    :param pool_size: spatial extend of the MLSP features
    :param name: name of the model
    :param return_sizes: return the sizes of each layer: (model, pool_sizes)
    :return: model or (model, pool_sizes)
    """
    
    print ('Loading InceptionResNetV2 multi-pooled with input_shape:', input_shape)
    model_base = InceptionResNetV2(weights     = 'imagenet', 
                                   include_top = False, 
                                   input_shape = input_shape)
    print ('Creating multi-pooled model')
    
    ImageResizer = Lambda(lambda x: K.tf.image.resize_area(x, pool_size),
                          name='feature_resizer') 

    feature_layers = [l for l in model_base.layers if 'mixed' in l.name]
    pools = [ImageResizer(l.output) for l in feature_layers]
    conc_pools = Concatenate(name='conc_pools', axis=3)(pools)

    model = Model(inputs  = model_base.input, 
                  outputs = conc_pools)
    if name: model.name = name

    if return_sizes:
        pool_sizes = [[np.int32(x) for x in f.get_shape()[1:]] for f in pools]
        return model, pool_sizes
    else:
        return model    


# ------------------
# RATING model utils
# ------------------

def test_rating_model(helper, output_layer=None, test_set='test',
                      accuracy_thresh=None, groups=1, remodel=False,
                      ids=None, show_plot=True):
    """
    Test rating model performance. The output of the mode is assumed to be
    either a single score, or distribution of scores (can be a histogram).

    :param helper: ModelHelper object that contains the trained model
    :param output_layer: the rating layer, if more than out output exists
                         if output_layer is None, we assume there is a single
                         rating output
    :param test_set: name of the test set in the helper `ids`
                     allows to test model on validation, or training set
    :param accuracy_thresh: compute binary classification accuracy, assuming a
                            split of the rating scale at `accuracy_thresh`
                            i.e. LQ class is score <= accuracy_thresh
    :param groups: if a number: repetitions of the testing procedure,
                                after which the results are averaged
                   if list of strings: group names to repeat over,
                                       they are assumed to be different augmentations
    :param remodel: change structure of the model when changing `output_layer`, or not
    :param ids: optionally provide another set of data instances,
                replacing those in `helper.ids`
    :param show_plot: plot results vs ground-truth
    :return: (y_true, y_pred, SRCC, PLCC, ACC)
    """
    print ('Testing model')
    print ('Model outputs:', helper.model.output_names)
    if ids is None: ids = helper.ids
    ids_test = ids if test_set is None else ids[ids.set==test_set]

    if isinstance(groups, numbers.Number):
        test_gen = helper.make_generator(ids_test, 
                                         shuffle = False,
                                         fixed_batches = False)
        y_pred = helper.predict(test_gen, repeats=groups, remodel=remodel,
                                output_layer=output_layer)
        groups_list = range(groups)
    else:
        if isinstance(groups[0], (list, tuple, str)):
            groups_list = groups
        else:
            groups_list = map(str, groups)
        y_pred = []
        print ('Predicting on groups:')
        for group in groups_list:
            print (group,)
            test_gen = helper.make_generator(ids_test, shuffle=False,
                                             fixed_batches = False,
                                             random_group  = False,
                                             group_names   = force_list(group))
            y_pred.append(helper.predict(test_gen, repeats=1, remodel=remodel,
                                         output_layer=output_layer))
        print
    
    if isinstance(y_pred, list):
        y_pred = reduce(lambda x, y: (x+y), y_pred) / len(y_pred)

    if y_pred.ndim == 2:  # for distributions
        outputs = helper.gen_params.outputs
        y_pred = dist2mos(y_pred, scale=np.arange(1, len(outputs)+1))
        y_test = np.array(ids_test.loc[:, outputs])
        y_test = dist2mos(y_test, scale=np.arange(1, len(outputs)+1))
    else:                 # for MOS
        y_test = np.array(ids_test.loc[:,'MOS'])

    # in case the last batch was not used, and dataset size
    # is not a multiple of batch_size
    y_test = y_test[:len(y_pred)]
    
    SRCC_test = np.round(srocc(y_pred, y_test), 3)
    PLCC_test = np.round(plcc(y_pred, y_test), 3)
    print( )
    print ('Evaluated on', test_set + '-set' if test_set else 'all data')
    print ('SRCC/PLCC:', SRCC_test, PLCC_test)

    ACC_test = None
    if accuracy_thresh is not None:
        if not isinstance(accuracy_thresh, list):
            accuracy_thresh = [accuracy_thresh]*2
        # assume binary classification for scores the LQ class is MOS<=accuracy_thresh
        ACC_test = np.sum((y_test <= accuracy_thresh[0]) ==
                          (y_pred <= accuracy_thresh[1]), dtype=float) / len(y_test)
        print ('ACCURACY:', ACC_test)
        
    if show_plot:
        plt.plot(y_pred, y_test, '.', markersize=1)
        plt.xlabel('prediction')
        plt.ylabel('ground-truth')
        plt.show()
    return y_test, y_pred, SRCC_test, PLCC_test, ACC_test


def get_train_test_sets(ids, stratify_on='MOS', test_size=(0.2, 0.2),
                        save_path=None, show_histograms=False, 
                        stratify=False, random_state=None):
    """
    Devise a train/validation/test partition for a pd.DataFrame
    Adds a column 'set' to the input `ids` that identifies each row as one of:
    ['training', 'validation', 'test'] sets.

    The partition can be stratified based on a continuous variable,
    meaning that the variable is first quantized, and then a kind of
    'class-balancing' is performed based on the quantization.

    :param ids: pd.DataFrame
    :param stratify_on: column name from `ids` to stratify
                        (class balancing) the partitions on
    :param test_size: ratio (or number) of rows to assign to each of
                      test and validation sets respectively
                      e.g. (<validation size>, <test size>) or
                           (<validation ratio>, <test ratio>)
    :param save_path: optional save path for generated partitioned table
    :param show_histograms: show histograms of the distribution of the
                            stratification column
    :param stratify: do stratification
    :param random_state: initialize random state with a fixed value,
                         for reproducibility
    :return: modified DataTable
    """
    if not(isinstance(test_size, tuple) or
           isinstance(test_size, list)):
        test_size = (test_size, test_size)
    ids = ids.copy().reset_index(drop=True)
    idx = range(len(ids))
    if not stratify:
        strata = None
    else:
        strata = np.int32(mapmm(ids.loc[:, stratify_on], 
                                (0, stratify-1-1e-6)))
   
    idx_train_valid, idx_test = train_test_split(idx,
                                test_size=test_size[1],
                                random_state=random_state, 
                                stratify=strata)
    strata_valid = None if strata is None else strata[idx_train_valid]
    idx_train, idx_valid = train_test_split(idx_train_valid,
                           test_size=test_size[0], 
                           random_state=random_state,
                           stratify=strata_valid)  
        
    print ('Train size:', len(idx_train), 'Validation size:'),\
            len(idx_valid), 'Test size:', len(idx_test)
    
    ids.loc[idx_train, 'set'] = 'training'
    ids.loc[idx_valid, 'set'] = 'validation'
    ids.loc[idx_test,  'set'] = 'test'    
    if save_path is not None:
        ids.to_csv(save_path, index=False)
        
    if show_histograms:
        plt.hist(ids.loc[idx_train, stratify_on], density=True, 
                 facecolor='g', alpha=0.75, bins=100)
        plt.show()
        plt.hist(ids.loc[idx_valid, stratify_on], density=True, 
                 facecolor='b', alpha=0.75, bins=100)
        plt.show()
        plt.hist(ids.loc[idx_test, stratify_on], density=True, 
                 facecolor='r', alpha=0.75, bins=100)
        plt.show()

    return ids


# --------------------
# Potentially OBSOLETE
# --------------------

# A bit overly specific w.r.t. the model architecture
def get_model_imagenet(net_name, input_shape=None, plot=False, **kwargs):
    """Get ImageNet models"""

    if net_name == ResNet50:
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_shape=input_shape, **kwargs)
        feats = base_model.layers[-2]
    elif net_name == NASNetMobile:
        base_model = NASNetMobile(weights='imagenet',
                                  include_top=True,
                                  input_shape=input_shape, **kwargs)
        feats = base_model.layers[-3]
    elif net_name in source_module.keys():
        base_model = net_name(weights='imagenet', include_top=False,
                              input_shape=input_shape, **kwargs)
        feats = base_model.layers[-1]
    else:
        raise Exception('Unknown model ' + net_name.func_name)

    gap = GlobalAveragePooling2D(name="final_gap")(feats.output)
    model = Model(inputs=base_model.input, outputs=gap)

    return model, process_input[net_name]
