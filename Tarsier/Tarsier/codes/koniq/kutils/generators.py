import numpy as np
import pandas as pd
import multiprocessing as mp
from munch import Munch

import keras
from .image_utils import *
from .generic import *
from .tensor_ops import *

# GENERATORS

class DataGeneratorDisk(keras.utils.Sequence):
    """
    Generates data for training Keras models
    - inherits from keras.utils.Sequence
    - reads images from disk and applies `process_fn` to each
    - `process_fn` needs to ensure that processed images are of the same size
    - on __getitem__() returns an ND-array containing `batch_size` images

    ARGUMENTS
    ids (pandas.dataframe): table containing image names, and output variables
    data_path  (string):    path of image folder
    batch_size (int):       how many images to read at a time
    shuffle (bool):         randomized reading order
    process_fn (function):  function applied to each image as it is read
    deterministic (None, int):  random seed for shuffling order
    inputs (tuple of strings):  column names from `ids` containing image names
    outputs (tuple of strings): column names from `ids`
    verbose (bool):             logging verbosity
    fixed_batches (bool):       only return full batches, ignore the last incomplete batch if needed
    process_args (None, dict):  dictionary of arguments to pass to `process_fn`
    """
    def __init__(self, ids, data_path, **args):
        params_defa = Munch(ids           = ids,      data_path = data_path,
                            batch_size    = 32,       shuffle = True,
                            input_shape   = (224, 224, 3), process_fn = None,
                            deterministic = None,     inputs=('image_name',),
                            outputs       = ('MOS',), verbose = False,
                            fixed_batches = False,    process_args = None)
        check_keys_exist(args, params_defa)
        params = updated_dict(params_defa, **args)  # update only existing
        params.inputs    = force_tuple(params.inputs)
        params.outputs   = force_tuple(params.outputs)
        params.deterministic = {True: 42, False: None}.\
                               get(params.deterministic,
                                   params.deterministic)
        params.process_args = params.process_args or {}
        self.__dict__.update(**params)  # set all as self.<param>

        self.on_epoch_end()  # initialize indexes

    def __len__(self):
        """Denotes the number of batches per epoch accounting for `fixed_batches`"""
        round_op = np.floor if self.fixed_batches else np.ceil
        return int(round_op(len(self.ids)*1. / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes_batch = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        ids_batch = self.ids.iloc[indexes_batch].reset_index(drop=True)
        return self._data_generation(ids_batch)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _data_generation(self, ids_batch):
        """Generates image-stack + outputs containing batch_size samples"""
        np.random.seed(self.deterministic)

        # return values from `outputs` columns
        outputs = self.outputs
        if outputs is not None:
            if (isinstance(outputs, (list, tuple)) and
                isinstance(outputs[0], (list, tuple))):
                y = [np.array(ids_batch.loc[:,o]) for o in outputs]
            else:
                y = np.array(ids_batch.loc[:,outputs])
        else:
            y = None

        # build array from reading images in `inputs` columns
        X_list = []
        for input_name in self.inputs:
            data = []
            # read the data from disk into a list
            for row in ids_batch[input_name]:
                fname = os.path.join(self.data_path, row)
                im = read_image(fname)
                data.append(im)

            # if needed, process each image, and add to X_list (inputs list)
            if self.process_fn not in [None, False]:
                for args in self.process_args.get(input_name, [{}]):
                    data_new = None
                    for i in range(len(data)):
                        data_i = self.process_fn(data[i], **args)
                        if data_new is None:
                            data_new = np.zeros((len(data),)+data_i.shape,
                                                dtype=np.float32)
                        data_new[i, ...] = data_i
                    X_list.append(data_new)
            else:
                data_new = None
                for i in range(len(data)):
                    if data_new is None:
                        data_new = np.zeros((len(data),)+data[i].shape,
                                            dtype=np.float32)
                    data_new[i, ...] = data[i]
                X_list.append(data_new)

        np.random.seed(None)
        return (X_list[0], y) if (len(X_list) == 1) else (X_list, y)


class DataGeneratorHDF5(DataGeneratorDisk):
    """
    Generates data for training Keras models
    - similar to the `DataGeneratorDisk`, but reads data instances from an HDF5 file e.g. images, features
    - inherits from `DataGeneratorDisk`, a child of keras.utils.Sequence
    - applies `process_fn` to each data instance
    - `process_fn` needs to ensure a fixed size for processed data instances
    - on __getitem__() returns an ND-array containing `batch_size` data instances

    ARGUMENTS
    ids (pandas.dataframe): table containing data instance names, and output variables
    data_path  (string):    path of HDF5 file
    batch_size (int):       how many instances to read at a time
    shuffle (bool):         randomized reading order
    process_fn (function):  function applied to each data instance as it is read
    deterministic (None, int): random seed for shuffling order
    inputs (strings tuple):    column names from `ids` containing data instance names, read from `data_path`
    inputs_df (strings tuple): column names from `ids`, returns values from the DataFrame itself
    outputs (strings tuple):   column names from `ids`, returns values from the DataFrame itself
    verbose (bool):            logging verbosity
    fixed_batches (bool):      only return full batches, ignore the last incomplete batch if needed
    process_args (dict):       dictionary of arguments to pass to `process_fn`
    group_names (strings tuple): read only from specified groups, or from any if `group_names` is None
                                 `group_names` are randomly sampled from meta-groups
                                 i.e. when group_names = [[group_names_1], [group_names_2]]
    random_group (bool):         read inputs from a random group for every data instance
    """
    def __init__(self, ids, data_path, **args):
        params_defa = Munch(ids         = ids,   data_path     = data_path, deterministic = False,
                            batch_size  = 32,    shuffle       = True,      inputs        = ('image_name',),
                            inputs_df   = None,  outputs       = ('MOS',),  memory_mapped = False,
                            verbose     = False, fixed_batches = False,     random_group  = False,
                            process_fn  = None,  process_args  = None,      group_names   = None,
                            input_shape = None)

        check_keys_exist(args, params_defa)
        params = updated_dict(params_defa, **args) # update only existing
        params.inputs    = force_tuple(params.inputs)
        params.inputs_df = force_tuple(params.inputs_df)
        params.outputs   = force_tuple(params.outputs)
        params.process_args  = params.process_args or {}
        params.group_names   = params.group_names or [None]
        params.deterministic = {True: 42, False: None}.\
                                get(params.deterministic,
                                    params.deterministic)
        self.__dict__.update(**params)  # set all as self.<param>

        if self.verbose:
            print ('Initialized DataGeneratorHDF5')
        self.on_epoch_end()  # initialize indexes

    def _data_generation(self, ids_batch):
        """Generates data containing batch_size samples"""
        params = self
        np.random.seed(params.deterministic)

        outputs = params.outputs
        if (isinstance(outputs, (list, tuple)) and
            isinstance(outputs[0], (list, tuple))):
            y = [np.array(ids_batch.loc[:, o]) for o in outputs]
        else:
            y = np.array(ids_batch.loc[:, outputs])

        X_list = []
        if params.inputs_df is not None:
            X_list.append(np.array(ids_batch.loc[:, params.inputs_df]))

        with H5Helper(params.data_path, file_mode='r',
                      memory_mapped=params.memory_mapped) as h:
            # group_names are randomly sampled from meta-groups 
            # i.e. when group_names = [[group_names1], [group_names2]]
            group_names = params.group_names
            if isinstance(group_names[0], (list, tuple)):
                idx = np.random.randint(0, len(group_names))
                group_names = group_names[idx]

            # get data for each input and add it to X_list
            for group_name in group_names:
                for input_name in params.inputs:
                    # get data
                    names = ids_batch.loc[:,input_name]
                    if params.random_group:
                        data = h.read_data_random_group(names)
                    elif group_name is None:
                        data = h.read_data(names)
                    else:
                        data = h.read_data(names, group_names=[group_name])[0]
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)

                    # add to X_list
                    if params.process_fn not in [None, False]:
                        for args in params.process_args.get(input_name,[{}]):
                            data_new = None
                            for i in range(len(data)):
                                data_i = params.process_fn(data[i,...], **args)
                                if data_new is None:
                                    data_new = np.zeros((len(data),)+data_i.shape,
                                                        dtype=np.float32)
                                data_new[i,...] = data_i
                            X_list.append(data_new)
                    else:
                        X_list.append(data)

        np.random.seed(None)
        return (X_list[0], y) if (len(X_list) == 1) else (X_list, y)


class DataGeneratorDataFrame(DataGeneratorDisk):
    """
    Generates data for training Keras models
    - similar to the `DataGeneratorDisk`, returned values are taken from the ids DataFrame itself
    - inherits from `DataGeneratorDisk`, a child of keras.utils.Sequence
    - minimal interface compared to the other generators

    ARGUMENTS
    ids (pandas.dataframe): table containing the input and output variables
    batch_size (int):       how many rows to read at a time
    shuffle (bool):         randomized reading order
    deterministic (None, int): random seed for shuffling order
    inputs (strings tuple):    column names from `ids` returns values from the DataFrame itself
    outputs (strings tuple):   column names from `ids`, returns values from the DataFrame itself
    verbose (bool):            logging verbosity
    fixed_batches (bool):      only return full batches, ignore the last incomplete batch if needed
    """
    def __init__(self, ids, inputs, outputs, verbose=False,
                 batch_size=1024, shuffle=True, deterministic=False, **kwargs):
        self.ids     = ids
        self.inputs  = force_tuple(inputs)
        self.outputs = force_tuple(outputs)
        self.shuffle = shuffle
        self.verbose = verbose
        self.batch_size    = batch_size
        self.fixed_batches = False
        self.deterministic = deterministic
        self.on_epoch_end()  # initialize indexes

    def _data_generation(self, ids_batch):
        """Generates data containing batch_size samples"""
        X = np.array(ids_batch.loc[:, self.inputs])
        y = np.array(ids_batch.loc[:, self.outputs])
        return (X, y)
