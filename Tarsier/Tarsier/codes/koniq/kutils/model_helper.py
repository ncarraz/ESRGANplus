import os, sys, keras, numbers, glob, shutil
import multiprocessing as mp, pandas as pd, numpy as np
from pprint import pprint
from munch import Munch

from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, EarlyStopping
from keras import optimizers
from keras.models import Model, load_model
from keras.utils import multi_gpu_model

from .generators import *
from .generic import *


class ModelHelper:
    """
    Wrapper class that simplifies default usage of Keras for training, testing, logging,
    storage of models by removing the need for some repetitive code segments (boilerplate).

    Encapsulates a Keras model and its configuration, generators for feeding the model
    during training, validation and testing, logging to TensorBoard, saving/loading
    model configurations to disk, and activations from within a model.

    When operating on a model, generators can be instantiated by the ModelHelper or pre-defined
    by the user and passed to the train/predict methods. Generators, rely on DataFrame objects
    (usually named `ids`, as they contain the IDs of data rows) for extracting data instances
    for all operations (train/validation/test).
    """
    def __init__(self, model, root_name, ids, 
                 gen_params={}, verbose=False, **params):
        """
        :param model: Keras model, compilation will be done at runtime.
        :param root_name: base name of the model, extended with
                          configuration parameters when saved
        :param ids: DataFrame table, used by generators
        :param gen_params: dict for parameters for the generator
                           defaults (shuffle = True, process_fn = False,
                                     deterministic = False, verbose = False)
        :param verbose: verbosity

        OTHER PARAMS
        lr   = 1e-4,               # learning rate
        loss = "MSE",              # loss function
        loss_weights   = None,     # loss weights, if multiple losses are used
        metrics        = ["mae"],  #
        class_weights  = None,     # class weights for unbalanced classes
        multiproc      = True,     # multi-processing params
        workers        = 5,        #
        max_queue_size = 10,       #
        monitor_metric      = 'val_mean_absolute_error',  # monitoring params
        monitor_mode        = 'min',                      #
        early_stop_patience = 20,                         #
        checkpoint_period   = 1,                          #
        save_best_only = True,                            #
        optimizer      = optimizers.Adam(),  # optimizer object
        write_graph    = False,              # TensorBoard params
        write_images   = False,              #
        histogram_freq = 0,                  #
        logs_root      = '../logs/',         # TensorBoard logs
        models_root    = '../models/',       # saved models path
        features_root  = '../features/',     # saved features path (by `save_activations`)
        gen_class      = None                # generator class
                                             # inferred from self.gen_params.data_path
        """
        self.model = model
        self.ids = ids
        self.verbose = verbose
        self.model_name = ShortNameBuilder(prefix=root_name+'/')
        self.model_cpu = None

        self.gen_params = Munch(shuffle       = True,  process_fn = False,
                                deterministic = False, verbose    = verbose)
        self.params = Munch(lr   = 1e-4,               # learning rate
                            loss = "MSE",              # loss function
                            loss_weights   = None,     # loss weights, if multiple losses are used
                            metrics        = ["mae"],  #
                            class_weights  = None,     # class weights for unbalanced classes

                            multiproc      = True,     # multi-processing params
                            workers        = 5,        #
                            max_queue_size = 10,       #

                            monitor_metric      = 'val_mean_absolute_error',  # monitoring params
                            monitor_mode        = 'min',                      #
                            early_stop_patience = 20,                         #
                            checkpoint_period   = 1,                          #
                            save_best_only = True,                            #
                            optimizer      = optimizers.Adam(),  # optimizer object, its parameters
                                                                 # can changed during runtime

                            write_graph    = False,              # TensorBoard params
                            write_images   = False,              #
                            histogram_freq = 0,                  #

                            logs_root      = '../logs/',         # TensorBoard logs
                            models_root    = '../models/',       # saved models path
                            features_root  = '../features/',     # saved features path (by `save_activations`)
                            gen_class      = None                # generator class
                                                                 # inferred from self.gen_params.data_path
                            )

        for key in params.keys():
            if key not in self.params.keys():
                raise Exception('Undefined parameter:' + key)

        self.gen_params.update(gen_params)        
        self.params = updated_dict(self.params, **params)

        # infer default generator class to use 
        # if params is not set yet
        if self.params.gen_class is None:
            # if has 'data_path' attribute
            if getattr(self.gen_params, 'data_path', None) is not None: 
                if self.gen_params.data_path[-3:] == '.h5':
                    self.params.gen_class = DataGeneratorHDF5
                else:
                    self.params.gen_class = DataGeneratorDisk
            else:
                self.params.gen_class = DataGeneratorDataFrame
        
        self.set_model_name()
    
    def set_model_name(self):
        """Update model name based on parameters in self. gen_params, params and model"""
        h = self
        tostr = lambda x: str(x) if x is not None else '?'
        format_size = lambda x: '[{}]'.format(','.join(map(tostr, x)))
        loss2str = lambda x: (x if isinstance(x, str) else x.__name__)[:8]

        loss = h.params.loss
        if not isinstance(loss, dict):
            loss_str = loss2str(loss)
        else:
            loss_str = '[%s]' % ','.join(map(loss2str, loss.values()))
                
        i = '{}{}'.format(len(h.model.inputs),
                           format_size(h.gen_params.input_shape))
        o = '{}{}'.format(len(h.model.outputs),
                           format_size(h.model.outputs[0].shape[1:].as_list()))
        name = dict(i   = i,
                    o   = o,
                    l   = loss_str,
                    bsz = h.gen_params.batch_size)
        
        self.model_name.update(name)
        return self.model_name
    
    def _callbacks(self):
        """Setup callbacks"""
        p = self.params
        log_dir = os.path.join(self.params.logs_root, self.model_name())
        if p.histogram_freq:
            valid_gen = self.make_generator(self.ids[self.ids.set=='validation'], 
                                            deterministic=True, fixed_batches=True)
            tb_callback = TensorBoardWrapper(valid_gen, log_dir=log_dir, 
                                      write_images=p.write_images, 
                                      histogram_freq=p.histogram_freq, 
                                      write_graph=p.write_graph)
        else:
            tb_callback = TensorBoard(log_dir=log_dir,
                                      write_graph=p.write_graph, 
                                      histogram_freq=0, 
                                      write_images=p.write_images)
        
        tb_callback.set_model(self.model)
        best_model_path = os.path.join(self.params.models_root, 
                                       self.model_name() + '_best_weights.h5')
        make_dirs(best_model_path)
        checkpointer = ModelCheckpoint(filepath = best_model_path, verbose=0,
                                       monitor  = p.monitor_metric, 
                                       mode     = p.monitor_mode, 
                                       period   = p.checkpoint_period,
                                       save_best_only    = p.save_best_only,
                                       save_weights_only = True)
        earlystop = EarlyStopping(monitor=p.monitor_metric, 
                                  patience=p.early_stop_patience, 
                                  mode=p.monitor_mode)
        return [tb_callback, earlystop, checkpointer]

    def _updated_gen_params(self, **kwargs):
        # private method
        params = self.gen_params.copy()
        params.update(kwargs)
        return params
    
    def make_generator(self, ids, **kwargs):
        """
        Create a generator of `self.params.gen_class` using new `ids`.

        :param ids: DataFrame table
        :param kwargs: updated parameters of the generator
        :return: Sequence (generator)
        """
        # an alternative to using the private method:
        # params = updated_dict(self.gen_params,
        #                       only_existing=False,
        #                       **kwargs)
        params = self._updated_gen_params(**kwargs)
        return self.params.gen_class(ids, **params)

    def test_generator(self, input_idx=0):
        """
        Basic utility to run the generator for one or more data instances in `ids`.
        Useful for testing the default generator functionality.

        :param input_idx: scalar or 2-tuple of indices
        :return: (generated_batch, generator_instance)
        """
        if not isinstance(input_idx, (list, tuple)):
            ids_gen = self.ids[input_idx:input_idx+1]
        else:
            ids_gen = self.ids[input_idx[0]:input_idx[1]]

        gen = self.make_generator(ids_gen)
        x = gen[0]  # generated data
        print_sizes(x)
        return x, gen

    def set_multi_gpu(self, gpus=None):
        """
        Enable multi-GPU processing.
        Creates a copy of the CPU model and calls `multi_gpu_model`.

        :param gpus: number of GPUs to use, defaults to all.
        """
        self.model_cpu = self.model
        self.model = multi_gpu_model(self.model, gpus=gpus)        

    def train(self, train_gen=None, valid_gen=None, lr=1e-4, epochs=1):
        """
        Run training iterations on existing model.
        Initializes `train_gen` and `valid_gen` if not defined.

        :param train_gen: train generator
        :param valid_gen: validation generator
        :param lr:        learning rate
        :param epochs:    number of epochs
        :return:          training history from self.model.fit_generator()
        """
        ids = self.ids
        params = self.params
                   
        print ('\nTraining model:', self.model_name())
        
        if train_gen is None:
            train_gen = self.make_generator(ids[ids.set == 'training'])
        if valid_gen is None:
            valid_gen = self.make_generator(ids[ids.set == 'validation'],
                                            deterministic=True)

        if lr: self.params.lr = lr
        self.params.optimizer = update_config(self.params.optimizer,
                                              lr=self.params.lr)
        
        self.model.compile(optimizer=self.params.optimizer, 
                           loss=params.loss, loss_weights=params.loss_weights, 
                           metrics=params.metrics)

        if self.verbose:
            print ('\nGenerator parameters:')
            print ('---------------------')
            pretty(self.gen_params)
            print ('\nMain parameters:')
            print ('----------------')
            pretty(self.params)
            print ('\nLearning')

        history = self.model.fit_generator(train_gen, epochs = epochs,
                                           steps_per_epoch   = len(train_gen),
                                           validation_data   = valid_gen, 
                                           validation_steps  = len(valid_gen),
                                           workers           = params.workers, 
                                           callbacks         = self._callbacks(),
                                           max_queue_size    = params.max_queue_size,
                                           class_weight      = self.params.class_weights,
                                           use_multiprocessing = params.multiproc)
        return history
    
    def clean_outputs(self):
        """
        Delete training logs or models created by the current helper configuration.
        Identifies the logs by the configuration paths and `self.model_name`.
        Asks for user confirmation before deleting any files.
        """
        log_dir = os.path.join(self.params.logs_root, self.model_name())
        model_path = os.path.join(self.params.models_root, self.model_name()) + '*.h5'
        model_path = model_path.replace('[', '[[]')
        model_files = glob.glob(model_path)

        if os.path.exists(log_dir):
            print ('Found logs:')
            print (log_dir)
            if raw_confirm('Delete?'):
                print ('Deleting', log_dir)
                shutil.rmtree(log_dir)
        else:
            print ('(No logs found)')

        if model_files:
            print ('Found model(s):')
            print (model_files)
            if raw_confirm('Delete?'):
                for mf in model_files: 
                    print ('Deleting', mf)
                    os.unlink(mf)
        else:
            print ('(No models found)')
        
    def predict(self, test_gen=None, output_layer=None, 
                repeats=1, batch_size=None, remodel=True):
        """
        Predict on `test_gen`.

        :param test_gen: generator used for prediction
        :param output_layer: layer at which activations are computed (defaults to output)
        :param repeats: how many times the prediction is repeated (with different augmentation)
        :param batch_size: size of each batch
        :param remodel: if true: change model such that new output is `output_layer`
        :return: if repeats == 1, np.ndarray, otherwise a list of np.ndarray
        """
        if not test_gen:
            params_test = self._updated_gen_params(shuffle=False, 
                                                   fixed_batches=False)
            if batch_size: params_test.batch_size = batch_size
            test_gen = self.params.gen_class(self.ids[self.ids.set == 'test'],
                                             **params_test)
        if output_layer is not None and remodel:
                # get last partial-matching layer
                layer_name = [l.name for l in self.model.layers 
                              if output_layer in l.name][-1]
                output_layer = self.model.get_layer(layer_name)
                print ('Output of layer:', output_layer.name)
                if isinstance(output_layer, Model):
                    outputs = output_layer.outputs[0]
                else:
                    outputs = output_layer.output
                print ('Output tensor:', outputs)
                model = Model(inputs  = self.model.input, 
                              outputs = outputs) 
        else: 
            model = self.model

        preds = []
        for i in range(repeats):
            y_pred = model.predict_generator(test_gen, workers=1, verbose=0,
                                             use_multiprocessing=False)
            if not remodel and output_layer is not None:
                y_pred = dict(zip(model.output_names, y_pred))[output_layer]
            preds.append(np.squeeze(y_pred))
        return preds[0] if repeats == 1 else preds

    def set_trainable(self, index):
        """
        Convenience method to set trainable layers.
        Layers up to `index` are frozen; the remaining
        after `index` are set as trainable.
        """
        for layer in self.model.layers[:index]:
            layer.trainable = False
        for layer in self.model.layers[index:]:
            layer.trainable = True

    def load_model(self, model_name='', best=True, 
                   from_weights=True, by_name=False):
        """
        Load model from file.

        :param model_name: new model name, otherwise self.model_name()
        :param best: load the best model, or otherwise final model
        :param from_weights: from weights, or from full saved model
        :param by_name: load layers by name
        :return: true if model was loaded successfully, otherwise false
        """
        model_name = model_name or self.model_name()
        model_file_name = (model_name + ('_best' if best else '_final') + 
                          ('_weights' if from_weights else '') + '.h5')
        model_path = os.path.join(self.params.models_root, model_file_name)
        if not os.path.exists(model_path):
            print ('Model NOT loaded:', model_file_name, 'does not exist')
            return False
        else:
            if from_weights:
                self.model.load_weights(model_path, by_name=by_name)
                print ('Model weights loaded:', model_file_name)
            else:
                self.model = load_model(model_path)
                print ('Model loaded:', model_file_name)
            return True

    def save_model(self, weights_only=False, model=None, name_extras=''):
        """
        Save model to HDF5 file.

        :param weights_only: save only weights,
                             or full model otherwise
        :param model: specify a particular model instance,
                      otherwise self.model is saved
        :param name_extras: append this to model_name
        """
        model = model or self.model
        print ('Saving model', model.name, 'spanning'),\
              len(self.model.layers), 'layers'
        if weights_only:
            model_file = self.model_name() + name_extras + '_final_weights.h5'
            model.save_weights(os.path.join(self.params.models_root, model_file))
            print ('Model weights saved:', model_file)
        else:
            model_file = self.model_name() + name_extras + '_final.h5'
            model.compile(optimizer=self.params.optimizer, loss="mean_absolute_error")
            model.save(os.path.join(self.params.models_root, model_file))
            print ('Model saved:', model_file)

    def save_activations(self, output_layer=None, file_path=None, ids=None,
                         groups=1, verbose=False, over_write=False, name_suffix='',
                         save_as_type=np.float32):
        """
        Save activations from a particular `output_layer` to an HDF5 file.

        :param output_layer: if not None, layer name, otherwise use the model output
        :param file_path:    HDF5 file path
        :param ids:          data entries to compute the activations for, defaults to `self.ids`
        :param groups:       a number denoting the number of augmentation repetitions, or list of group names
        :param verbose:      verbosity
        :param over_write:   overwrite HDF5 file
        :param name_suffix:  append suffix to name of file (before ext)
        :param save_as_type: save as a different data type, defaults to np.float32
        """
        if ids is None:
            ids = self.ids
        if isinstance(groups, numbers.Number):
            groups_count = groups
            groups_list = map(str, range(groups))
        else:
            groups_count = len(groups)
            groups_list  = map(str, groups)

        if file_path is None:         
            short_name = self.model_name.subset(['i', 'o'])
            short_name(r = groups_count,
                       l = (output_layer or 'final'))
            if name_suffix:
                name_suffix = '_'+name_suffix
            file_path = os.path.join(self.params.features_root, 
                                     str(short_name) + name_suffix + '.h5')
            make_dirs(file_path)

        params = self._updated_gen_params(shuffle       = False, 
                                          verbose       = verbose,
                                          fixed_batches = False)
        if verbose:
            print ('Saving activations for layer:', (output_layer or 'final'))
            print ('File:', file_path)

        data_gen = self.make_generator(ids, **params)

        for group_name in groups_list:
            activ = self.predict(data_gen, output_layer = output_layer).\
                                                          astype(save_as_type)
            if len(activ.shape)==1:
                activ = np.expand_dims(activ, 0)
            with H5Helper(file_path, 
                          over_write = over_write, 
                          verbose    = verbose) as h:
                if groups == 1:
                    h.write_data(activ, ids[data_gen.inputs[0]])
                else:
                    h.write_data([activ], ids[data_gen.inputs[0]], 
                                 group_names=[group_name])
            del activ


class TensorBoardWrapper(TensorBoard):
    """Sets the self.validation_data property for use with TensorBoard callback."""

    def __init__(self, valid_gen, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.valid_gen = valid_gen  # The validation generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property.
        X, y = self.valid_gen[0]
        X = np.float32(X);
        y = np.float32(y)
        sample_weights = np.ones(X.shape[0], dtype=np.float32)
        self.validation_data = [X, y, sample_weights, np.float32(0.0)]
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)


# MISC helper functions

def get_layer_index(model, name):
    """Get index of layer by name"""
    for idx, layer in enumerate(model.layers):
        if layer.name == name:
            return idx

def get_activations(im, model, layer):
    """Get activations from `layer` in `model` using `K.function`"""
    if len(im.shape) < 4:
        im = np.expand_dims(im, 0)
    inputs = [K.learning_phase()] + model.inputs
    fn = K.function(inputs, [layer.output])
    act = fn([0] + [im])
    act = np.squeeze(act)
    return act
