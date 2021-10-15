import numpy as np, pandas as pd
import multiprocessing as mp
import os, scipy, h5py
from munch import Munch


# Helps with the DataGeneratorHDF5

class H5Helper:
    """
    Read/Write named data sets from/to an HDF5 file.
    The structure of the data inside the HDF5 file is:
    'group_name/dataset_name' or 'dataset_name'

    Enables reading from random groups e.g. 'augmentation_type/file_name'
    such that the helper can be used during training Keras models.
    """
    def __init__(self, file_name, file_mode=None, 
                 memory_mapped=False, over_write=False, 
                 backing_store=False, verbose=False):
        """
        :param file_name: HDF5 file path
        :param file_mode: one of 'a','w','r','w+','r+'
        :param memory_mapped: enables memory mapped backing store
        :param over_write: over-write existing file
        :param backing_store: use another backing store
        :param verbose: verbosity
        """
        self.hf = None
        self.file_name = file_name
        self.verbose = verbose
        self.memory_mapped = memory_mapped
        self.backing_store = backing_store        
        self._lock = mp.Lock()
        _file_mode = file_mode or ('w' if over_write else 'a')
        with self._lock:
            if memory_mapped:
                # memory mapping via h5py built-ins
                self.hf = h5py.File(file_name, _file_mode, driver='core', 
                                    backing_store = backing_store)
            else:
                self.hf = h5py.File(file_name, _file_mode)

    def _write_datasets(self, writer, data, dataset_names):
        # internal to the class
        for i in range(len(data)):
            writer.create_dataset(dataset_names[i], data=data[i, ...])

    def write_data(self, data, dataset_names, group_names=None):
        """
        Write `data` to HDF5 file, using `dataset_names` for datasets,
        and optionally `group_names` for groups.

        :param data: if `group_names` is None: np.ndarray of N data instances of size N x [...]
                     else list of np.ndarray of N data instances of size N x [...] each
        :param dataset_names: list of strings
        :param group_names: None, or list of strings
        """
        with self._lock:
            hf = self.hf
            if group_names is None:
                assert not isinstance(data, list),\
                       'Data should be a numpy.ndarray when no groups are specified'
                self._write_datasets(hf, data, dataset_names)
            else:
                assert isinstance(data, list) and len(data) == len(group_names),\
                       'Each group name should correspond to a data list entry'
                for i, name in enumerate(group_names):
                    group = hf.require_group(name)
                    self._write_datasets(group, data[i], dataset_names)

    def _read_datasets(self, reader, dataset_names):
        # internal to the class
        name = dataset_names[0]
        data0 = reader[name][...]
        data = np.empty((len(dataset_names),) + data0.shape, 
                        dtype=data0.dtype)
        data[0,...] = data0
        for i in range(1, len(dataset_names)):
            data[i, ...] = reader[dataset_names[i]][...]
        return data
  
    def read_data(self, dataset_names, group_names=None):
        """
        Read `dataset_names` from HDF5 file, optionally using `group_names`.

        :param dataset_names: list of strings
        :param group_names: None, or list of strings
        :return: np.ndarray
        """
        with self._lock:
            hf = self.hf
            if group_names is None:
                return self._read_datasets(hf, dataset_names)
            else:
                return [self._read_datasets(hf[group_name], dataset_names)
                        for group_name in group_names]        

    def read_data_random_group(self, dataset_names):
        """
        Reads `dataset_names` each one from a random group.
        At least one group must exist.

        :param dataset_names: list of strings
        :return: np.ndarray
        """
        with self._lock:
            hf = self.hf
            group_names = np.array(self.group_names)
            idx = np.random.randint(0, len(group_names), len(dataset_names))
            names = ['{}/{}'.format(g, d) for g, d in zip(group_names[idx], dataset_names)]
            return self._read_datasets(hf, names)

    def summary(self, print_limit=100):
        """
        Prints a summary of the contents of the HDF5 file.
        Lists all groups and first `print_limit` datasets for each group.

        :param print_limit: number of datasets to list per group.
        """
        hf = self.hf
        keys = hf.keys()
        for i, group_name in enumerate(keys):
            if i > print_limit:
                print ('[...] first %d items from a total of %d' % (print_limit, len(keys)))
                break
            print (group_name, '\b/')
            group = hf[group_name]
            try:
                group_keys = group.keys()
                print (' '),
                for j, dataset_name in enumerate(group_keys):
                    if j > print_limit:
                        print ('[...] first %d from a total of %d' % (print_limit, len(group_keys)))
                        break
                    print (dataset_name),
            except: pass
        return ''

    @property
    def group_names(self):
        """
        :return: list of group names
        """
        return self.hf.keys()

    @property
    def dataset_names(self):
        """
        :return: list of dataset names 
                 (if groups are present, from the first group)
        """
        values = self.hf.values()
        if isinstance(values[0], h5py._hl.dataset.Dataset):
            return self.hf.keys()
        else:
            return values[0].keys()

    # enable 'with' statements
    def __del__(self):
        self.__exit__(None, None, None)

    def __enter__(self): 
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hf is not None:
            self.hf.flush()
            self.hf.close()
            del self.hf
            self.hf = None
                

def minmax(x):
    """
    Range of x.

    :param x: list or np.ndarray
    :return: (min, max)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x.min(), x.max()

def mapmm(x, new_range = (0, 1)):
    """
    Remap values in `x` to `new_range`.

    :param x: np.ndarray
    :param new_range: (min, max)
    :return: np.ndarray with values mapped to [new_range[0], new_range[1]]
    """
    mina, maxa = new_range
    if not type(x) == np.ndarray: 
        x = np.asfarray(x, dtype='float32')
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    minx, maxx = minmax(x)
    if minx < maxx:
        x = (x-minx)/(maxx-minx)*(maxa-mina)+mina
    return x

def plcc(x, y):
    """Pearson Linear Correlation Coefficient"""
    return scipy.stats.pearsonr(x, y)[0]

def srocc(xs, ys):
    """Spearman Rank Order Correlation Coefficient"""
    xranks = pd.Series(xs).rank()    
    yranks = pd.Series(ys).rank()    
    return plcc(xranks, yranks)

def dist2mos(x, scale=np.arange(1, 6)):
    """
    Find the MOS of a distribution of scores `x`, given a `scale`.
    e.g. x=[0,2,2], scale=[1,2,3] => MOS=2.5
    """
    x = x / np.reshape(np.sum(x*1., axis=1), (len(x), 1))
    return np.sum(x * scale, axis=1)
    
def force_tuple(x):
    """Make tuple out of `x` if not already a tuple or `x` is None"""
    if x is not None and not isinstance(x, tuple):
        return (x,)
    else:
        return x
    
def force_list(x):
    """Make list out of `x` if not already a list or `x` is None"""
    if x is not None and not isinstance(x, list):
        return [x]
    else:
        return x

def make_dirs(filename):
    """
    Create directory structure described by `filename`.
    :param filename: a valid system path
    """
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except: pass
        
def updated_dict(d, only_existing=True, **updates):
    """
    Update dictionary `d` with `updates`, optionally changing `only_existing` keys.

    :param d: dict
    :param only_existing: do not add new keys, update existing ones only
    :param updates: dict
    :return: updated dictionary
    """
    d = d.copy()
    if only_existing:
        common = {key: value for key, value in updates.items()
                  if key in d.keys()}     
        d.update(common)
    else:
        d.update(updates)
    return d

def chunks(l, n):
    """Yields successive `n`-sized chunks from list `l`."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def pretty(d, indent=0, key_sep=':', trim=True):
    """
    Pretty print dictionary, recursively.

    :param d: dict
    :param indent: indentation amount
    :param key_sep: separator printed between key and values
    :param trim: remove redundant white space from printed values
    """
    if indent == 0 and not isinstance(d, dict):
        d = d.__dict__
    max_key_len = 0
    keys = d.keys()
    if isinstance(keys, list) and len(keys)>0:
        max_key_len = max([len(str(k)) for k in keys])        
    for key, value in d.items():
        equal_offset = ' '*(max_key_len - len(str(key)))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            value_str = str(value).strip()
            if trim:
                value_str = ' '.join(value_str.split())
            if len(value_str) > 70:
                value_str = value_str[:70] + ' [...]' 

class ShortNameBuilder(Munch):
    """
    Utility for building short (file) names
    that contain multiple parameters.

    For internal use in ModelHelper.
    """
    def __init__(self, prefix='', sep=('', '_'), max_len=32,
                 **kwargs):
        self.__prefix  = prefix
        self.__sep     = sep
        self.__max_len = max_len
        super(ShortNameBuilder, self).__init__(**kwargs)

    def subset(self, selected_keys):
        subself = ShortNameBuilder(**self)
        for k in subself.keys():
            if k not in selected_keys and \
               '_ShortNameBuilder' not in k:
                del subself[k]
        return subself

    def __call__(self, subset=None, **kwargs):
        self.update(**kwargs)
        return str(self)
    
    def __str__(self):
        def combine(k, v):
            k = str(k)[:self.__max_len]
            v = str(v)[:self.__max_len]
            return k + self.__sep[0] + v        
        return self.__prefix + \
               self.__sep[1].join([combine(k, self[k])
                    for k in sorted(self.keys())
                    if '_ShortNameBuilder' not in k])
    
def check_keys_exist(new, old):
    """
    Check that keys in `new` dict existing in `old` dict.
    :param new: dict
    :param old: dict
    :return: exception if `new` keys don't existing in `old` ones

    Utility function used internally.
    """
    for key in new.keys():
        if key not in old.keys():
            raise Exception('Undefined parameter: "%s"' % key)
            
def print_sizes(x):
    """
    Recursively prints the shapes of elements in lists.
    """
    if isinstance(x, list) or isinstance(x, tuple):
        print ('[',)
        for _x_ in x:
            print_sizes(_x_)
            print (',',)
        print ('\b\b]',)
    elif hasattr(x, 'shape'):
        print (x.shape,)
    else: print (x,)
    return ''
        
def raw_confirm(message):
    """
    Ask for confirmation.
    :param message: message to show
    :return: true if confirmation given, false otherwise
    """
    print (message, '(y/[n])')
    confirmation = raw_input()
    if not confirmation:
        return False  # do not confirm by default
    else:
        return confirmation.lower()[0] == 'y'

def update_config(obj, **kwargs):
    """
    Update configuration of Keras `obj` e.g. layer, model.

    :param obj: object that has .get_config() and .from_config() methods.
    :param kwargs: dict of updates
    :return: updated object
    """
    cfg = obj.get_config()
    cfg.update(**kwargs)
    return obj.from_config(cfg)
