# Keras extended Utilities for MLSP feature learning

The project contains tools for easier development with Keras/Tensorflow. 
It was developed for [AVA-MLSP](https://github.com/subpic/ava-mlsp) feature learning.
The code in [applications.py](/applications.py) and [model_helper.py](/model_helper.py) pertains more to [MLSP feature learning](https://github.com/subpic/ava-mlsp), 
wheras the remaining tools are general purpose.

## Overview

Some of the key components of each file:

**`generic.py`**:

* `H5Helper`: Manage named data sets in HDF5 files, for us in Keras generators.
* `pretty`: Pretty-print dictionary type objects.
* `ShortNameBuilder`: Utility for building short (file) names that contain multiple parameters.

**`image_utils.py`**:

* `ImageAugmenter`: Create custom image augmentation functions for training Keras models.
* `read_image`, `read_image_batch`: utility functions for manipulating images.

**`model_helper.py`**:

* `ModelHelper`: Wrapper class that simplifies default usage of Keras for regression models.

**`applications.py`**:

* `model_inception_multigap`, `model_inceptionresnet_multigap`: Model definitions for extracting MLSP narrow features
* `model_inception_pooled`, `model_inceptionresnet_pooled`: Model definitions for extracting MLSP wide features

**`generators.py`**:

* `DataGeneratorDisk`, `DataGeneratorHDF5`: Keras generators for on-disk images, and HDF5 stored features/images

You can find more information in the Python Docstrings.
