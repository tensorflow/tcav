"""Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from tcav.activation_generator import DiscreteActivationGeneratorBase
from tcav.tcav_examples.discrete.kdd99_model import encode_variables, kBytesIndices, kFloatIndices, kIntIndices
import numpy as np
import tensorflow as tf


class KDD99DiscreteActivationGenerator(DiscreteActivationGeneratorBase):
  """ Activation generator for the KDD99 dataset.

  This uses the KDD99 dataset from sklearn. It is automatically loaded when we
  try to train models, so no data downloading is required.

  You can read more about it here:
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html

  To see it in action, please check kdd99_discrete_example.ipynb
  """

  def __init__(self, model, source_dir, acts_dir, max_examples):
    self.source_dir = source_dir
    super(KDD99DiscreteActivationGenerator,
          self).__init__(model, source_dir, acts_dir, max_examples)

  def load_data(self, concept):
    """ Reads csv files into a numpy.ndarray containing income data

    For this case, we create directories follow the following structure
    source_dir >>
                  concept >>
                            >> concept.csv

    The concepts will then be extracted into a numpy ndarray.

    Args:
      concept: string, This method takes names of a concept folder is an input.
        They should be located in source_dir

    Returns:
      texts: A numpy array, where each subarray contains one row of the dataset

    """
    concept_folder = os.path.join(self.source_dir, concept)
    concept_file = os.path.join(concept_folder, concept + '.csv')
    with tf.io.gfile.GFile(concept_file, 'r') as f:
      texts = [
          l.strip().split(',') for l in f.readlines()[:self.max_examples + 1]
      ]
    texts = np.array(texts, dtype='O')
    texts = texts[1:]  # remove headers
    texts = texts[:, :-1]  # remove labels
    texts = self._convert_types(texts)  # Assign proper data types
    return texts

  def transform_data(self, data):
    """ Encodes categorical columns and returns them as a numpy array

     We first encode our categorical variables, so that they can be parsed by
     the model. Finally,
     we return the data in the form of a numpy array.

    Args:
      data: numpy.ndarray, parsed from load_data

    Returns:
      encoded_data: numpy.ndarray. Categorical variables are encoded.


    """
    encoded_data = encode_variables(data)
    return encoded_data

  def _convert_types(self, texts):
    """ When read from .csv, all variables are parsed as string.

    This function assigns the proper types

    Args:
      texts: numpy.ndarray. Contains data parsed from dataset.

    Returns:
      texts: numpy.ndarray. Returns data with the proper types assigned

    """
    texts[:, kBytesIndices] = texts[:, kBytesIndices].astype(str)
    texts[:, kFloatIndices] = texts[:, kFloatIndices].astype(np.float32)
    texts[:, kIntIndices] = texts[:, kIntIndices].astype(np.int)
    return texts
