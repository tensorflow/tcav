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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABCMeta
from abc import abstractmethod
from multiprocessing import dummy as multiprocessing
import os
import os.path
import numpy as np
import PIL.Image
import six
import tensorflow as tf

class ActivationGeneratorInterface(six.with_metaclass(ABCMeta, object)):
  """Interface for an activation generator for a model"""

  @abstractmethod
  def process_and_load_activations(self, bottleneck_names, concepts):
    pass

  @abstractmethod
  def get_model(self):
    pass


class ActivationGeneratorBase(ActivationGeneratorInterface):
  """Basic abstract activation generator for a model"""

  def __init__(self, model, acts_dir, max_examples=500):
    self.model = model
    self.acts_dir = acts_dir
    self.max_examples = max_examples

  def get_model(self):
    return self.model

  @abstractmethod
  def get_examples_for_concept(self, concept):
    pass

  def get_activations_for_concept(self, concept, bottleneck):
    examples = self.get_examples_for_concept(concept)
    return self.get_activations_for_examples(examples, bottleneck)

  def get_activations_for_examples(self, examples, bottleneck):
    acts = self.model.run_examples(examples, bottleneck)
    return self.model.reshape_activations(acts).squeeze()

  def process_and_load_activations(self, bottleneck_names, concepts):
    acts = {}
    if self.acts_dir and not tf.io.gfile.exists(self.acts_dir):
      tf.io.gfile.makedirs(self.acts_dir)

    for concept in concepts:
      if concept not in acts:
        acts[concept] = {}
      for bottleneck_name in bottleneck_names:
        acts_path = os.path.join(self.acts_dir, 'acts_{}_{}'.format(
            concept, bottleneck_name)) if self.acts_dir else None
        if acts_path and tf.io.gfile.exists(acts_path):
          with tf.io.gfile.GFile(acts_path, 'rb') as f:
            acts[concept][bottleneck_name] = np.load(
                f, allow_pickle=True).squeeze()
            tf.compat.v1.logging.info('Loaded {} shape {}'.format(
                acts_path, acts[concept][bottleneck_name].shape))
        else:
          acts[concept][bottleneck_name] = self.get_activations_for_concept(
              concept, bottleneck_name)
          if acts_path:
            tf.compat.v1.logging.info(
                '{} does not exist, Making one...'.format(acts_path))
            tf.io.gfile.mkdir(os.path.dirname(acts_path))
            with tf.io.gfile.GFile(acts_path, 'w') as f:
              np.save(f, acts[concept][bottleneck_name], allow_pickle=False)
    return acts


class ImageActivationGenerator(ActivationGeneratorBase):
  """Activation generator for a basic image model"""

  def __init__(self,
               model,
               source_dir,
               acts_dir,
               max_examples=10,
               normalize_image=True):
    """Initialize ImageActivationGenerator class."

    Args:
      normalize_image: A boolean indicating whether image pixels should be
        normalized to between 0 and 1.
    """
    self.source_dir = source_dir
    self.normalize_image = normalize_image
    super(ImageActivationGenerator, self).__init__(model, acts_dir,
                                                   max_examples)

  def get_examples_for_concept(self, concept):
    concept_dir = os.path.join(self.source_dir, concept)
    img_paths = [
        os.path.join(concept_dir, d) for d in tf.io.gfile.listdir(concept_dir)
    ]
    imgs = self.load_images_from_files(
        img_paths, self.max_examples, shape=self.model.get_image_shape()[:2])
    return imgs

  def load_image_from_file(self, filename, shape):
    """Given a filename, try to open the file.

    If failed, return None.

    Args:
      filename: location of the image file
      shape: the shape of the image file to be scaled

    Returns:
      the image if succeeds, None if fails.

    Rasies:
      exception if the image was not the right shape.
    """
    if not tf.io.gfile.exists(filename):
      tf.compat.v1.logging.error('Cannot find file: {}'.format(filename))
      return None
    try:
      # ensure image has no transparency channel
      img = np.array(
          PIL.Image.open(tf.io.gfile.GFile(filename,
                                           'rb')).convert('RGB').resize(
                                               shape, PIL.Image.BILINEAR),
          dtype=np.float32)
      if self.normalize_image:
        # Normalize pixel values to between 0 and 1.
        img = img / 255.0
      if not (len(img.shape) == 3 and img.shape[2] == 3):
        return None
      else:
        return img

    except Exception as e:
      tf.compat.v1.logging.info(e)
      return None
    return img

  def load_images_from_files(self,
                             filenames,
                             max_imgs=500,
                             do_shuffle=True,
                             run_parallel=True,
                             shape=(299, 299),
                             num_workers=50):
    """Return image arrays from filenames.

    Args:
      filenames: locations of image files.
      max_imgs: maximum number of images from filenames.
      do_shuffle: before getting max_imgs files, shuffle the names or not
      run_parallel: get images in parallel or not
      shape: desired shape of the image
      num_workers: number of workers in parallelization.

    Returns:
      image arrays

    """
    imgs = []
    # First shuffle a copy of the filenames.
    filenames = filenames[:]
    if do_shuffle:
      np.random.shuffle(filenames)

    if run_parallel:
      pool = multiprocessing.Pool(num_workers)
      imgs = pool.map(
          lambda filename: self.load_image_from_file(filename, shape),
          filenames[:max_imgs])
      imgs = [img for img in imgs if img is not None]
      if len(imgs) <= 1:
        raise ValueError(
            'You must have more than 1 image in each class to run TCAV.')
    else:
      for filename in filenames:
        img = self.load_image_from_file(filename, shape)
        if img is not None:
          imgs.append(img)
        if len(imgs) >= max_imgs:
          break
      if len(imgs) <= 1:
        raise ValueError(
            'You must have more than 1 image in each class to run TCAV.')

    return np.array(imgs)


"""Discrete activation generators"""


class DiscreteActivationGeneratorBase(ActivationGeneratorBase):
  """ Base class for discrete data. """

  def __init__(self, model, source_dir, acts_dir, max_examples):
    self.source_dir = source_dir
    super(DiscreteActivationGeneratorBase, self).__init__(
        model=model, acts_dir=acts_dir, max_examples=max_examples)

  def get_examples_for_concept(self, concept):
    """Extracts examples for a concept and transforms them to the desired format.

    Args:
      concept: Name of a concept. Names for the folder containing data for that
        concept. Path is fetched based on the source_dir used by the activation
        generator

    Returns:
          data_parsed:
            Examples from the data folder. Format is according to the
            load_data() and transform_data() functions.

    """
    data = self.load_data(concept)
    data_parsed = self.transform_data(data)
    return data_parsed

  @abstractmethod
  def load_data(self, concept):
    """Extracts data from a source and returns it in a user specified format.

    It takes in as input the name for a concept folder that lies inside of the
    source_dir.

    Args
      concept: name of the concept (e.g., name of a folder that contains concept
      examples in source_dir). They should be located in source_dir.
    Returns:
      data parsed
    """
    # Needs to be implemented
    raise NotImplementedError()

  def transform_data(self, data):
    """Transforms data into a format that can be directly processed by the model.

    Once the data is parsed, use this function to transform the data into the
    format your model needs. Some example transformations can be:
      - Converting to proto type
      - Encoding categorical features
      - Tokenizing an input sequence

    Args:
      data: The previously extracted data from load_data

    Returns:
      Transformed data, in the desired dormat
    """
    # By default, regurgitates input data if not implemented
    return data
