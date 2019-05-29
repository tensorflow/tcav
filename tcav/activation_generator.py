"""
Copyright 2018 Google LLC

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
  def get_model():
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
    if self.acts_dir and not tf.gfile.Exists(self.acts_dir):
      tf.gfile.MakeDirs(self.acts_dir)

    for concept in concepts:
      if concept not in acts:
        acts[concept] = {}
      for bottleneck_name in bottleneck_names:
        acts_path = os.path.join(self.acts_dir, 'acts_{}_{}'.format(
            concept, bottleneck_name)) if self.acts_dir else None
        if acts_path and tf.gfile.Exists(acts_path):
          with tf.gfile.Open(acts_path, 'rb') as f:
            acts[concept][bottleneck_name] = np.load(f).squeeze()
            tf.logging.info('Loaded {} shape {}'.format(
                acts_path, acts[concept][bottleneck_name].shape))
        else:
          acts[concept][bottleneck_name] = self.get_activations_for_concept(
              concept, bottleneck_name)
          if acts_path:
            tf.logging.info('{} does not exist, Making one...'.format(
                acts_path))
            with tf.gfile.Open(acts_path, 'w') as f:
              np.save(f, acts[concept][bottleneck_name], allow_pickle=False)
    return acts


class ImageActivationGenerator(ActivationGeneratorBase):
  """Activation generator for a basic image model"""

  def __init__(self, model, source_dir, acts_dir, max_examples=10):
    self.source_dir = source_dir
    super(ImageActivationGenerator, self).__init__(
        model, acts_dir, max_examples)

  def get_examples_for_concept(self, concept):
    concept_dir = os.path.join(self.source_dir, concept)
    img_paths = [os.path.join(concept_dir, d)
                 for d in tf.gfile.ListDirectory(concept_dir)]
    imgs = self.load_images_from_files(img_paths, self.max_examples,
                                       shape=self.model.get_image_shape()[:2])
    return imgs

  def load_image_from_file(self, filename, shape):
    """Given a filename, try to open the file. If failed, return None.

    Args:
      filename: location of the image file
      shape: the shape of the image file to be scaled

    Returns:
      the image if succeeds, None if fails.

    Rasies:
      exception if the image was not the right shape.
    """
    if not tf.gfile.Exists(filename):
      tf.logging.error('Cannot find file: {}'.format(filename))
      return None
    try:
      # ensure image has no transparency channel
      img = np.array(PIL.Image.open(tf.gfile.Open(filename, 'rb')).convert(
          'RGB').resize(shape, PIL.Image.BILINEAR))
      # Normalize pixel values to between 0 and 1.
      img = np.float32(img) / 255.0
      if not (len(img.shape) == 3 and img.shape[2] == 3):
        return None
      else:
        return img

    except Exception as e:
      tf.logging.info(e)
      return None
    return img

  def load_images_from_files(self, filenames, max_imgs=500,
                             do_shuffle=True, run_parallel=True,
                             shape=(299, 299),
                             num_workers=100):
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
        raise ValueError('You must have more than 1 image in each class to run TCAV.')
    else:
      for filename in filenames:
        img = self.load_image_from_file(filename, shape)
        if img is not None:
          imgs.append(img)
        if len(imgs) <= 1:
          raise ValueError('You must have more than 1 image in each class to run TCAV.')
        elif len(imgs) >= max_imgs:
          break

    return np.array(imgs)
