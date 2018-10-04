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

""" collection of various helper functions for TCAV"""

from multiprocessing import dummy as multiprocessing
import os.path
import re
import numpy as np
import PIL.Image
import scipy.stats as stats
import tensorflow as tf
import json

# General helper functions


def flatten(nested_list):
  """Flatten a nested list."""
  return [item for a_list in nested_list for item in a_list]


def make_key(a_dict, key):
  """Make the key in a_dict, if not exists already."""
  if key not in a_dict:
    a_dict[key] = {}


def save_np_array(array, path):
  """Save an array as numpy array (loading time is better than pickling."""
  with tf.gfile.Open(path, 'w') as f:
    np.save(f, array, allow_pickle=False)


def read_np_array(path):
  """Read a saved numpy array and return."""
  with tf.gfile.Open(path) as f:
    data = np.load(f)
  return data


def read_file(path):
  """Read a file in path ."""
  with tf.gfile.Open(path, 'r') as f:
    data = f.read()
  return data


def write_file(data, path, mode='w'):
  """Wrtie data to path to cns."""
  with tf.gfile.Open(path, mode) as f:
    if mode == 'a':
      f.write('\n')
    f.write(data)



# Helper functions to deal with images


def load_image_from_file(filename, shape):
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
    img = np.array(PIL.Image.open(tf.gfile.Open(filename)).resize(
        shape, PIL.Image.BILINEAR))
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


def load_images_from_files(filenames, max_imgs=500, return_filenames=False,
                           do_shuffle=True, run_parallel=True,
                           shape=(299, 299),
                           num_workers=100):
  """Return image arrays from filenames.

  Args:
    filenames: locations of image files.
    max_imgs: maximum number of images from filenames.
    return_filenames: return the succeeded filenames or not
    do_shuffle: before getting max_imgs files, shuffle the names or not
    run_parallel: get images in parallel or not
    shape: desired shape of the image
    num_workers: number of workers in parallelization.

  Returns:
    image arrays and succeeded filenames if return_filenames=True.

  """
  imgs = []
  # First shuffle a copy of the filenames.
  filenames = filenames[:]
  if do_shuffle:
    np.random.shuffle(filenames)
  if return_filenames:
    final_filenames = []

  if run_parallel:
    pool = multiprocessing.Pool(num_workers)
    imgs = pool.map(lambda filename: load_image_from_file(filename, shape),
                    filenames[:max_imgs])
    if return_filenames:
      final_filenames = [f for i, f in enumerate(filenames[:max_imgs])
                         if imgs[i] is not None]
    imgs = [img for img in imgs if img is not None]
  else:
    for filename in filenames:
      img = load_image_from_file(filename, shape)
      if img is not None:
        imgs.append(img)
        if return_filenames:
          final_filenames.append(filename)
      if len(imgs) >= max_imgs:
        break

  if return_filenames:
    return np.array(imgs), final_filenames
  else:
    return np.array(imgs)

""" high level overview.
get_acts_from_images: run images on a model and return activations.
get_imgs_and_acts_save: loads images from image path and
                         calls get_acts_from_images to get images
                         and save them.
"""


def get_acts_from_images(imgs, model, bottleneck_name):
  """Run images in the model to get the activations.

  Args:
    imgs: a list of images
    model: a model instance
    bottleneck_name: bottleneck name to get the activation from

  Returns:
    numpy array of activations.
  """
  img_acts = model.run_imgs(imgs, bottleneck_name)
  return model.reshape_activations(img_acts)


def get_imgs_and_acts_save(model, bottleneck_name, img_paths, acts_path,
                           img_shape, max_images=500):
  """Get images from files, process acts and saves.

  Args:
    model: a model instance
    bottleneck_name: name of the bottleneck that activations are from
    img_paths: where image lives
    acts_path: where to store activations
    img_shape: shape of the image.
    max_images: maximum number of images to save to acts_path

  Returns:
    success or not.
  """
  imgs = load_images_from_files(img_paths, max_images, shape=img_shape)

  tf.logging.info('got %s imgs' % (len(imgs)))
  acts = get_acts_from_images(imgs, model, bottleneck_name)
  tf.logging.info('Writing acts to {}'.format(acts_path))
  with tf.gfile.Open(acts_path, 'w') as f:
    np.save(f, acts, allow_pickle=False)
  del acts
  del imgs
  return True

def process_and_load_activations(model, bottleneck_names, concepts,
                                 source_dir,
                                 acts_dir, acts=None, max_images=500):
  """If activations doesn't alreayd exists, make one, and returns them.

     For each concept, check images exists. Then for each bottleneck, make
     activations if there isn't one alreayd. If there is one, load it from
     activation path.

  Args:
    model: a model instance.
    bottleneck_names: list of bottlenecks we want to process
    concepts: list of concepts of interest
    source_dir: directory containing the concept images
    acts_dir: activations dir to save
    acts: a dictionary of activations if there were any pre-loaded
    max_images: maximum number of images for each concept

  Returns:
    acts: dictionary of activations
  """
  if not tf.gfile.Exists(acts_dir):
    tf.gfile.MakeDirs(acts_dir)
  if acts is None:
    acts = {}

  for concept in concepts:
    concept_dir = os.path.join(source_dir, concept)
    if concept not in acts:
      acts[concept] = {}
    # Check if the image directory exists.
    if not tf.gfile.Exists(concept_dir):
      tf.logging.fatal('Image directory does not exist: {}'.format(concept_dir))
      raise ValueError('Image directory does not exist: {}'.format(concept_dir))

    for bottleneck_name in bottleneck_names:
      acts_path = os.path.join(acts_dir, 'acts_{}_{}'.format(
          concept, bottleneck_name))
      if not tf.gfile.Exists(acts_path):
        tf.logging.info('{} does not exist, Making one...'.format(acts_path))
        img_paths = [os.path.join(concept_dir, d)
                     for d in tf.gfile.ListDirectory(concept_dir)]
        get_imgs_and_acts_save(model, bottleneck_name, img_paths, acts_path,
                               model.get_image_shape()[:2],
                               max_images=max_images)

      if bottleneck_name not in acts[concept].keys():
        with tf.gfile.Open(acts_path) as f:
          acts[concept][bottleneck_name] = np.load(f).squeeze()
          tf.logging.info('Loaded {} shape {}'.format(
              acts_path,
              acts[concept][bottleneck_name].shape))
      else:
        tf.logging.info('%s, %s already exists in acts. Skipping...' % (
            concept, bottleneck_name))

  return acts
