
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

"""Tests dcav.py"""


import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import googletest

from dtcav import ConceptDiscovery


class DtcavTestModel(object):
  """A mock model of model class for TcavTest class.

  """

  def __init__(self):
    self.model_name = 'test_model'

  def get_gradient(self, act, class_id, bottleneck):
    return act

  def reshape_activations(self, acts):
    return acts

  def run_imgs(self, imgs, bottleneck_name):
    return np.random.random((len(imgs), 100))

  def get_predictions(self, images):
    """Returning one label.

    Args:
      images: dummy input images

    Returns:
      zeros
    """
    return np.zeros(len(images))

  def get_image_shape(self):

    return (299, 299, 3)

  def label_to_id(self, target_class):
    """Returning one label.

    Args:
      target_class: dummy target_class

    Returns:
      zero.
    """
    return 0


class DtcavTest(googletest.TestCase):

  def setUp(self):

    target_class = 'zebra'
    bottlenecks = ['bn1', 'bn2']
    source_dir = '/cns/is-d/home/beenkim/imgnet_subset/'
    cav_dir = None
    activation_dir = None
    random_concept = 'random500_0'
    mymodel = DtcavTestModel()
    sess = tf.Session()
    self.mycd = ConceptDiscovery(
        mymodel,
        target_class,
        random_concept,
        bottlenecks,
        sess,
        source_dir,
        activation_dir,
        cav_dir,
        max_imgs=5,
        min_imgs=2,
        num_discovery_imgs=5,
        num_workers=0)
    self.mycd.dic = {
        'bn1': {
            'concept1': {
                'images': np.random.random((10, 299, 299, 3)),
                'patches': np.random.random((10, 299, 299, 3)),
                'image_numbers': np.zeros(10)
            },
            'concepts': ['concept1']
        },
        'bn2': {
            'concept1': {
                'images': np.random.random((10, 299, 299, 3)),
                'patches': np.random.random((10, 299, 299, 3)),
                'image_numbers': np.zeros(10)
            },
            'concepts': ['concept1']
        }
    }

  def test_create_patches(self):

    self.mycd.create_patches(param_dict={'n_segments': [10]},
                              discovery_images=np.random.random((50, 299, 299,
                                                                3)))
    self.assertEqual(len(self.mycd.dataset), len(self.mycd.patches))
    self.assertEqual(len(self.mycd.dataset), len(self.mycd.image_numbers))
    self.mycd.num_workers = 0
    self.mycd.create_patches(param_dict={'n_segments': [10]})
    self.assertGreater(len(self.mycd.dataset), 0)
    self.assertEqual(len(self.mycd.dataset), len(self.mycd.patches))
    self.assertEqual(len(self.mycd.dataset), len(self.mycd.image_numbers))
    self.mycd.create_patches(param_dict={'n_segments': [10]})
    self.mycd.num_workers = 25
    self.mycd.create_patches(param_dict={'n_segments': [10]})
    self.assertGreater(len(self.mycd.dataset), 0)
    self.assertEqual(len(self.mycd.dataset), len(self.mycd.patches))
    self.assertEqual(len(self.mycd.dataset), len(self.mycd.image_numbers))

  def test_discover_concepts(self):

    self.mycd.create_patches(param_dict={'n_segments': [10]},
                              discovery_images=np.random.random((50, 299, 299,
                                                                3)))
    self.mycd.discover_concepts(
        activations={'bn1': np.random.random((len(self.mycd.dataset), 100))},
        param_dicts={'n_clusters': 10})
    self.assertEqual(set(self.mycd.dic.keys()), set(self.mycd.bottlenecks))


if __name__ == '__main__':
  googletest.main()
