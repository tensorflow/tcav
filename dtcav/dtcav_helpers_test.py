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

"""Test for dtcav_helpers.py"""


import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import googletest

from dtcav import ConceptDiscovery
import dtcav_helpers

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


class TestConceptDiscovery(ConceptDiscovery):

  def __init__(self, model, target_class, random_concept, bottlenecks, sess,
               source_dir, activation_dir, cav_dir):
    self.model = model
    self.sess = sess
    self.target_class = target_class
    self.num_random_exp = 2
    if isinstance(bottlenecks, str):
      bottlenecks = [bottlenecks]
    self.bottlenecks = bottlenecks
    self.source_dir = source_dir
    self.activation_dir = activation_dir
    self.cav_dir = cav_dir
    self.channel_mean = True
    self.random_concept = random_concept
    self.image_shape = model.get_image_shape()[:2]
    self.max_imgs = 20
    self.num_discovery_imgs = 20
    self.min_imgs = 10
    self.num_workers = 20

  def load_cav_direction(self, c, r, bn, directory=None):

    return np.random.random((1, 100))


class DtcavHelpersTest(googletest.TestCase):

  def setUp(self):

    target_class = 'zebra'
    bottlenecks = ['bn1', 'bn2']
    source_dir = '/cns/is-d/home/amiratag/imagenet_train/'
    source_dir = None
    cav_dir = None
    activation_dir = None
    random_concept = 'random500_0'
    mymodel = DtcavTestModel()
    sess = tf.Session()
    self.mycd = TestConceptDiscovery(mymodel, target_class, random_concept,
                                     bottlenecks, sess, source_dir,
                                     activation_dir, cav_dir)
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

  def testFlatProfile(self):

    profile = dtcav_helpers.flat_profile(self.mycd,
                                         np.random.random((10, 299, 299, 3)))
    self.assertEqual(profile.shape, np.random.random((10, 2)).shape)

  def testCrossVal(self):

    lm, _ = dtcav_helpers.cross_val(
        np.random.random((10, 5)), np.random.random((10, 5)),
        ['logistic', 'sgd'])
    self.assertEqual(
        lm.predict(np.random.random((10, 5))).shape,
        np.random.random(10).shape)

  def testBinaryDataset(self):

    x, y = dtcav_helpers.binary_dataset(
        np.random.random((10, 5)), np.random.random((10, 5)))
    self.assertEqual(x.shape[0], y.shape[0])
    self.assertAlmostEqual(np.mean(y), 0.5)

  def testCosineSimilarity(self):

    a = np.random.random(100)
    b = np.random.random(100)
    c = a - np.sum(a * b) / (np.linalg.norm(b)**2) * b
    self.assertAlmostEqual(dtcav_helpers.cosine_similarity(a, a), 1.)
    self.assertAlmostEqual(dtcav_helpers.cosine_similarity(b, c), 0.)
    self.assertEqual(dtcav_helpers.cosine_similarity(a, np.zeros(100)), 0.)

  def testSimilarity(self):

    sim = dtcav_helpers.similarity(self.mycd, 2)
    self.assertEqual(set(sim.keys()), set(self.mycd.bottlenecks))
    for bn in self.mycd.bottlenecks:
      for c1 in self.mycd.dic[bn]['concepts']:
        for c2 in self.mycd.dic[bn]['concepts']:
          self.assertIn((c1, c2), sim[bn].keys())
          self.assertAlmostEqual(
              np.mean(sim[bn][(c1, c2)]), np.mean(sim[bn][(c2, c1)]))


if __name__ == '__main__':
  googletest.main()
