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
import os
import pickle
import shutil
import numpy as np
import tensorflow as tf
from sklearn import linear_model
from six.moves import range
from tcav.cav import CAV, get_or_train_cav
from tensorflow.python.platform import flags
from tensorflow.python.platform import googletest

FLAGS = flags.FLAGS
flags.DEFINE_string(name='tcav_test_tmpdir', default='/tmp',
                    help='Temporary directory for test files')

class CavTest(googletest.TestCase):

  def setUp(self):
    """Makes a cav instance and writes it to tmp direcotry.

    The cav instance uses preset values.
    """
    self.hparams = {
      'model_type':'linear', 'alpha':.01, 'max_iter':1000, 'tol':1e-3}
    self.concepts = ['concept1', 'concept2']
    self.bottleneck = 'bottleneck'
    self.accuracies = {'concept1': 0.8, 'concept2': 0.5, 'overall': 0.65}
    self.cav_vecs = [[1, 2, 3], [4, 5, 6]]

    self.test_subdirectory = os.path.join(FLAGS.tcav_test_tmpdir, 'test')
    self.cav_dir = self.test_subdirectory
    self.cav_file_name = CAV.cav_key(self.concepts, self.bottleneck,
                                         self.hparams['model_type'],
                                         self.hparams['alpha']) + '.pkl'
    self.save_path = os.path.join(self.cav_dir, self.cav_file_name)
    self.cav = CAV(self.concepts, self.bottleneck, self.hparams)
    # pretend that it was trained and cavs are stored
    self.cav.cavs = np.array(self.cav_vecs)
    shape = (1, 3)
    self.acts = {
        concept: {
            self.bottleneck: np.tile(i * np.ones(shape), (4, 1))
        } for i, concept in enumerate(self.concepts)
    }

    if os.path.exists(self.cav_dir):
      shutil.rmtree(self.cav_dir)
    os.mkdir(self.cav_dir)
    with tf.io.gfile.GFile(self.save_path, 'w') as pkl_file:
      pickle.dump({
          'concepts': self.concepts,
          'bottleneck': self.bottleneck,
          'hparams': self.hparams,
          'accuracies': self.accuracies,
          'cavs': self.cav_vecs,
          'saved_path': self.save_path
      }, pkl_file)

  def test_default_hparams(self):
    hparam = CAV.default_hparams()
    self.assertEqual(hparam['alpha'], 0.01)
    self.assertEqual(hparam['model_type'], 'linear')

  def test_load_cav(self):
    """Load up the cav file written in setup function and check values.
    """
    cav_instance = CAV.load_cav(self.save_path)
    self.assertEqual(cav_instance.concepts, self.concepts)
    self.assertEqual(cav_instance.cavs, self.cav_vecs)

  def test_cav_key(self):
    self.assertEqual(
        self.cav.cav_key(self.concepts, self.bottleneck,
                         self.hparams['model_type'], self.hparams['alpha']),
        '-'.join(self.concepts) + '-' + self.bottleneck + '-' +
        self.hparams['model_type'] + '-' + str(self.hparams['alpha']))

  def test_check_cav_exists(self):
    exists = self.cav.check_cav_exists(self.cav_dir, self.concepts,
                                       self.bottleneck, self.hparams)
    self.assertTrue(exists)

  def test__create_cav_training_set(self):
    x, labels, labels2text = self.cav._create_cav_training_set(
        self.concepts, self.bottleneck, self.acts)
    # check values of some elements.
    self.assertEqual(x[0][0], 0)
    self.assertEqual(x[5][0], 1)
    self.assertEqual(labels[0], 0)
    self.assertEqual(labels[5], 1)
    self.assertEqual(labels2text[0], 'concept1')

  def test_perturb_act(self):
    perturbed = self.cav.perturb_act(
        np.array([1., 0, 1.]), 'concept1', operation=np.add, alpha=1.0)
    self.assertEqual(2., perturbed[0])
    self.assertEqual(2., perturbed[1])
    self.assertEqual(4., perturbed[2])

  def test_get_key(self):
    self.assertEqual(
        CAV.cav_key(self.concepts, self.bottleneck, self.hparams['model_type'],
                        self.hparams['alpha']),
        '-'.join([str(c) for c in self.concepts]) + '-' + self.bottleneck + '-'
        + self.hparams['model_type'] + '-' + str(self.hparams['alpha']))

  def test_get_direction(self):
    idx_concept1 = self.cav.concepts.index('concept1')
    cav_directly_from_member = self.cav.cavs[idx_concept1]
    cav_via_get_direction = self.cav.get_direction('concept1')
    for i in range(len(cav_directly_from_member)):
      self.assertEqual(cav_directly_from_member[i], cav_via_get_direction[i])

  def test_train(self):
    self.cav.train({c: self.acts[c] for c in self.concepts})
    # check values of some elements.
    # the two coefficients of the classifier must be negative.
    self.assertLess(self.cav.cavs[0][0] * self.cav.cavs[1][0], 0)

  def test__train_lm(self):
    lm = linear_model.SGDClassifier(alpha=self.hparams['alpha'])
    acc = self.cav._train_lm(lm, np.array([[0], [0], [0], [1], [1], [1]]),
                             np.array([0, 0, 0, 1, 1, 1]), {
                                 0: 0,
                                 1: 1
                             })
    # the given data is so easy it should get this almost perfect.
    self.assertGreater(acc[0], 0.99)
    self.assertGreater(acc[1], 0.99)

  def test_get_or_train_cav_save_test(self):
    cav_instance = get_or_train_cav(
        self.concepts,
        self.bottleneck,
        self.acts,
        cav_dir=self.cav_dir,
        cav_hparams=self.hparams)
    # check values of some elements.
    self.assertEqual(cav_instance.cavs[0][0], self.cav_vecs[0][0])
    self.assertEqual(cav_instance.cavs[1][2], self.cav_vecs[1][2])


if __name__ == '__main__':
  googletest.main()
