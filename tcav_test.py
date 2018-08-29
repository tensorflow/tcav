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

"""Tests for tcav.py"""

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import googletest
from cav import CAV
from tcav import TCAV


class TcavTest_model():
  """A mock model of model class for TcavTest class.
  """

  def __init__(self):
    self.model_name = 'test_model'

  def get_gradient(self, act, class_id, bottleneck):
    return act

  def label_to_id(self, target_class):
    """Returning one label.

    Args:
      target_class: dummy target_class

    Returns:
      zero.
    """
    return 0


class TcavTest(googletest.TestCase):

  def setUp(self):
    self.acts = np.array([[0, 1., 2.]])
    self.concepts = ['c1', 'c2']
    self.target = 't1'
    self.class_id = 0
    self.bottleneck = 'bn'
    self.cav_dir = None
    self.hparams = tf.contrib.training.HParams(model_type='linear', alpha=.01)
    self.cav = CAV(self.concepts,
                       self.bottleneck,
                       self.hparams)
    self.cav.cavs = [[1., 2., 3.,]]
    self.random_counterpart = 'random500_1'
    self.activation_generator = None
    self.mymodel = TcavTest_model()

    self.mytcav = TCAV(None,
                       self.target,
                       self.concepts,
                       [self.bottleneck],
                       self.mymodel,
                       self.activation_generator,
                       [self.hparams.alpha],
                       self.random_counterpart)

  def test_get_direction_dir_sign(self):
    self.assertFalse(TCAV.get_direction_dir_sign(self.mymodel,
                                                 self.acts,
                                                 self.cav,
                                                 self.concepts[0],
                                                 self.class_id))


  def test_compute_tcav_score(self):
    score = TCAV.compute_tcav_score(self.mymodel,
                                    self.target,
                                    self.concepts[0],
                                    self.cav,
                                    self.acts,
                                    run_parallel=False)
    self.assertAlmostEqual(0., score)

  def test_get_directional_dir(self):
    directional_dirs = TCAV.get_directional_dir(self.mymodel,
                                                self.target,
                                                self.concepts[0],
                                                self.cav,
                                                self.acts)
    self.assertAlmostEqual(8., directional_dirs[0])

  def test__run_single_set(self):
    """TODO(beenkim) not sure how to test this yet.
    """
    pass

  def test__process_what_to_run_expand(self):
    # _process_what_to_run_expand stores results to all_concepts,
    # and pairs_to_test.
    self.mytcav._process_what_to_run_expand(
        num_random_exp=2)
    self.assertEqual(sorted(self.mytcav.all_concepts),
                     sorted(['t1',
                             'c1',
                             'c2',
                             'random500_0',
                             'random500_1',
                             'random500_2'])
                    )
    self.assertEqual(sorted(self.mytcav.pairs_to_test),
                    sorted([('t1',['c1', 'random500_0']),
                            ('t1',['c1', 'random500_2']),
                            ('t1',['c2', 'random500_0']),
                            ('t1',['c2', 'random500_2']),
                            ('t1',['random500_1', 'random500_0']),
                            ('t1',['random500_1', 'random500_2'])
                           ]))

  def test_get_params(self):
    """Check if the first param was correct.
    """
    params = self.mytcav.get_params()
    self.assertEqual(params[0].bottleneck, 'bn')
    self.assertEqual(params[0].concepts, ['c1', 'random500_0'])
    self.assertEqual(params[0].target_class, 't1')
    self.assertEqual(params[0].activation_generator, None)
    self.assertEqual(params[0].cav_dir, self.cav_dir)
    self.assertEqual(params[0].alpha, self.hparams.alpha)
    self.assertEqual(params[0].model, self.mymodel)


if __name__ == '__main__':
  googletest.main()
