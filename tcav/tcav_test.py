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
from tcav.activation_generator import ActivationGeneratorBase
from tcav.cav import CAV
from tcav.tcav import TCAV
from tensorflow.python.platform import googletest
import numpy as np
import tensorflow as tf


class TcavTest_model():
  """A mock model of model class for TcavTest class.
  """

  def __init__(self):
    self.model_name = 'test_model'

  def get_gradient(self, act, class_id, bottleneck, example):
    return act

  def label_to_id(self, target_class):
    """Returning one label.

    Args:
      target_class: dummy target_class

    Returns:
      zero.
    """
    return 0

class TcavTest_ActGen(ActivationGeneratorBase):
  """A mock act gen.
  """

  def __init__(self, model):
    super(TcavTest_ActGen, self).__init__(model, None, 10)

  def get_examples_for_concept(self, concept):
    return []

class TcavTest(googletest.TestCase):

  def setUp(self):
    self.acts = np.array([[0, 1., 2.]])
    self.examples = [None, None, None]
    self.concepts = ['c1', 'c2']
    self.target = 't1'
    self.class_id = 0
    self.bottleneck = 'bn'
    self.cav_dir = None
    self.hparams = {'model_type':'linear', 'alpha':.01}
    self.cav = CAV(self.concepts,
                       self.bottleneck,
                       self.hparams)
    self.cav.cavs = [[1., 2., 3.,]]
    self.activation_generator = None
    self.mymodel = TcavTest_model()
    self.act_gen = TcavTest_ActGen(self.mymodel)
    self.random_counterpart = 'random500_1'

    self.mytcav = TCAV(None,
                       self.target,
                       self.concepts,
                       [self.bottleneck],
                       self.act_gen,
                       [self.hparams['alpha']])

    self.mytcav_random_counterpart = TCAV(None,
                                          self.target,
                                          self.concepts,
                                          [self.bottleneck],
                                          self.act_gen,
                                          [self.hparams['alpha']],
                                          self.random_counterpart)

  def test_get_direction_dir_sign(self):
    self.assertFalse(TCAV.get_direction_dir_sign(self.mymodel,
                                                 self.acts,
                                                 self.cav,
                                                 self.concepts[0],
                                                 self.class_id,
                                                 None))


  def test_compute_tcav_score(self):
    score = TCAV.compute_tcav_score(self.mymodel,
                                    self.target,
                                    self.concepts[0],
                                    self.cav,
                                    self.acts,
                                    self.examples,
                                    run_parallel=False)
    self.assertAlmostEqual(0., score)

  def test_get_directional_dir(self):
    directional_dirs = TCAV.get_directional_dir(self.mymodel,
                                                self.target,
                                                self.concepts[0],
                                                self.cav,
                                                self.acts,
                                                self.examples)
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
                             'random500_1'])
                    )
    self.assertEqual(sorted(self.mytcav.pairs_to_test),
                    sorted([('t1',['c1', 'random500_0']),
                            ('t1',['c1', 'random500_1']),
                            ('t1',['c2', 'random500_0']),
                            ('t1',['c2', 'random500_1']),
                            ('t1',['random500_0', 'random500_1']),
                            ('t1',['random500_1', 'random500_0'])
                           ]))

  def test__process_what_to_run_expand_random_counterpart(self):
    # _process_what_to_run_expand stores results to all_concepts,
    # and pairs_to_test.
    # test when random_counterpart is supplied
    self.mytcav_random_counterpart._process_what_to_run_expand(
        num_random_exp=2)
    self.assertEqual(sorted(self.mytcav_random_counterpart.all_concepts),
                     sorted(['t1',
                             'c1',
                             'c2',
                             'random500_0',
                             'random500_1',
                             'random500_2'])
                    )
    self.assertEqual(sorted(self.mytcav_random_counterpart.pairs_to_test),
                    sorted([('t1',['c1', 'random500_0']),
                            ('t1',['c1', 'random500_2']),
                            ('t1',['c2', 'random500_0']),
                            ('t1',['c2', 'random500_2']),
                            ('t1',['random500_1', 'random500_0']),
                            ('t1',['random500_1', 'random500_2'])
                           ]))

  def test__process_what_to_run_expand_specify_dirs(self):
    # _process_what_to_run_expand stores results to all_concepts,
    # and pairs_to_test.
    self.mytcav._process_what_to_run_expand(
        num_random_exp=2, random_concepts=['random_dir1', 'random_dir2'])
    self.assertEqual(sorted(self.mytcav.all_concepts),
                     sorted(['t1',
                             'c1',
                             'c2',
                             'random_dir1',
                             'random_dir2'])
                    )
    self.assertEqual(sorted(self.mytcav.pairs_to_test),
                    sorted([('t1',['c1', 'random_dir1']),
                            ('t1',['c1', 'random_dir2']),
                            ('t1',['c2', 'random_dir1']),
                            ('t1',['c2', 'random_dir2']),
                            ('t1',['random_dir1', 'random_dir2']),
                            ('t1',['random_dir2', 'random_dir1'])
                           ]))

  def test__process_what_to_run_expand_specify_dirs_random_concepts(self):
    # _process_what_to_run_expand stores results to all_concepts,
    # and pairs_to_test.
    # test when random_counterpart is supplied
    self.mytcav_random_counterpart._process_what_to_run_expand(
        num_random_exp=2, random_concepts=['random_dir1', 'random_dir2'])
    self.assertEqual(sorted(self.mytcav_random_counterpart.all_concepts),
                     sorted(['t1',
                             'c1',
                             'c2',
                             'random500_1',
                             'random_dir1',
                             'random_dir2'])
                    )
    self.assertEqual(sorted(self.mytcav_random_counterpart.pairs_to_test),
                    sorted([('t1',['c1', 'random_dir1']),
                            ('t1',['c1', 'random_dir2']),
                            ('t1',['c2', 'random_dir1']),
                            ('t1',['c2', 'random_dir2']),
                            ('t1',['random500_1', 'random_dir1']),
                            ('t1',['random500_1', 'random_dir2'])
                           ]))

  def test__process_what_to_run_expand_relative_tcav(self):
    # _process_what_to_run_expand stores results to all_concepts,
    # and pairs_to_test.
    # test when concepts and random_concepts contain the same elements
    concepts_relative = ['c1', 'c2', 'c3']
    my_relative_tcav = TCAV(None,
                            self.target,
                            concepts_relative,
                            [self.bottleneck],
                            self.act_gen,
                            [self.hparams['alpha']],
                            random_concepts=concepts_relative)
    self.mytcav_random_counterpart._process_what_to_run_expand(
        num_random_exp=2, random_concepts=concepts_relative)
    self.assertEqual(sorted(my_relative_tcav.all_concepts),
                     sorted(['t1', 'c1', 'c2', 'c3']))
    self.assertEqual(sorted(my_relative_tcav.pairs_to_test),
                     sorted([('t1',['c1', 'c2']),
                             ('t1',['c1', 'c3']),
                             ('t1',['c2', 'c1']),
                             ('t1',['c2', 'c3']),
                             ('t1',['c3', 'c1']),
                             ('t1',['c3', 'c2']),
                             ]))

  def test_get_params(self):
    """Check if the first param was correct.
    """
    params = self.mytcav.get_params()
    self.assertEqual(params[0].bottleneck, 'bn')
    self.assertEqual(params[0].concepts, ['c1', 'random500_0'])
    self.assertEqual(params[0].target_class, 't1')
    self.assertEqual(params[0].activation_generator, self.act_gen)
    self.assertEqual(params[0].cav_dir, self.cav_dir)
    self.assertEqual(params[0].alpha, self.hparams['alpha'])
    self.assertEqual(params[0].model, self.mymodel)


if __name__ == '__main__':
  googletest.main()
