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

"""Tests for utils.py"""


from tensorflow.python.platform import googletest
from utils import flatten, process_what_to_run_expand, process_what_to_run_concepts, process_what_to_run_randoms


class UtilsTest(googletest.TestCase):

  def setUp(self):
    self.a_list = [[1, 2], [3, 4]]
    self.pairs_to_test = [('t1', ['c1', 'c2']), ('t2', ['c1', 'c2', 'c3'])]
    self.pair_to_test_one_concept = [('t1', ['c1']), ('t1', ['random500_1'])]
    self.random_counterpart = 'random500_1'

  def test_flatten(self):
    self.assertEqual([1, 2, 3, 4], flatten(self.a_list))

  def test_process_what_to_run_expand(self):
    all_concepts, pairs_to_test = process_what_to_run_expand(
        self.pair_to_test_one_concept,
        self.random_counterpart,
        num_random_exp=2)
    print(pairs_to_test)
    self.assertEqual(
        sorted(all_concepts),
        sorted(['t1', 'c1', 'random500_2', 'random500_1', 'random500_0']))
    self.assertEqual(
        sorted(pairs_to_test),
        sorted([('t1', ['c1', 'random500_0']), ('t1', ['c1', 'random500_2']),
                ('t1', ['random500_1', 'random500_0']),
                ('t1', ['random500_1', 'random500_2'])]))

  def test_process_what_to_run_concepts(self):
    self.assertEqual(
        process_what_to_run_concepts(self.pairs_to_test), [
            ['t1', ['c1']],
            ['t1', ['c2']],
            ['t2', ['c1']],
            ['t2', ['c2']],
            ['t2', ['c3']],
        ])

  def test_process_what_to_run_randoms(self):
    self.assertEqual(
        sorted(
            process_what_to_run_randoms(self.pairs_to_test,
                                        self.random_counterpart)),
        sorted([['t1', ['random500_1']], ['t2', ['random500_1']]]))


if __name__ == '__main__':
  googletest.main()
