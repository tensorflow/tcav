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

from scipy.stats import ttest_ind
import numpy as np
import tensorflow as tf
from tcav.tcav_results.results_pb2 import Result, Results

_KEYS = [
    "cav_key", "cav_concept", "negative_concept", "target_class", "i_up",
    "val_directional_dirs_abs_mean", "val_directional_dirs_mean",
    "val_directional_dirs_std", "note", "alpha", "bottleneck"
]


def create_session(timeout=10000, interactive=True):
  """Create a tf session for the model.
  # This function is slight motification of code written by Alex Mordvintsev

  Args:
    timeout: tfutil param.

  Returns:
    TF session.
  """
  graph = tf.Graph()
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  config.operation_timeout_in_ms = int(timeout*1000)
  if interactive:
    return tf.compat.v1.InteractiveSession(graph=graph, config=config)
  else:
    return tf.compat.v1.Session(graph=graph, config=config)


def flatten(nested_list):
  """Flatten a nested list."""
  return [item for a_list in nested_list for item in a_list]


def process_what_to_run_expand(pairs_to_test,
                               random_counterpart=None,
                               num_random_exp=100,
                               random_concepts=None):
  """Get concept vs. random or random vs. random pairs to run.

    Given set of target, list of concept pairs, expand them to include
     random pairs. For instance [(t1, [c1, c2])...] becomes
     [(t1, [c1, random1],
      (t1, [c1, random2],...
      (t1, [c2, random1],
      (t1, [c2, random2],...]

  Args:
    pairs_to_test: [(target1, concept1), (target1, concept2), ...,
                    (target2, concept1), (target2, concept2), ...]
    random_counterpart: random concept that will be compared to the concept.
    num_random_exp: number of random experiments to run against each concept.
    random_concepts: A list of names of random concepts for the random
                     experiments to draw from. Optional, if not provided, the
                     names will be random500_{i} for i in num_random_exp.

  Returns:
    all_concepts: unique set of targets/concepts
    new_pairs_to_test: expanded
  """
  def get_random_concept(i):
    return (random_concepts[i] if random_concepts
            else 'random500_{}'.format(i))

  new_pairs_to_test = []
  for (target, concept_set) in pairs_to_test:
    new_pairs_to_test_t = []
    # if only one element was given, this is to test with random.
    if len(concept_set) == 1:
      i = 0
      while len(new_pairs_to_test_t) < min(100, num_random_exp):
        # make sure that we are not comparing the same thing to each other.
        if concept_set[0] != get_random_concept(
            i) and random_counterpart != get_random_concept(i):
          new_pairs_to_test_t.append(
              (target, [concept_set[0], get_random_concept(i)]))
        i += 1
    elif len(concept_set) > 1:
      new_pairs_to_test_t.append((target, concept_set))
    else:
      tf.compat.v1.logging.info('PAIR NOT PROCCESSED')
    new_pairs_to_test.extend(new_pairs_to_test_t)

  all_concepts = list(set(flatten([cs + [tc] for tc, cs in new_pairs_to_test])))

  return all_concepts, new_pairs_to_test


def process_what_to_run_concepts(pairs_to_test):
  """Process concepts and pairs to test.

  Args:
    pairs_to_test: a list of concepts to be tested and a target (e.g,
     [ ("target1",  ["concept1", "concept2", "concept3"]),...])

  Returns:
    return pairs to test:
       target1, concept1
       target1, concept2
       ...
       target2, concept1
       target2, concept2
       ...

  """

  pairs_for_sstesting = []
  # prepare pairs for concpet vs random.
  for pair in pairs_to_test:
    for concept in pair[1]:
      pairs_for_sstesting.append([pair[0], [concept]])
  return pairs_for_sstesting


def process_what_to_run_randoms(pairs_to_test, random_counterpart):
  """Process concepts and pairs to test.

  Args:
    pairs_to_test: a list of concepts to be tested and a target (e.g,
     [ ("target1",  ["concept1", "concept2", "concept3"]),...])
    random_counterpart: a random concept that will be compared to the concept.

  Returns:
    return pairs to test:
          target1, random_counterpart,
          target2, random_counterpart,
          ...
  """
  # prepare pairs for random vs random.
  pairs_for_sstesting_random = []
  targets = list(set([pair[0] for pair in pairs_to_test]))
  for target in targets:
    pairs_for_sstesting_random.append([target, [random_counterpart]])
  return pairs_for_sstesting_random


# helper functions to write summary files
def print_results(results, random_counterpart=None, random_concepts=None, num_random_exp=100,
    min_p_val=0.05):
  """Helper function to organize results.
  If you ran TCAV with a random_counterpart, supply it here, otherwise supply random_concepts.
  If you get unexpected output, make sure you are using the correct keywords.

  Args:
    results: dictionary of results from TCAV runs.
    random_counterpart: name of the random_counterpart used, if it was used.
    random_concepts: list of random experiments that were run.
    num_random_exp: number of random experiments that were run.
    min_p_val: minimum p value for statistical significance
  """

  # helper function, returns if this is a random concept
  def is_random_concept(concept):
    if random_counterpart:
      return random_counterpart == concept

    elif random_concepts:
      return concept in random_concepts

    else:
      return 'random500_' in concept

  # print class, it will be the same for all
  print("Class =", results[0]['target_class'])

  # prepare data
  # dict with keys of concepts containing dict with bottlenecks
  result_summary = {}

  # random
  random_i_ups = {}

  for result in results:
    if result['cav_concept'] not in result_summary:
      result_summary[result['cav_concept']] = {}

    if result['bottleneck'] not in result_summary[result['cav_concept']]:
      result_summary[result['cav_concept']][result['bottleneck']] = []

    result_summary[result['cav_concept']][result['bottleneck']].append(result)

    # store random
    if is_random_concept(result['cav_concept']):
      if result['bottleneck'] not in random_i_ups:
        random_i_ups[result['bottleneck']] = []

      random_i_ups[result['bottleneck']].append(result['i_up'])

  # print concepts and classes with indentation
  for concept in result_summary:

    # if not random
    if not is_random_concept(concept):
      print(" ", "Concept =", concept)

      for bottleneck in result_summary[concept]:
        i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]

        # Calculate statistical significance
        _, p_val = ttest_ind(random_i_ups[bottleneck], i_ups)

        print(3 * " ", "Bottleneck =", ("%s. TCAV Score = %.2f (+- %.2f), "
            "random was %.2f (+- %.2f). p-val = %.3f (%s)") % (
            bottleneck, np.mean(i_ups), np.std(i_ups),
            np.mean(random_i_ups[bottleneck]),
            np.std(random_i_ups[bottleneck]), p_val,
            "undefined" if np.isnan(p_val) else "not significant" if p_val > min_p_val else "significant"))


def make_dir_if_not_exists(directory):
  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)


def result_to_proto(result):
  """Given a result dict, convert it to a tcav.Result proto.

  Args:
    result: a dictionary returned by tcav._run_single_set()

  Returns:
    TCAV.Result proto
  """
  result_proto = Result()
  for key in _KEYS:
    setattr(result_proto, key, result[key])
  positive_set_name = result["cav_concept"]
  negative_set_name = result["negative_concept"]
  for val in result["val_directional_dirs"]:
    result_proto.val_directional_dirs.append(val)
  result_proto.cav_accuracies.positive_set_accuracy = result["cav_accuracies"][
      positive_set_name]
  result_proto.cav_accuracies.negative_set_accuracy = result["cav_accuracies"][
      negative_set_name]
  result_proto.cav_accuracies.overall_accuracy = result["cav_accuracies"][
      "overall"]
  return result_proto


def results_to_proto(results):
  """Given a list of result dicts, convert it to a tcav.Results proto.

  Args:
    results: a list of dictionaries returned by tcav.run()

  Returns:
    TCAV.Results proto
  """
  results_proto = Results()
  for result in results:
    results_proto.results.append(result_to_proto(result))
  return results_proto
