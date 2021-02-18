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
from multiprocessing import dummy as multiprocessing
from six.moves import range
from tcav.cav import CAV
from tcav.cav import get_or_train_cav
from tcav import run_params
from tcav import utils
import numpy as np
import time
import tensorflow as tf


class TCAV(object):
  """TCAV object: runs TCAV for one target and a set of concepts.
  The static methods (get_direction_dir_sign, compute_tcav_score,
  get_directional_dir) invole getting directional derivatives and calculating
  TCAV scores. These are static because they might be useful independently,
  for instance, if you are developing a new interpretability method using
  CAVs.
  See https://arxiv.org/abs/1711.11279
  """

  @staticmethod
  def get_direction_dir_sign(mymodel, act, cav, concept, class_id, example):
    """Get the sign of directional derivative.

    Args:
        mymodel: a model class instance
        act: activations of one bottleneck to get gradient with respect to.
        cav: an instance of cav
        concept: one concept
        class_id: index of the class of interest (target) in logit layer.
        example: example corresponding to the given activation

    Returns:
        sign of the directional derivative
    """
    # Grad points in the direction which DECREASES probability of class
    grad = np.reshape(mymodel.get_gradient(
        act, [class_id], cav.bottleneck, example), -1)
    dot_prod = np.dot(grad, cav.get_direction(concept))
    return dot_prod < 0

  @staticmethod
  def compute_tcav_score(mymodel,
                         target_class,
                         concept,
                         cav,
                         class_acts,
                         examples,
                         run_parallel=True,
                         num_workers=20):
    """Compute TCAV score.

    Args:
      mymodel: a model class instance
      target_class: one target class
      concept: one concept
      cav: an instance of cav
      class_acts: activations of the examples in the target class where
        examples[i] corresponds to class_acts[i]
      examples: an array of examples of the target class where examples[i]
        corresponds to class_acts[i]
      run_parallel: run this parallel fashion
      num_workers: number of workers if we run in parallel.

    Returns:
        TCAV score (i.e., ratio of pictures that returns negative dot product
        wrt loss).
    """
    count = 0
    class_id = mymodel.label_to_id(target_class)
    if run_parallel:
      pool = multiprocessing.Pool(num_workers)
      directions = pool.map(
          lambda i: TCAV.get_direction_dir_sign(
              mymodel, np.expand_dims(class_acts[i], 0),
              cav, concept, class_id, examples[i]),
          range(len(class_acts)))
      return sum(directions) / float(len(class_acts))
    else:
      for i in range(len(class_acts)):
        act = np.expand_dims(class_acts[i], 0)
        example = examples[i]
        if TCAV.get_direction_dir_sign(
            mymodel, act, cav, concept, class_id, example):
          count += 1
      return float(count) / float(len(class_acts))

  @staticmethod
  def get_directional_dir(
      mymodel, target_class, concept, cav, class_acts, examples):
    """Return the list of values of directional derivatives.

       (Only called when the values are needed as a referece)

    Args:
      mymodel: a model class instance
      target_class: one target class
      concept: one concept
      cav: an instance of cav
      class_acts: activations of the examples in the target class where
        examples[i] corresponds to class_acts[i]
      examples: an array of examples of the target class where examples[i]
        corresponds to class_acts[i]

    Returns:
      list of values of directional derivatives.
    """
    class_id = mymodel.label_to_id(target_class)
    directional_dir_vals = []
    for i in range(len(class_acts)):
      act = np.expand_dims(class_acts[i], 0)
      example = examples[i]
      grad = np.reshape(
          mymodel.get_gradient(act, [class_id], cav.bottleneck, example), -1)
      directional_dir_vals.append(np.dot(grad, cav.get_direction(concept)))
    return directional_dir_vals

  def __init__(self,
               sess,
               target,
               concepts,
               bottlenecks,
               activation_generator,
               alphas,
               random_counterpart=None,
               cav_dir=None,
               num_random_exp=5,
               random_concepts=None):
    """Initialze tcav class.

    Args:
      sess: tensorflow session.
      target: one target class
      concepts: A list of names of positive concept sets.
      bottlenecks: the name of a bottleneck of interest.
      activation_generator: an ActivationGeneratorInterface instance to return
                            activations.
      alphas: list of hyper parameters to run
      cav_dir: the path to store CAVs
      random_counterpart: the random concept to run against the concepts for
                  statistical testing. If supplied, only this set will be
                  used as a positive set for calculating random TCAVs
      num_random_exp: number of random experiments to compare against.
      random_concepts: A list of names of random concepts for the random
                       experiments to draw from. Optional, if not provided, the
                       names will be random500_{i} for i in num_random_exp.
                       Relative TCAV can be performed by passing in the same
                       value for both concepts and random_concepts.
    """
    self.target = target
    self.concepts = concepts
    self.bottlenecks = bottlenecks
    self.activation_generator = activation_generator
    self.cav_dir = cav_dir
    self.alphas = alphas
    self.mymodel = activation_generator.get_model()
    self.model_to_run = self.mymodel.model_name
    self.sess = sess
    self.random_counterpart = random_counterpart
    self.relative_tcav = (random_concepts is not None) and (set(concepts) == set(random_concepts))

    if num_random_exp < 2:
        tf.compat.v1.logging.error('the number of random concepts has to be at least 2')
    if random_concepts:
      num_random_exp = len(random_concepts)

    # make pairs to test.
    self._process_what_to_run_expand(num_random_exp=num_random_exp,
                                     random_concepts=random_concepts)
    # parameters
    self.params = self.get_params()
    tf.compat.v1.logging.info('TCAV will %s params' % len(self.params))

  def run(self, num_workers=10, run_parallel=False, overwrite=False, return_proto=False):
    """Run TCAV for all parameters (concept and random), write results to html.

    Args:
      num_workers: number of workers to parallelize
      run_parallel: run this parallel.
      overwrite: if True, overwrite any saved CAV files.
      return_proto: if True, returns results as a tcav.Results object; else,
        return as a list of dicts.

    Returns:
      results: an object (either a Results proto object or a list of
        dictionaries) containing metrics for TCAV results.
    """
    # for random exp,  a machine with cpu = 30, ram = 300G, disk = 10G and
    # pool worker 50 seems to work.
    tf.compat.v1.logging.info('running %s params' % len(self.params))
    results = []
    now = time.time()
    if run_parallel:
      pool = multiprocessing.Pool(num_workers)
      for i, res in enumerate(pool.imap(
          lambda p: self._run_single_set(
            p, overwrite=overwrite, run_parallel=run_parallel),
          self.params), 1):
        tf.compat.v1.logging.info('Finished running param %s of %s' % (i, len(self.params)))
        results.append(res)
    else:
      for i, param in enumerate(self.params):
        tf.compat.v1.logging.info('Running param %s of %s' % (i, len(self.params)))
        results.append(self._run_single_set(param, overwrite=overwrite, run_parallel=run_parallel))
    tf.compat.v1.logging.info('Done running %s params. Took %s seconds...' % (len(
        self.params), time.time() - now))
    if return_proto:
      return utils.results_to_proto(results)
    else:
      return results

  def _run_single_set(self, param, overwrite=False, run_parallel=False):
    """Run TCAV with provided for one set of (target, concepts).

    Args:
      param: parameters to run
      overwrite: if True, overwrite any saved CAV files.
      run_parallel: run this parallel.

    Returns:
      a dictionary of results (panda frame)
    """
    bottleneck = param.bottleneck
    concepts = param.concepts
    target_class = param.target_class
    activation_generator = param.activation_generator
    alpha = param.alpha
    mymodel = param.model
    cav_dir = param.cav_dir
    # first check if target class is in model.

    tf.compat.v1.logging.info('running %s %s' % (target_class, concepts))

    # Get acts
    acts = activation_generator.process_and_load_activations(
        [bottleneck], concepts + [target_class])
    # Get CAVs
    cav_hparams = CAV.default_hparams()
    cav_hparams['alpha'] = alpha
    cav_instance = get_or_train_cav(
        concepts,
        bottleneck,
        acts,
        cav_dir=cav_dir,
        cav_hparams=cav_hparams,
        overwrite=overwrite)

    # clean up
    for c in concepts:
      del acts[c]

    # Hypo testing
    a_cav_key = CAV.cav_key(concepts, bottleneck, cav_hparams['model_type'],
                            cav_hparams['alpha'])
    target_class_for_compute_tcav_score = target_class

    cav_concept = concepts[0]

    i_up = self.compute_tcav_score(
        mymodel, target_class_for_compute_tcav_score, cav_concept,
        cav_instance, acts[target_class][cav_instance.bottleneck],
        activation_generator.get_examples_for_concept(target_class),
        run_parallel=run_parallel)
    val_directional_dirs = self.get_directional_dir(
        mymodel, target_class_for_compute_tcav_score, cav_concept,
        cav_instance, acts[target_class][cav_instance.bottleneck],
        activation_generator.get_examples_for_concept(target_class))
    result = {
        'cav_key':
            a_cav_key,
        'cav_concept':
            cav_concept,
        'negative_concept':
            concepts[1],
        'target_class':
            target_class,
        'cav_accuracies':
            cav_instance.accuracies,
        'i_up':
            i_up,
        'val_directional_dirs_abs_mean':
            np.mean(np.abs(val_directional_dirs)),
        'val_directional_dirs_mean':
            np.mean(val_directional_dirs),
        'val_directional_dirs_std':
            np.std(val_directional_dirs),
        'val_directional_dirs':
            val_directional_dirs,
        'note':
            'alpha_%s ' % (alpha),
        'alpha':
            alpha,
        'bottleneck':
            bottleneck
    }
    del acts
    return result

  def _process_what_to_run_expand(self, num_random_exp=100, random_concepts=None):
    """Get tuples of parameters to run TCAV with.

    TCAV builds random concept to conduct statistical significance testing
    againts the concept. To do this, we build many concept vectors, and many
    random vectors. This function prepares runs by expanding parameters.

    Args:
      num_random_exp: number of random experiments to run to compare.
      random_concepts: A list of names of random concepts for the random experiments
                   to draw from. Optional, if not provided, the names will be
                   random500_{i} for i in num_random_exp.
    """

    target_concept_pairs = [(self.target, self.concepts)]

    # take away 1 random experiment if the random counterpart already in random concepts
    # take away 1 random experiment if computing Relative TCAV
    all_concepts_concepts, pairs_to_run_concepts = (
        utils.process_what_to_run_expand(
            utils.process_what_to_run_concepts(target_concept_pairs),
            self.random_counterpart,
            num_random_exp=num_random_exp -
            (1 if random_concepts and self.random_counterpart in random_concepts
             else 0) - (1 if self.relative_tcav else 0),
            random_concepts=random_concepts))

    pairs_to_run_randoms = []
    all_concepts_randoms = []

    # ith random concept
    def get_random_concept(i):
      return (random_concepts[i] if random_concepts
              else 'random500_{}'.format(i))

    if self.random_counterpart is None:
      # TODO random500_1 vs random500_0 is the same as 1 - (random500_0 vs random500_1)
      for i in range(num_random_exp):
        all_concepts_randoms_tmp, pairs_to_run_randoms_tmp = (
            utils.process_what_to_run_expand(
                utils.process_what_to_run_randoms(target_concept_pairs,
                                                  get_random_concept(i)),
                num_random_exp=num_random_exp - 1,
                random_concepts=random_concepts))

        pairs_to_run_randoms.extend(pairs_to_run_randoms_tmp)
        all_concepts_randoms.extend(all_concepts_randoms_tmp)

    else:
      # run only random_counterpart as the positve set for random experiments
      all_concepts_randoms_tmp, pairs_to_run_randoms_tmp = (
          utils.process_what_to_run_expand(
              utils.process_what_to_run_randoms(target_concept_pairs,
                                                self.random_counterpart),
              self.random_counterpart,
              num_random_exp=num_random_exp - (1 if random_concepts and
                  self.random_counterpart in random_concepts else 0),
              random_concepts=random_concepts))

      pairs_to_run_randoms.extend(pairs_to_run_randoms_tmp)
      all_concepts_randoms.extend(all_concepts_randoms_tmp)

    self.all_concepts = list(set(all_concepts_concepts + all_concepts_randoms))
    self.pairs_to_test = pairs_to_run_concepts if self.relative_tcav else pairs_to_run_concepts + pairs_to_run_randoms

  def get_params(self):
    """Enumerate parameters for the run function.

    Returns:
      parameters
    """
    params = []
    for bottleneck in self.bottlenecks:
      for target_in_test, concepts_in_test in self.pairs_to_test:
        for alpha in self.alphas:
          tf.compat.v1.logging.info('%s %s %s %s', bottleneck, concepts_in_test,
                          target_in_test, alpha)
          params.append(
              run_params.RunParams(bottleneck, concepts_in_test, target_in_test,
                                   self.activation_generator, self.cav_dir,
                                   alpha, self.mymodel))
    return params
