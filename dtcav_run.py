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

"""This script runs the whole DTCAV method."""


import os
import numpy as np
import sklearn.metrics as metrics
from tcav import utils
import tensorflow.google as tf

import dtcav_helpers
from dtcav import ConceptDiscovery
import argparse


def main(argv):

  model_to_run = args.model_to_run
  num_random_exp = args.num_random_exp
  max_imgs = args.max_imgs
  min_imgs = args.min_imgs
  num_concepts = args.num_concepts
  num_test = args.num_test
  target_class = args.target_class
  bottlenecks = args.bottlenecks.split(',')
  folder_name = args.folder_name
  patches = False
  superpixel_params = []
  for params in args.superpixel_params.split('-'):
    superpixel_params.append([float(param) for param in params.split(',')])
  ###### related DIRs on CNS to store results #######
  source_dir = '/cns/is-d/home/amiratag/imagenet_train/'
  test_dir = '/cns/is-d/home/amiratag/imagenet_val/'
  working_dir = '/cns/is-d/home/' + user + '/KM_new/' + folder_name
  discovered_concepts_dir = working_dir + '/concepts/'
  results_dir = working_dir + '/results/'
  cavs_dir = working_dir + '/cavs/'
  activations_dir = working_dir + '/acts/'
  results_summaries_dir = working_dir + '/results_summaries/'
  if tf.gfile.Exists(working_dir):
    tf.gfile.DeleteRecursively(working_dir)
  tf.gfile.MakeDirs(working_dir)
  tf.gfile.MakeDirs(discovered_concepts_dir)
  tf.gfile.MakeDirs(results_dir)
  tf.gfile.MakeDirs(cavs_dir)
  tf.gfile.MakeDirs(activations_dir)
  tf.gfile.MakeDirs(results_summaries_dir)
  random_concept = 'random500_100'  # Random concept for statistical testing
  sess = utils.create_session()
  mymodel = dtcav_helpers.make_model(model_to_run, sess)
  # Creating the ConceptDiscovery class instance
  cd = ConceptDiscovery(
      mymodel,
      target_class,
      random_concept,
      bottlenecks,
      sess,
      source_dir,
      activations_dir,
      cavs_dir,
      num_random_exp=num_random_exp,
      channel_mean=True,
      max_imgs=max_imgs,
      min_imgs=min_imgs,
      num_discovery_imgs=max_imgs,
      num_workers=25)
  # Creating the dataset of image patches
  cd.create_patches(param_dict={'n_segments': [15, 50, 80]})
  # Saving the concept discovery target class images
  image_dir = os.path.join(discovered_concepts_dir, 'images')
  tf.gfile.MakeDirs(image_dir)
  dtcav_helpers.save_images(image_dir,
                            (cd.discovery_images * 256).astype(np.uint8))
  # Discovering Concepts
  cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
  del cd.dataset  # Free memory
  del cd.image_numbers
  del cd.patches
  # Save discovered concept images (resized and original sized)
  dtcav_helpers.save_concepts(cd, discovered_concepts_dir)
  # Calculating CAVs and TCAV scores
  cav_accuraciess = cd.cavs(min_acc=0.0)
  scores = cd.tcavs(test=False)
  dtcav_helpers.save_dtcav_report(cd, cav_accuraciess, scores,
                                 results_summaries_dir + 'dtcav_results.txt')
  # Plot examples of discovered concepts
  for bn in cd.bottlenecks:
    dtcav_helpers.plot_concepts(cd, bn, 10, address=results_dir)
  # Delete concepts that don't pass statistical testing
  cd.test_and_remove_concepts(scores)
  # Train a binary classifier on concept profiles
  report = '\n\n\t\t\t ---Concept space---'
  report += '\n\t ---Classifier Weights---\n\n'
  pos_imgs = cd.load_concept_imgs(cd.target_class,
                                  2 * cd.max_imgs + num_test)[-num_test:]
  neg_imgs = cd.load_concept_imgs('random500_180', num_test)
  a = dtcav_helpers.flat_profile(cd, pos_imgs)
  b = dtcav_helpers.flat_profile(cd, neg_imgs)
  lm, _ = dtcav_helpers.cross_val(a, b, methods=['logistic'])
  for bn in cd.bottlenecks:
    report += bn + ':\n'
    for i, concept in enumerate(cd.dic[bn]['concepts']):
      report += concept + ':' + str(lm.coef_[-1][i]) + '\n'
  # Test profile classifier on test images
  cd.source_dir = test_dir
  pos_imgs = cd.load_concept_imgs(cd.target_class, num_test)
  neg_imgs = cd.load_concept_imgs('random500_180', num_test)
  a = dtcav_helpers.flat_profile(cd, pos_imgs)
  b = dtcav_helpers.flat_profile(cd, neg_imgs)
  x, y = dtcav_helpers.binary_dataset(a, b, balanced=True)
  probs = lm.predict_proba(x)[:, 1]
  report += '\nProfile Classifier accuracy= {}'.format(
      np.mean((probs > 0.5) == y))
  report += '\nProfile Classifier AUC= {}'.format(
      metrics.roc_auc_score(y, probs))
  report += '\nProfile Classifier PR Area= {}'.format(
      metrics.average_precision_score(y, probs))
  # Compare original network to profile classifier
  target_id = cd.model.label_to_id(cd.target_class.replace('_', ' '))
  predictions = []
  for img in pos_imgs:
    predictions.append(mymodel.get_predictions([img]))
  predictions = np.concatenate(predictions, 0)
  true_predictions = (np.argmax(predictions, -1) == target_id).astype(int)
  truly_predicted = np.where(true_predictions)[0]
  report += '\nNetwork Recall = ' + str(np.mean(true_predictions))
  report += ', ' + str(np.mean(np.max(predictions, -1)[truly_predicted]))
  agreeableness = np.sum(lm.predict(a) * true_predictions)*1./\
      np.sum(true_predictions + 1e-10)
  report += '\nProfile classifier agrees with network in {}%'.format(
      100 * agreeableness)
  with tf.gfile.Open(results_summaries_dir + 'profile_classifier.txt', 'w') as f:
    f.write(report)

def parse_arguments(argv):
  """Parses the arguments passed to the run.py script."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='InceptionV3')
  parser.add_argument('--working_dir', type=str,
      help='Directory to save the results.', default='./DTCAV')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='InceptionV3')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='Zebra')
  parser.add_argument('--bottlenecks', type=str,
      help='Names of the target layers of the network (comma separated)',
                      default='mixed_8')
  parser.add_argument('--num_test', type=int,
      help="Number of test images used for binary profile classifier",
                      default=20)
  parser.add_argument('--num_random_exp', type=int,
      help="Number of random experiments used for statistical testing, etc",
                      default=20)
  parser.add_argument('--max_imgs', type=int,
      help="Maximum number of images in a discovered concept",
                      default=40)
  parser.add_argument('--min_imgs', type=int,
      help="Minimum number of images in a discovered concept",
                      default=40)
  return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
