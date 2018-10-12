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

""" collection of various helper functions for running DTCAV"""

from multiprocessing import dummy as multiprocessing
import sys
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.append('..')
import model
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import tensorflow as tf

def make_model(model_to_run, sess, randomize=False, model_path=None,
               labels_path=None):
  """Make an instance of a model.

  Args:
    model_to_run: a string that describes which model to make.
    sess: tf session instance.
    randomize: Start with random weights
    model_path: Path to models saved graph. If None uses default paths
    labels_path: Path to models line separated labels text file. If None uses
                 default labels.

  Returns:
    a model instance.

  Raises:
    ValueError: If model name is not valid.
  """
  models_root = MODEL_LOCATION 
  if model_to_run == 'InceptionV3':
    basepath = models_root + 'inception_v3/'
    if model_path is None:
      model_path = basepath + 'tensorflow_inception_graph.pb'
    if labels_path is None:
      labels_path = basepath + 'imagenet_comp_graph_label_strings.txt'
    mymodel = model.InceptionV3Wrapper_public(
        sess, model_saved_path=model_path, labels_path=labels_path)
  elif model_to_run == 'GoogleNet':
    basepath = models_root + 'tensorflow_inception/'
    if model_path is None:
      model_path = basepath + 'tensorflow_inception_graph.pb'
    if labels_path is None:
      labels_path = basepath + 'imagenet_comp_graph_label_strings.txt'
    # common_typos_disable
    mymodel = model.GoolgeNetWrapper_public(
        sess, model_saved_path=model_path, labels_path=labels_path)
  else:
    raise ValueError('Invalid model name')
  if randomize:  # randomize the network!
    sess.run(tf.global_variables_initializer())
  return mymodel


def flat_profile(cd, images, bottlenecks=None):
  """Returns concept profile of given images.

  Given a ConceptDiscovery class instance and a set of images, and desired
  bottleneck layers, calculates the profile of each image with all concepts and
  returns a profile vector

  Args:
    cd: The concept discovery class instance
    images: The images for which the concept profile is calculated
    bottlenecks: Bottleck layers where the profile is calculated. If None, cd
      bottlenecks will be used.

  Returns:
    The concepts profile of input images using discovered concepts in
    all bottleneck layers.

  Raises:
    ValueError: If bottlenecks is not in right format.
  """
  profiles = []
  if bottlenecks is None:
    bottlenecks = list(cd.dic.keys())
  if isinstance(bottlenecks, str):
    bottlenecks = [bottlenecks]
  elif not isinstance(bottlenecks, list) and not isinstance(bottlenecks, tuple):
    raise ValueError('Invalid bottlenecks parameter!')
  for bn in bottlenecks:
    profiles.append(cd.find_profile(str(bn), images).reshape((len(images), -1)))
  profile = np.concatenate(profiles, -1)
  return profile


def cross_val(a, b, methods):
  """Performs cross validation for a binary classification task.

  Args:
    a: First class data points as rows
    b: Second class data points as rows
    methods: The sklearn classification models to perform cross-validation on

  Returns:
    The best performing trained binary classification odel
  """
  x, y = binary_dataset(a, b)
  best_acc = 0.
  if isinstance(methods, str):
    methods = [methods]
  best_acc = 0.
  for method in methods:
    temp_acc = 0.
    params = [10**e for e in [-4, -3, -2, -1, 0, 1, 2, 3]]
    for param in params:
      clf = give_classifier(method, param)
      acc = cross_val_score(clf, x, y, cv=min(100, max(2, int(len(y) / 10))))
      if np.mean(acc) > temp_acc:
        temp_acc = np.mean(acc)
        best_param = param
    if temp_acc > best_acc:
      best_acc = temp_acc
      final_clf = give_classifier(method, best_param)
  final_clf.fit(x, y)
  return final_clf, best_acc


def give_classifier(method, param):
  """Returns an sklearn classification model.

  Args:
    method: Name of the sklearn classification model
    param: Hyperparameters of the sklearn model

  Returns:
    An untrained sklearn classification model

  Raises:
    ValueError: if the model name is invalid.
  """
  if method == 'logistic':
    return linear_model.LogisticRegression(C=param)
  elif method == 'sgd':
    return linear_model.SGDClassifier(alpha=param)
  else:
    raise ValueError('Invalid model!')


def binary_dataset(pos, neg, balanced=True):
  """Creates a binary dataset given instances of two classes.

  Args:
     pos: Data points of the first class as rows
     neg: Data points of the second class as rows
     balanced: If true, it creates a balanced binary dataset.

  Returns:
    The data points of the created data set as rows and the corresponding labels
  """
  if balanced:
    min_len = min(neg.shape[0], pos.shape[0])
    ridxs = np.random.permutation(np.arange(2 * min_len))
    x = np.concatenate([neg[:min_len], pos[:min_len]], 0)[ridxs]
    y = np.concatenate([np.zeros(min_len), np.ones(min_len)], 0)[ridxs]
  else:
    ridxs = np.random.permutation(np.arange(len(neg) + len(pos)))
    x = np.concatenate([neg, pos], 0)[ridxs]
    y = np.concatenate(
        [np.zeros(neg.shape[0]), np.ones(pos.shape[0])], 0)[ridxs]
  return x, y


def plot_concepts(cd, bn, num=10, address=None, mode='diverse', concepts=None):
  """Plots examples of discovered concepts.

  Args:
    cd: The concept discovery instance
    bn: Bottleneck layer name
    num: Number of images to print out of each concept
    address: If not None, saves the output to the address as a .PNG image
    mode: If 'diverse', it prints one example of each of the target class images
      is coming from. If 'radnom', randomly samples exmples of the concept. If
      'max', prints out the most activating examples of that concept.
    concepts: If None, prints out examples of all discovered concepts.
      Otherwise, it should be either a list of concepts to print out examples of
      or just one concept's name

  Raises:
    ValueError: If the mode is invalid.
  """
  if concepts is None:
    concepts = cd.dic[bn]['concepts']
  elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
    concepts = [concepts]
  num_concepts = len(concepts)
  plt.rcParams['figure.figsize'] = num * 2.1, 4.3 * num_concepts
  fig = plt.figure(figsize=(num * 2, 4 * num_concepts))
  outer = gridspec.GridSpec(num_concepts, 1, wspace=0., hspace=0.3)
  for n, concept in enumerate(concepts):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, num, subplot_spec=outer[n], wspace=0, hspace=0.1)
    concept_images = cd.dic[bn][concept]['images']
    concept_patches = cd.dic[bn][concept]['patches']
    concept_image_numbers = cd.dic[bn][concept]['image_numbers']
    if mode == 'max':
      idxs = np.arange(len(concept_images))
    elif mode == 'random':
      idxs = np.random.permutation(np.arange(len(concept_images)))
    elif mode == 'diverse':
      idxs = []
      while True:
        seen = set()
        for idx in range(len(concept_images)):
          if concept_image_numbers[idx] not in seen and idx not in idxs:
            seen.add(concept_image_numbers[idx])
            idxs.append(idx)
        if len(idxs) == len(concept_images):
          break
    else:
      raise ValueError('Invalid mode!')
    idxs = idxs[:num]
    for i, idx in enumerate(idxs):
      ax = plt.Subplot(fig, inner[i])
      ax.imshow(concept_images[idx])
      ax.set_xticks([])
      ax.set_yticks([])
      if i == int(num / 2):
        ax.set_title(concept)
      ax.grid(False)
      fig.add_subplot(ax)
      ax = plt.Subplot(fig, inner[i + num])
      mask = 1 - (np.mean(concept_patches[idx] == float(
          cd.average_image_value) / 255, -1) == 1)
      image = cd.discovery_images[concept_image_numbers[idx]]
      ax.imshow(mark_boundaries(image, mask, color=(1, 1, 0), mode='thick'))
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_title(str(concept_image_numbers[idx]))
      ax.grid(False)
      fig.add_subplot(ax)
  plt.suptitle(bn)
  if address is not None:
    with tf.gfile.Open(address + bn + '.png', 'w') as f:
      fig.savefig(f)
    plt.clf()
    plt.close(fig)


def cosine_similarity(a, b):
  """Cosine similarity of two vectors."""
  assert a.shape == b.shape, 'Two vectors must have the same dimensionality'
  a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
  if a_norm * b_norm == 0:
    return 0.
  cos_sim = np.sum(a * b) / (a_norm * b_norm)
  return cos_sim


def similarity(cd, num_random_exp=None, num_workers=25):
  """Returns cosine similarity of all discovered concepts.

  Args:
    cd: The ConceptDiscovery module for discovered conceps.beenkim.tcav_refactor
      import tcav_meta from google3.experimental.userts of which we calculate
      pairwise similarity
    num_random_exp: If None, calculates average similarity using all the class's
      random concepts. If a number, uses that many random counterparts.
    num_workers: If greater than 0, runs the function in parallel.

  Returns:
    A similarity dict in the form of {(concept1, concept2):[list of cosine
    similarities]}
  """

  def concepts_similarity(cd, concepts, rnd, bn):
    """Calcualtes the cosine similarity of concept cavs.

    This function calculates the pairwise cosine similarity of all concept cavs
    versus an specific random concept

    Args:
      cd: The ConceptDiscovery instance
      concepts: List of concepts to calculate similarity for
      rnd: a random counterpart
      bn: bottleneck layer the concepts belong to

    Returns:
      A dictionary of cosine similarities in the form of
      {(concept1, concept2): [list of cosine similarities], ...}
    """
    similarity_dic = {}
    for c1 in concepts:
      cav1 = cd.load_cav_direction(c1, rnd, bn)
      for c2 in concepts:
        if (c1, c2) in similarity_dic.keys():
          continue
        cav2 = cd.load_cav_direction(c2, rnd, bn)
        similarity_dic[(c1, c2)] = cosine_similarity(cav1, cav2)
        similarity_dic[(c2, c1)] = similarity_dic[(c1, c2)]
    return similarity_dic

  similarity_dic = {bn: {} for bn in cd.bottlenecks}
  if num_random_exp is None:
    num_random_exp = cd.num_random_exp
  randoms = ['random500_{}'.format(i) for i in np.arange(num_random_exp)]
  concepts = {}
  for bn in cd.bottlenecks:
    concepts[bn] = [cd.target_class, cd.random_concept] + cd.dic[bn]['concepts']
  for bn in cd.bottlenecks:
    concept_pairs = [(c1, c2) for c1 in concepts[bn] for c2 in concepts[bn]]
    similarity_dic[bn] = {pair: [] for pair in concept_pairs}
    def t_func(rnd):
      return concepts_similarity(cd, concepts[bn], rnd, bn)
    if num_workers:
      pool = multiprocessing.Pool(num_workers)
      sims = pool.map(lambda rnd: t_func(rnd), randoms)
    else:
      sims = [t_func(rnd) for rnd in randoms]
    while sims:
      sim = sims.pop()
      for pair in concept_pairs:
        similarity_dic[bn][pair].append(sim[pair])
  return similarity_dic


def save_dtcav_report(cd, accs, scores, address):
  """Saves TCAV scores.

  Saves the average CAV accuracies and average TCAV scores of the concepts
  discovered in ConceptDiscovery instance.

  Args:
    cd: The ConceptDiscovery instance.
    accs: The cav accuracy dictionary returned by cavs method of the
      ConceptDiscovery instance
    scores: The tcav score dictionary returned by tcavs method of the
      ConceptDiscovery instance
    address: The address to save the text file in.
  """
  report = '\n\n\t\t\t ---CAV accuracies---'
  for bn in cd.bottlenecks:
    report += '\n'
    for concept in cd.dic[bn]['concepts']:
      report += '\n' + bn + ':' + concept + ':' + str(
          np.mean(accs[bn][concept]))
  save_report(address, report)
  report = '\n\n\t\t\t ---TCAV scores---'
  for bn in cd.bottlenecks:
    report += '\n'
    for concept in cd.dic[bn]['concepts']:
      pvalue = cd.do_statistical_testings(
          scores[bn][concept], scores[bn][cd.random_concept])
      report += '\n{}:{}:{},{}'.format(bn, concept,
                                       np.mean(scores[bn][concept]), pvalue)
  with tf.gfile.open(address, 'w') as f:
    f.write(report)


def save_concepts(cd, concepts_dir):
  """Saves discovered concept's images or patches.

  Args:
    cd: The ConceptDiscovery instance the concepts of which we want to save
    concepts_dir: The directory to save the concept images
  """
  for bn in cd.bottlenecks:
    for concept in cd.dic[bn]['concepts']:
      patches_dir = os.path.join(concepts_dir, bn + '_' + concept + '_patches')
      images_dir = os.path.join(concepts_dir, bn + '_' + concept)
      patches = (np.clip(cd.dic[bn][concept]['patches'], 0, 1) * 256).astype(
          np.uint8)
      images = (np.clip(cd.dic[bn][concept]['images'], 0, 1) * 256).astype(
          np.uint8)
      tf.gfile.MakeDirs(patches_dir)
      tf.gfile.MakeDirs(images_dir)
      image_numbers = cd.dic[bn][concept]['image_numbers']
      image_addresses, patch_addresses = [], []
      for i in range(len(images)):
        image_name = '0' * int(np.ceil(2 - np.log10(i + 1))) + '{}_{}'.format(
            i + 1, image_numbers[i])
        patch_addresses.append(os.path.join(patches_dir, image_name + '.png'))
        image_addresses.append(os.path.join(images_dir, image_name + '.png'))
      save_images(patch_addresses, patches)
      save_images(image_addresses, images)


def save_images(addresses, images):
  """Save images in the addresses.

  Args:
    addresses: The list of addresses to save the images as or the address of the
      directory to save all images in. (list or str)
    images: The list of all images in numpy uint8 format.
  """
  if not isinstance(addresses, list):
    image_addresses = []
    for i, image in enumerate(images):
      image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
      image_addresses.append(os.path.join(addresses, image_name))
    addresses = image_addresses
  assert len(addresses) == len(images), 'Invalid number of addresses'
  for address, image in zip(addresses, images):
    with tf.gfile.Open(address, 'w') as f:
      Image.fromarray(image).save(f, format='PNG')
