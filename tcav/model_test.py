"""
Copyright 2019 Google LLC
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
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import googletest
from tcav.model import ModelWrapper


tf.compat.v1.disable_eager_execution()


class ModelTest_model(ModelWrapper):
  """A mock model of model class for ModelTest class."""

  def __init__(self, model_path=None, node_dict=None):
    super(ModelTest_model, self).__init__(
        model_path=model_path, node_dict=node_dict)


class ModelTest(googletest.TestCase):

  def setUp(self):
    # Create an execution graph
    x = tf.compat.v1.placeholder(dtype=tf.float64, shape=[], name='input')
    a = tf.Variable(111, name='var1', dtype=tf.float64)
    y = tf.math.multiply(x, a, name='output')

    self.ckpt_dir = '/tmp/ckpts/'
    self.saved_model_dir = '/tmp/saved_model/'
    self.frozen_graph_dir = '/tmp/frozen_graph/'
    self.tmp_dirs = [self.ckpt_dir, self.saved_model_dir, self.frozen_graph_dir]
    for d in self.tmp_dirs:
      if tf.io.gfile.exists(d):
        tf.io.gfile.rmtree(d)
        tf.io.gfile.makedirs(d)

    with tf.compat.v1.Session() as sess:
      tf.compat.v1.initialize_all_variables().run()

      # Save as checkpoint
      saver = tf.compat.v1.train.Saver()
      saver.save(sess, self.ckpt_dir + 'model.ckpt', write_meta_graph=True)

      # Save as SavedModel
      tf.compat.v1.saved_model.simple_save(
          sess,
          self.saved_model_dir,
          inputs={'input': x},
          outputs={'output': y})

      graph = sess.graph
      input_graph_def = graph.as_graph_def()
      output_node_names = ['output']
      output_graph_def = graph_util.convert_variables_to_constants(
          sess, input_graph_def, output_node_names)

      # Save as binary graph
      tf.io.write_graph(
          output_graph_def, self.frozen_graph_dir, 'graph.pb', as_text=False)

      # Save as text graph
      tf.io.write_graph(
          output_graph_def, self.frozen_graph_dir, 'graph.pbtxt', as_text=True)

  def tearDown(self):
    for d in self.tmp_dirs:
      tf.io.gfile.rmtree(d)

  def _check_output_and_gradient(self, model_path, import_prefix=False):
    model = ModelTest_model(model_path=model_path, node_dict={'v1': 'var1'})
    input_name = 'input:0'
    output_name = 'output:0'
    if import_prefix:
      input_name = 'import/' + input_name
      output_name = 'import/' + output_name
    out = model.sess.run(output_name, feed_dict={input_name: 3})
    self.assertEqual(out, 333.0)

    model.loss = model.sess.graph.get_tensor_by_name(output_name)

    # Make sure that loaded graph can be modified
    model._make_gradient_tensors()
    grad = model.sess.run(
        model.bottlenecks_gradients['v1'], feed_dict={input_name: 555})
    self.assertEqual(grad, 555.0)

  def test_try_loading_model_from_ckpt(self):
    self._check_output_and_gradient(self.ckpt_dir)

  def test_try_loading_model_from_saved_model(self):
    self._check_output_and_gradient(self.saved_model_dir)

  def test_try_loading_model_from_frozen_pb(self):
    model_path = self.frozen_graph_dir + 'graph.pb'
    self._check_output_and_gradient(model_path, import_prefix=True)

  def test_try_loading_model_from_frozen_txt(self):
    model_path = self.frozen_graph_dir + 'graph.pbtxt'
    self._check_output_and_gradient(model_path, import_prefix=True)


if __name__ == '__main__':
  googletest.main()
