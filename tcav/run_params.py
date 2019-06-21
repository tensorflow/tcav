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


class RunParams(object):
  """Run parameters for TCAV."""

  def __init__(self,
               bottleneck,
               concepts,
               target_class,
               activation_generator,
               cav_dir,
               alpha,
               model,
               overwrite=True):
    """A simple class to take care of TCAV parameters.

    Args:
      bottleneck: the name of a bottleneck of interest.
      concepts: one concept
      target_class: one target class
      activation_generator: an ActivationGeneratorInterface instance
      cav_dir: the path to store CAVs
      alpha: cav parameter.
      model: an instance of a model class.
      overwrite: if set True, rewrite any files written in the *_dir path
    """
    self.bottleneck = bottleneck
    self.concepts = concepts
    self.target_class = target_class
    self.activation_generator = activation_generator
    self.cav_dir = cav_dir
    self.alpha = alpha
    self.overwrite = overwrite
    self.model = model

  def get_key(self):
    return '_'.join([
        str(self.bottleneck), '_'.join(self.concepts),
        'target_' + self.target_class, 'alpha_' + str(self.alpha)
    ])
