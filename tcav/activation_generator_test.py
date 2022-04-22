import os
import tempfile
import numpy as np
from tensorflow.python.platform import googletest
from PIL import Image
import weakref as _weakref
import sys

from tcav.activation_generator import ImageActivationGenerator
from tcav.tcav_examples.discrete.kdd99_activation_generator import KDD99DiscreteActivationGenerator
from tcav.tcav import TCAV
from tcav.utils import CONCEPT_SEPARATOR

IMG_SHAPE = (28, 28, 3)
MAX_TEST_IMAGES = 255


class TemporaryDirectory(tempfile.TemporaryDirectory):
    """
    Create temporary directories as defined int tempfile.TemporaryDirectory but when
    prefix starts with 'random500' do not add unique directory name, but keep directory
    name as is, defined by prefix+suffix.
    """
    def __init__(self, suffix=None, prefix=None, dir=None):
        prefix, suffix, dir, output_type = tempfile._sanitize_params(prefix, suffix, dir)
        if prefix.startswith('random500_'):
            self.name = os.path.join(dir, prefix + suffix)
            sys.audit("tempfile.mkstemp", self.name)
            os.mkdir(self.name, 0o700)
        else:
            self.name = tempfile.mkdtemp(suffix, prefix, dir)
        self._finalizer = _weakref.finalize(
            self, self._cleanup, self.name,
            warn_message="Implicitly cleaning up {!r}".format(self))


def _create_random_test_image_files(dst_dir, img_shape, count, start_offset, prefix):
    if (count + start_offset) > MAX_TEST_IMAGES:
        raise Exception(f"Cannot create more than '{MAX_TEST_IMAGES}' in one directory")

    count_magnitude = int(np.floor(np.log10(count)))
    test_file_paths = []
    for i in range(count):
        # create image with recognizable image values
        # image value corresponds to image filename,
        # zero values for img_0.jpg, ones for img_1.jpg, ...
        values = np.ones(shape=img_shape, dtype=np.uint8) * (i + start_offset)  # because of this, max_test_images=255
        img = Image.fromarray(values)
        # save to disk
        test_file = os.path.join(dst_dir, f"{prefix}_testfile_{i:0>{count_magnitude}d}.jpg")
        img.save(test_file)

        test_file_paths.append(test_file)

    return test_file_paths


def _create_concept_dirs(dst_dir, concept_settings, image_shape, test_file_count, test_file_prefix):
    concept_dirs = []
    for concept in concept_settings.keys():
        # create concept dir
        concept_dir = TemporaryDirectory(prefix=concept, dir=dst_dir)

        # fill concept dir with test files
        start_offset = concept_settings[concept].get('start_offset')
        _ = _create_random_test_image_files(concept_dir.name, image_shape, test_file_count, start_offset, test_file_prefix)

        concept_dirs.append(concept_dir)

    return concept_dirs


def _get_image_value(image):
    # return first pixel value
    return image[0, 0, 0]  # this works only for image len(shape)==3, e.g., for HxWxC


class MockTestModel:
    """
    A mock model of model class.
    """
    def __init__(self, image_shape):
        self.model_name = 'test_model'
        self.image_shape = image_shape

    def get_image_shape(self):
        return self.image_shape


class ActivationGeneratorTest(googletest.TestCase):
    def setUp(self):
        self.concept_image_dir = tempfile.TemporaryDirectory()

        self.image_test_file_prefix = 'img'
        self.image_test_file_count = 7

        self.concept_settings = {
            'concept-1': {'start_offset': 0},
            'concept-2': {'start_offset': self.image_test_file_count},
            'concept-3': {'start_offset': self.image_test_file_count*2},
            'random500_0': {'start_offset': self.image_test_file_count*3},
            'random500_1': {'start_offset': self.image_test_file_count*4},
            'random500_2': {'start_offset': self.image_test_file_count*5},
        }
        self.concept_dirs = _create_concept_dirs(self.concept_image_dir.name, self.concept_settings, IMG_SHAPE, self.image_test_file_count, self.image_test_file_prefix)
        self.concepts = [os.path.basename(concept_dir.name) for concept_dir in self.concept_dirs]

        self.model = MockTestModel(IMG_SHAPE)
        self.max_examples = 1000
        self.img_act_gen = ImageActivationGenerator(model=self.model, source_dir=self.concept_image_dir.name, acts_dir=None, max_examples=self.max_examples, normalize_image=False)

        self.target = 't0'
        self.bottleneck = 'bn'
        self.hparams = {'model_type': 'linear', 'alpha': .01}
        self.num_random_exp = 2
        self.normal_tcav = TCAV(sess=None,
                                target=self.target,
                                concepts=self.concepts,
                                bottlenecks=[self.bottleneck],
                                activation_generator=self.img_act_gen,
                                alphas=[self.hparams['alpha']],
                                num_random_exp=self.num_random_exp)
        self.relative_tcav = TCAV(sess=None,
                                  target=self.target,
                                  concepts=self.concepts,
                                  bottlenecks=[self.bottleneck],
                                  activation_generator=self.img_act_gen,
                                  alphas=[self.hparams['alpha']],
                                  random_concepts=self.concepts)  # this makes it relative_tcav

    def tearDown(self):
        self.concept_image_dir.cleanup()

        for concept_dir in self.concept_dirs:
            concept_dir.cleanup()

    def _get_concept_setting(self, concept_dir_name):
        for concept in self.concept_settings.keys():
            if concept_dir_name.startswith(concept):
                return self.concept_settings[concept]
        return None

    def _get_expected_values_by_concept(self, concept_name, is_relative_tcav=False):
        if is_relative_tcav:
            concepts = concept_name.split(CONCEPT_SEPARATOR)
        else:
            concepts = [concept_name]

        expected_set_image_values = set()
        for concept in concepts:
            concept_settings = self._get_concept_setting(concept)
            self.assertIsNotNone(concept_settings)

            start = concept_settings.get('start_offset')
            end = start + self.image_test_file_count
            expected_set_image_values.update(np.arange(start, end, dtype=np.float32))

        return expected_set_image_values

    def test_get_examples_for_concept(self):
        # (target, [positive-concept, negative-concept])
        concept_pairs = sorted(self.normal_tcav.pairs_to_test)

        # extract concepts
        concept_pair_id = 0
        target, (pos_concept, neg_concept) = concept_pairs[concept_pair_id]

        # get positive concept image values
        pos_set_images = self.img_act_gen.get_examples_for_concept(pos_concept)
        actual_pos_set_image_values = [_get_image_value(img) for img in pos_set_images]

        # compute expected values for positive concept
        expected_pos_set_image_values = self._get_expected_values_by_concept(pos_concept)

        # test whether correct positive set images were loaded
        self.assertEqual(len(expected_pos_set_image_values), len(actual_pos_set_image_values))
        actual_pos_set_image_values = set(actual_pos_set_image_values)
        self.assertEqual(expected_pos_set_image_values, actual_pos_set_image_values)

        # get negative concept image values
        neg_set_images = self.img_act_gen.get_examples_for_concept(neg_concept)
        actual_neg_set_image_values = [_get_image_value(img) for img in neg_set_images]

        # compute expected values for negative concept
        expected_neg_set_image_values = self._get_expected_values_by_concept(neg_concept)

        # test whether correct negative set images were loaded
        self.assertEqual(len(expected_neg_set_image_values), len(actual_neg_set_image_values))
        actual_neg_set_image_values = set(actual_neg_set_image_values)
        self.assertEqual(expected_neg_set_image_values, actual_neg_set_image_values)

    def test_get_examples_for_concept_relative_tcav(self):
        # (target, [positive-concept, negative-concept1+negative-concept2+...])
        concept_pairs = sorted(self.relative_tcav.pairs_to_test)

        # test if tcav object is relative tcav
        is_relative_tcav = self.relative_tcav.is_relative_tcav
        self.assertTrue(is_relative_tcav)

        # extract concepts
        concept_pair_id = 0
        target, (pos_concept, neg_concept) = concept_pairs[concept_pair_id]

        # get positive concept image values
        pos_set_images = self.img_act_gen.get_examples_for_concept(pos_concept, is_relative_tcav)
        actual_pos_set_image_values = [_get_image_value(img) for img in pos_set_images]

        # compute expected values for positive concept
        expected_pos_set_image_values = self._get_expected_values_by_concept(pos_concept, is_relative_tcav)

        # test whether correct positive set images were loaded
        self.assertEqual(len(expected_pos_set_image_values), len(actual_pos_set_image_values))
        actual_pos_set_image_values = set(actual_pos_set_image_values)
        self.assertEqual(expected_pos_set_image_values, actual_pos_set_image_values)

        # get negative concept image values
        neg_set_images = self.img_act_gen.get_examples_for_concept(neg_concept, is_relative_tcav)
        actual_neg_set_image_values = [_get_image_value(img) for img in neg_set_images]

        # compute expected values for negative concept
        expected_neg_set_image_values = self._get_expected_values_by_concept(neg_concept, is_relative_tcav)

        # test whether correct negative set images were loaded
        self.assertEqual(len(expected_neg_set_image_values), len(actual_neg_set_image_values))
        actual_neg_set_image_values = set(actual_neg_set_image_values)
        self.assertEqual(expected_neg_set_image_values, actual_neg_set_image_values)

    def test_get_examples_for_concept_discrete(self):
        discrete_act_gen = KDD99DiscreteActivationGenerator(
            model=None,
            source_dir=None,
            acts_dir=None,
            max_examples=0
        )
        concept = None
        is_relative = True
        self.assertRaises(NotImplementedError, discrete_act_gen.get_examples_for_concept, concept, is_relative)


if __name__ == '__main__':
    googletest.main()
