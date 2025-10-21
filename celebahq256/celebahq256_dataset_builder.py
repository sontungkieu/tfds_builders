"""celebahq256 dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
import numpy as np
from datasets import load_dataset

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for celebahq256 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(celebahqhq): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(256, 256, 3), encoding_format='png'),
            'label': tfds.features.ClassLabel(names=['female', 'male']),
        }),
        # homepage='https://huggingface.co/datasets/mattymchen/celeba-hq',  # Optional
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # The dataset only has a 'train' split; adjust if you want to split it manually
    return {
        'train': self._split_generator(split_type='train'),
        # 'validation': self._split_generator(split_type='validation'),  # Commented out; dataset lacks this split
    }

  def _split_generator(self, split_type):
    """Helper to create SplitGenerator."""
    return tfds.core.SplitGenerator(
        split_name=split_type,
        gen_kwargs={'split': split_type},
    )

  def _generate_examples(self, **kwargs):
    """Yields examples."""
    split = kwargs['split']
    # TODO(celebahqhq): Yields (key, example) tuples from the dataset

    dataset = load_dataset("mattymchen/celeba-hq", split=split)
    for i, example in enumerate(dataset):
        # Resize image using PIL (efficient, no TF needed; LANCZOS ≈ bilinear + antialias)
        image = example['image'].resize((256, 256), Image.Resampling.LANCZOS)
        image_np = np.array(image, dtype=np.uint8)
        label = example['label']  # int: 0 (female) or 1 (male)
        yield str(i), {
            'image': image_np,
            'label': label,
        }
