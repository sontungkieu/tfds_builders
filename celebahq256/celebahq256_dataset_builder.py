"""celebahq256 dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for celebahq256 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(256, 256, 3), encoding_format='png'),
            'label': tfds.features.ClassLabel(names=['female', 'male']),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # Fix: Directly call _generate_examples to return the generator object (not callable)
    return {
        'train': self._generate_examples('train'),
        # 'validation': self._generate_examples('validation'),  # Uncomment nếu manual split
    }

  def _generate_examples(self, split):
    """Yields examples."""
    # Lazy imports inside method to avoid module-level import issues in tfds CLI
    from datasets import load_dataset
    from PIL import Image

    dataset = load_dataset("mattymchen/celeba-hq", split=split)
    for i, example in enumerate(dataset):
        # Resize image using PIL (LANCZOS for quality)
        image = example['image'].resize((256, 256), Image.Resampling.LANCZOS)
        image_np = np.array(image, dtype=np.uint8)
        label = example['label']  # 0: female, 1: male
        yield str(i), {
            'image': image_np,
            'label': label,
        }
