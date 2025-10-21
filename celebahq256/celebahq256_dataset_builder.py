"""celebahq256 dataset."""

import tensorflow_datasets as tfds
import numpy as np
from PIL import Image  # Để load PNG nhanh

class Builder(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Initial release.'}

  def _info(self) -> tfds.core.DatasetInfo:
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(256, 256, 3), encoding_format='png'),
            'label': tfds.features.ClassLabel(names=['female', 'male']),
        }),
    )

  def _split_generators(self, dl_manager):
    return {'train': self._generate_examples('train')}

  def _generate_examples(self, split):
    # Lazy imports to avoid module-level issues
    import pandas as pd
    from pathlib import Path

    # Load metadata
    df = pd.read_csv('/kaggle/working/metadata.csv')
    image_dir = Path('/kaggle/working/resized_images')

    # Yield từ files (no resize, siêu nhanh)
    for _, row in df.iterrows():
      image_path = row['image_path']
      image = np.array(Image.open(image_path))  # Load PNG → np array
      yield str(row['id']), {
          'image': image,
          'label': row['label'],
      }
