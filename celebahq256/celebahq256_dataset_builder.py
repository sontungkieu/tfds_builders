"""celebahq256 dataset."""

import tensorflow_datasets as tfds
import numpy as np
import cv2  # Faster resize (install if needed: !pip install opencv-python)

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
    from datasets import load_dataset
    from tqdm import tqdm  # Progress bar for generation

    dataset = load_dataset("mattymchen/celeba-hq", split=split)
    total_samples = len(dataset)  # Known size for tqdm

    def resize_image(image_pil):
      img_np = np.array(image_pil)
      resized = cv2.resize(img_np, (256, 256), interpolation=cv2.INTER_LANCZOS4)
      return resized.astype(np.uint8)

    # Iterate with tqdm progress bar
    for i, example in tqdm(enumerate(dataset), total=total_samples, desc=f"Resizing {split} split"):
      resized_img = resize_image(example['image'])
      yield str(i), {
          'image': resized_img,
          'label': example['label'],
      }
