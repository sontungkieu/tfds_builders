"""celebahq256 dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf

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
            'image': tfds.features.Image(shape=(256, 256, 3)),
            'label': tfds.features.ClassLabel(names=['female', 'male']),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    # TODO(celebahqhq): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples('train'),
        'validation': self._generate_examples('validation'),
    }

  def _generate_examples(self, split):
    """Yields examples."""
    # TODO(celebahqhq): Yields (key, example) tuples from the dataset

    from datasets import load_dataset
    import numpy as np
    import tensorflow as tf  # Ensure TF is imported for tf.image

    dataset = load_dataset("mattymchen/celeba-hq", split=split)
    # Fix: Provide the required positional arguments to to_tf_dataset
    # columns: list of features to include (all available: 'image', 'label')
    # batch_size=1: Process one example at a time (avoids large batches)
    # shuffle=False: No shuffling for deterministic generation
    # collate_fn=None: Default collation (no custom needed for batch_size=1)
    dataset = dataset.to_tf_dataset(['image', 'label'], 1, False, None)

    def deserialization_fn(data):
        image = data['image']  # Shape: (1, H, W, 3) due to batch_size=1
        image = tf.image.resize(image, (256, 256), method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        # Squeeze the batch dim for label if needed, but since it's scalar per example, data['label'] is (1,)
        # Return as-is; we'll handle in yield
        return {'image': image, 'label': data['label']}

    dataset = dataset.map(deserialization_fn)
    dataset = tfds.as_numpy(dataset)
    for i, example in enumerate(dataset):
        # Since batched=1, squeeze the batch dimension
        yield i, {
            'image': example['image'][0].astype(np.uint8),  # Remove batch dim: from (1,256,256,3) to (256,256,3)
            'label': example['label'][0],  # Remove batch dim for scalar label
        }
