"""lsun dataset."""

import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for lsun dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(None, None, 3)),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    # TODO(celebahq): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'church-train': self._generate_examples('train'),
        'church-test': self._generate_examples('test'),
    }

  def _generate_examples(self, split):
    """Yields examples."""

    from datasets import load_dataset
    import numpy as np
    dataset = load_dataset("tglcourse/lsun_church_train", split=split)
    dataset = dataset.to_tf_dataset()
    dataset = tfds.as_numpy(dataset)
    for i, example in enumerate(dataset):
      yield i, {
          'image': example['image'].astype(np.uint8),
      }
