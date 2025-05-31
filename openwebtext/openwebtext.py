"""openwebtext dataset."""

import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for openwebtext dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'text': tfds.features.Text(),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    return {
        'train': self._generate_examples('train'),
    }

  def _generate_examples(self, split):
    """Yields examples."""

    from datasets import load_dataset
    import numpy as np
    dataset = load_dataset("Skylion007/openwebtext", split=split, cache_dir='/home/kvfrans/gcs/hf_cache')
    dataset = dataset.to_tf_dataset()
    dataset = tfds.as_numpy(dataset)
    for i, example in enumerate(dataset):
      yield i, {
          'text': example['text'],
      }