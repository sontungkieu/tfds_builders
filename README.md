# tfds_builders
A set of TFDS dataset builders for common datasets

To build these datasets, first install `tfds`. Then, `cd` into each directory and run `tfds build`.

For `imagenet2012`, this is the code from the official [TFDS repository](https://www.tensorflow.org/datasets/catalog/imagenet2012). You will need to first manually download imagenet from the Imagenet website. Download the training and validation sets, and make sure they are in the `downloads` folder of your TFDS data directory. 

A useful thing to do is to set `$TFDS_DATA_DIR$` to a shared directory if working with a cluster of machines. I like to set it to a GCP bucket, e.g.
```
export TFDS_DATA_DIR=gs://{bucket-name}/tfds
```
