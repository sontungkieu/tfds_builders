# tfds_builders
A set of TFDS dataset builders for common datasets

To build these datasets, first install `tfds`. Then, `cd` into each directory and run `tfds build`.

For `imagenet2012`, this is the code from the official [TFDS repository](https://www.tensorflow.org/datasets/catalog/imagenet2012). You will need to first manually download imagenet from the Imagenet website. Download the training and validation sets, and make sure they are in the `downloads` folder of your TFDS data directory. 