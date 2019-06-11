## Convolutional Generative Adverserial Network (CGAN)

Here we have summarized documentation regarding the `train_CGAN.py` function.

```
usage: train_CGAN.py [-h] [--data DATA] [--learning-rate LEARNING_RATE]
                     [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                     [--im-dim IM_DIM] [--num-filters NUM_FILTERS]
                     [--g-factor G_FACTOR] [--drop-rate DROP_RATE] -l
                     LATENT_DIMENSIONS

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           data source to train CGAN, possibilities are 'mnist',
                        'fashion_mnist' and 'faces' <default: 'mnist'>
  --learning-rate LEARNING_RATE
                        learning rate, <default: 0.001>
  --epochs EPOCHS       number of epochs for training <default: 5>
  --batch-size BATCH_SIZE
                        size of training data batches <default: 50>
  --im-dim IM_DIM       square dimensionality of input images <default: 28>
  --num-filters NUM_FILTERS
                        number of filters to be used in convolutional layers
                        <default: 32>
  --g-factor G_FACTOR   scalar multiple by which learning rate for the
                        generator is multiplied <default: 1>
  --drop-rate DROP_RATE
                        dropout rate for hidden dense layers <default: 0.5>

required named arguments:
  -l LATENT_DIMENSIONS, --latent-dimensions LATENT_DIMENSIONS
                        number of central latent dimensions in CGAN, 2
                        dimensions are recommended for quick manifold
                        visualization
```

This script currently supports TensorFlow eager execution for easy debugging. For conversion to AutoGraph, minor modifications such as additions of `@tf.function` calls would need to be made. The script in `train_CGAN.py` will train a corresponding CGAN and save its weights to `/src/pickles`. An example of running `train_CGAN.py` is given below:

```shell
$ python3 train_CGAN.py --epochs 100 --latent-dimensions 2
```

Pre-trained CGAN weights for MNIST and fashion-MNIST can be found in the `/src/pickles` directory.

**Note:** Due to a bug, loading these weights back into a tf.keras model results in warnings for unresolved objects. The models can still be used and the bug will be fixed in a later commit.

### CGAN Samples

Using a pre-trained CGAN, we generated 100 samples from MNIST images:

<img src="/img/cgan_mnist.png" width="800">

Similarly, we generated 100 samples from fashion-MNIST images:

<img src="/img/cgan_fashion_mnist.png" width="800">
