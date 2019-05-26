## Variational Auto-Encoder (VAE)

Here we have summarized documentation regarding the `train_VAE.py` function. Note that this script has been adapted and modified from the TensorFlow2 tutorial [here](https://www.tensorflow.org/alpha/tutorials/generative/cvae)
```
usage: train_VAE.py [-h] [--data DATA] [--learning-rate LEARNING_RATE]
                    [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                    [--im-dim IM_DIM] [--num-filters NUM_FILTERS] -l
                    LATENT_DIMENSIONS

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           data source to train VAE, possibilities are 'mnist',
                        'fashion_mnist' and 'faces', defaults to 'mnist'
  --learning-rate LEARNING_RATE
                        learning rate, defaults to 0.001
  --epochs EPOCHS       number of epochs for training, defaults to 5
  --batch-size BATCH_SIZE
                        size of training data batches, defaults to 50
  --im-dim IM_DIM       square dimensionality of input images, defaults to 28
  --num-filters NUM_FILTERS
                        number of filters to be used in convolutional layers,
                        defaults to 32

required named arguments:
  -l LATENT_DIMENSIONS, --latent-dimensions LATENT_DIMENSIONS
                        number of central latent dimensions in VAE, 2
                        dimensions are recommended for quick manifold
                        visualization
```

This script currently supports TensorFlow eager execution for easy debugging. For conversion to AutoGraph, minor modifications such as additions of `@tf.function` calls would need to be made. The script in `train_VAE.py` will train a corresponding VAE and save its weights to `/src/pickles`. An example of running `train_VAE.py` is given below:

```shell
$ python3 train_VAE.py --epochs 10 --latent-dimensions 2
```

Pre-trained VAE weights for MNIST, fashion-MNIST and LFW cropped greyscale faces can be found in the `/src/pickles` directory.

### VAE Latent Manifold Visualization

Using a pre-trained VAE, we generated 1600 MNIST images derived from the VAE latent manifold:

<img src="/img/vae_mnist.png" width="800">

Similarly, we generated 1600 manifold samples of fashion-MNIST images:

<img src="/img/vae_fashion_mnist.png" width="800">

The same process was done for 1600 samples of LFW cropped-face images. The image below shows variation on a 2-dimensional latent space.

<img src="/img/vae_faces_2.png" width="800">

The image below shows 2-dimensional projected variation on a 6-dimensional latent space.

<img src="/img/vae_faces_6.png" width="800">

The image below shows 2-dimensional projected variation on a 10-dimensional latent space.

<img src="/img/vae_faces_10.png" width="800">

### VAE Latent Manifold Animation

To visualize the smoothness of changes in the latent manifold, we animated the above plots in a spiral sequence.

**MNIST:**

<img src="/img/vae_mnist.gif" width="250">

**Fashion-MNIST:**

<img src="/img/vae_fashion_mnist.gif" width="250">

**LFW Faces, 2-d latent space:**

<img src="/img/vae_faces_2.gif" width="250">

**LFW Faces, 6-d latent space:**

<img src="/img/vae_faces_6.gif" width="250">

**LFW Faces, 10-d latent space:**

<img src="/img/vae_faces_10.gif" width="250">
