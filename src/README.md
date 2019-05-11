## Deep Generative Models using TensorFlow 2.0 Alpha

### Deep Boltzmann Machine (DBM)

The file `train_DBM.py` includes source-code for training and writing a Deep Boltzmann Machine. This code has default functionalities for MNIST, fashion-MNIST and [Labeled Faces in the Wild](http://conradsanderson.id.au/lfwcrop/) (LFW) cropped input data. However, the code and its internal classes can principally be used for other data, without any particular hard-coding to the above-specified datasets.

In order to download/deploy the LFW cropped grayscale data, you can execute the following:

```shell
$ wget http://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip && unzip lfwcrop_grey.zip -d ./data/

$ mv lfwcrop_grey.zip ./data && cd ./data && ln -s lfwcrop_grey/faces .
```

Here we have summarized documentation regarding the `train_DBM.py` function.

```
usage: train_DBM.py [-h] [--data DATA] [--learning-rate LEARNING_RATE]
                    [--k1 K1] [--k2 K2] [--k3 K3] [--epochs EPOCHS]
                    [--batch-size BATCH_SIZE] -d DIMENSIONS

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           data source to train DBM, possibilities are 'mnist',
                        'fashion_mnist' and 'faces', defaults to 'mnist'
  --learning-rate LEARNING_RATE
                        learning rate for stacked RBMs, defaults to 0.01
  --k1 K1               number of Gibbs-sampling steps pre-PCD-k algorithm,
                        defaults to 1
  --k2 K2               number of Gibbs-sampling steps during PCD-k algorithm,
                        defaults to 5
  --k3 K3               number of Gibbs-sampling steps before transferring
                        samples to next model, defaults to 5
  --epochs EPOCHS       number of overall training data passes for each RBM,
                        defaults to 1
  --batch-size BATCH_SIZE
                        size of training data batches, defaults to 5

required named arguments:
  -d DIMENSIONS, --dimensions DIMENSIONS
                        consecutive enumeration of visible and hidden layers
                        separated by a comma character, eg. 784,500,500,1000
```

This script currently supports TensorFlow eager execution for easy debugging. For conversion to AutoGraph, minor modifications such as additions of `@tf.function` calls would need to be made. The script in `train_DBM.py` will train a corresponding DBM and write it as a pickle into a local log directory in `/src/pickles`. An example of running `train_DBM.py` is given below:

```shell
$ python3 train_DBM.py --epochs 2 --dimensions 784,500,500,1000
```

Pre-trained DBMs for MNIST, fashion-MNIST and LFW cropped faces been saved in the `/src/pickles` directory.

### Mean-Field Sample Visualizations

Using the pre-trained DBM, we generated 100 random (mean-field) samples of MNIST images:

<img src="/img/sample.png" width="800">

The same process was done for 100 random (mean-field) samples of fashion-MNIST images:

<img src="/img/sample2.png" width="800">

The same process was done for 100 random (mean-field) samples of LFW cropped-face images. Here, the results of generating samples was not as clear-cut as per MNIST and fashion-MNIST. This is due to the lack of sparseness in face vectors. As a result, we had to force sparseness in the vectors by converting pixels with low intensities to zero.

<img src="/img/sample18.png" width="800">

## Developments

Further developments are summarized in a local [change log](/src/todos.md).
