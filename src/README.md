## Deep Generative Models using TensorFlow 2.0 Alpha

### Deep Boltzmann Machine (DBM)

The file `train_DBM.py` includes source-code for training and writing a Deep Boltzmann Machine. By default, this code has been written for MNIST data. However, the code and its internal classes can principally be used for other data, without any particular preference being hard-coded for MNIST.

```
usage: train_DBM.py [-h] [--learning-rate LEARNING_RATE] [--k1 K1] [--k2 K2]
                    [--k3 K3] [--epochs EPOCHS] [--batch-size BATCH_SIZE] -d
                    DIMENSIONS

optional arguments:
  -h, --help            show this help message and exit
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

A pre-trained DBM with dimensions `784,500,500,1000` has been saved in the `/src/pickles` directory.

### Sample Visualizations

Using the pre-trained DBM, we generated new samples of 100 random MNIST test images:

<img src="/img/sample.png" width="800">

Below are generated samples of specific MNIST test images:

<img src="/img/two.png" width="800">

<img src="/img/three.png" width="800">

<img src="/img/four.png" width="800">

<img src="/img/six.png" width="800">

<img src="/img/nine.png" width="800">
