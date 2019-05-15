## Deep Belief Network (DBN)

Here we have summarized documentation regarding the `train_DBN.py` function.

```
usage: train_DBN.py [-h] [--data DATA] [--learning-rate LEARNING_RATE]
                    [--k1 K1] [--k2 K2] [--k3 K3] [--epochs EPOCHS]
                    [--batch-size BATCH_SIZE] -d DIMENSIONS

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           data source to train DBN, possibilities are 'mnist',
                        'fashion_mnist' and 'faces', defaults to 'mnist'
  --learning-rate LEARNING_RATE
                        learning rate for stacked RBMs, defaults to 0.01
  --k1 K1               number of Gibbs-sampling steps pre-PCD-k algorithm,
                        defaults to 1
  --k2 K2               number of Gibbs-sampling steps during PCD-k algorithm,
                        defaults to 5
  --epochs EPOCHS       number of overall training data passes for each RBM,
                        defaults to 1
  --batch-size BATCH_SIZE
                        size of training data batches, defaults to 5

required named arguments:
  -d DIMENSIONS, --dimensions DIMENSIONS
                        consecutive enumeration of visible and hidden layers
                        separated by a comma character, eg. 784,500,500,1000
```

This script currently supports TensorFlow eager execution for easy debugging. For conversion to AutoGraph, minor modifications such as additions of `@tf.function` calls would need to be made. The script in `train_DBN.py` will train a corresponding DBN and write it as a pickle into a local log directory in `/src/pickles`. An example of running `train_DBN.py` is given below:

```shell
$ python3 train_DBN.py --epochs 2 --dimensions 784,500,784
```

Pre-trained DBNs for MNIST, fashion-MNIST and LFW cropped faces have been saved in the `/src/pickles` directory.

### Mean-Field Sample Visualizations

Using the pre-trained DBN, we generated 100 random (mean-field) samples of MNIST images:

<img src="/img/sample.png" width="800">

The same process was done for 100 random (mean-field) samples of fashion-MNIST images:

<img src="/img/sample2.png" width="800">

The same process was done for 100 random (mean-field) samples of LFW cropped-face images. Here, the results of generating samples was not as clear-cut as per MNIST and fashion-MNIST. This is due to the lack of sparseness in face vectors. As a result, we had to force sparseness in the vectors by converting pixels with low intensities to zero.

<img src="/img/sample18.png" width="800">
