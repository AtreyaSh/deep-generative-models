## Deep Boltzmann Machine (DBM)

Here we have summarized documentation regarding the `train_DBM.py` function.

```
usage: train_DBM.py [-h] [--data DATA] [--learning-rate LEARNING_RATE]
                    [--k1 K1] [--k2 K2] [--epochs EPOCHS]
                    [--batch-size BATCH_SIZE] -d DIMENSIONS

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           data source to train DBM, possibilities are 'mnist',
                        'fashion_mnist' and 'faces' <default: 'mnist'>
  --learning-rate LEARNING_RATE
                        learning rate for stacked RBMs <default: 0.01>
  --k1 K1               number of Gibbs-sampling steps pre-PCD-k algorithm
                        <default: 1>
  --k2 K2               number of Gibbs-sampling steps during PCD-k algorithm
                        <default: 5>
  --epochs EPOCHS       number of overall training data passes for each RBM
                        <default: 1>
  --batch-size BATCH_SIZE
                        size of training data batches <default: 5>

required named arguments:
  -d DIMENSIONS, --dimensions DIMENSIONS
                        consecutive enumeration of visible and hidden layers
                        separated by a comma character, eg. 784,500,784,500
```

This script currently supports TensorFlow eager execution for easy debugging. For conversion to AutoGraph, minor modifications such as additions of `@tf.function` calls would need to be made. The script in `train_DBM.py` will train a corresponding DBM and write it as a pickle into a local log directory in `/src/pickles`. An example of running `train_DBM.py` is given below:

```shell
$ python3 train_DBM.py --epochs 2 --dimensions 784,500,784
```

Pre-trained DBMs are not offered in this repository due to large memory requirements.

### Mean-Field Sample Visualizations

Using a pre-trained DBM, we generated 100 random (mean-field) samples of MNIST images:

<img src="/img/sample19.png" width="800">

Similarly, we generated 100 random (mean-field) samples of fashion-MNIST images:

<img src="/img/sample22.png" width="800">

The same process was done for 100 random (mean-field) samples of LFW cropped-face images. Here, the results of generating samples was not as clear-cut as per MNIST and fashion-MNIST. This is due to the lack of sparseness in face vectors. As a result, we had to force sparseness in the vectors by converting pixels with low intensities to zero.

<img src="/img/sample24.png" width="800">
