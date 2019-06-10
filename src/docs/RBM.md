## Restricted Boltzmann Machine (RBM)

Here we have summarized documentation regarding the `train_RBM.py` function.

```
usage: train_RBM.py [-h] [--data DATA] [--learning-rate LEARNING_RATE]
                    [--k1 K1] [--k2 K2] [--epochs EPOCHS]
                    [--batch-size BATCH_SIZE] -d DIMENSIONS

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           data source to train RBM, possibilities are 'mnist',
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
                        consecutive enumeration of visible and hidden units
                        separated by a comma character, eg. 784,500
```

This script currently supports TensorFlow eager execution for easy debugging. For conversion to AutoGraph, minor modifications such as additions of `@tf.function` calls would need to be made. The script in `train_RBM.py` will train a corresponding RBM and write it as a pickle into a local log directory in `/src/pickles`. An example of running `train_RBM.py` is given below:

```shell
$ python3 train_RBM.py --epochs 2 --dimensions 784,500
```

### Mean-Field Sample Visualizations

Using a pre-trained RBM, we generated 100 random (mean-field) samples of MNIST images:

<img src="/img/rbm_mnist.png" width="800">
