## Deep Generative Models using TensorFlow 2.0 Beta

This repository offers source-code for training Restricted Boltzmann Machines (RBM), Deep Belief Networks (DBN) and Deep Boltzmann Machines (DBM), Convolutional Variational Auto-Encoders (CVAE) and Convolutional Generative Adverserial Networks (CGAN). This code has default functionalities for MNIST, fashion-MNIST and [Labeled Faces in the Wild](http://conradsanderson.id.au/lfwcrop/) (LFW) cropped input data. However, the code and its internal classes can principally be used for other data, without any particular hard-coding to the above-specified datasets.

In order to download/deploy the LFW cropped grayscale data, you can execute the following:

```shell
$ cd ./data/ && ./lfw_setup.sh
```

### 1. Restricted Boltzmann Machine (RBM)

Further information on training a RBM can be found in this [guide](/src/docs/RBM.md).

### 2. Deep Belief Network (DBN)

Further information on training a DBN can be found in this [guide](/src/docs/DBN.md).

### 3. Deep Boltzmann Machine (DBM)

Further information on training a DBM can be found in this [guide](/src/docs/DBM.md).

### 4. Convolutional Variational Auto-Encoder (CVAE)

Further information on training a CVAE can be found in this [guide](/src/docs/CVAE.md).

### 5. Convolutional Generative Adverserial Network (CGAN)

Further information on training a CGAN can be found in this [guide](/src/docs/CGAN.md).

### Literature

Algorithms in this source-code for DBM/DBN/RBMs were adapted from Salakhutdinov and Hinton (2009) with the following bibtex citation:

```
@inproceedings{salakhutdinov2009deep,
  title={Deep boltzmann machines},
  author={Salakhutdinov, Ruslan and Hinton, Geoffrey},
  booktitle={Artificial intelligence and statistics},
  pages={448--455},
  year={2009}
}
```

### Developments

Further developments are summarized in a local [change log](/docs/todos.md).
