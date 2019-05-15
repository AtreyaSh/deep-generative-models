## Deep Generative Models using TensorFlow 2.0 Alpha

This repository offers source-code for training Deep Belief Networks (DBN) Deep Boltzmann Machines (DBM). This code has default functionalities for MNIST, fashion-MNIST and [Labeled Faces in the Wild](http://conradsanderson.id.au/lfwcrop/) (LFW) cropped input data. However, the code and its internal classes can principally be used for other data, without any particular hard-coding to the above-specified datasets.

In order to download/deploy the LFW cropped grayscale data, you can execute the following:

```shell
$ cd ./data/ && ./lfw_setup.sh
```

### Deep Belief Network (DBN)

Further information on training a DBN can be found in this [guide](/src/docs/DBN.md).

### Deep Boltzmann Machine (DBM)

Further information on training a DBM can be found in this [guide](/src/docs/DBM.md).

## Literature

Algorithms in this source-code were adapted from Salakhutdinov and Hinton, 2009; bibtex citation can be found below.

```
@inproceedings{salakhutdinov2009deep,
  title={Deep boltzmann machines},
  author={Salakhutdinov, Ruslan and Hinton, Geoffrey},
  booktitle={Artificial intelligence and statistics},
  pages={448--455},
  year={2009}
}
```

## Developments

Further developments are summarized in a local [change log](/docs/todos.md).
