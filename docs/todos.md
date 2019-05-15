## TODOs:

### qns:
* for DBM, save samples during training to use later
* or update them during each gibbs sampling

### quick-fixes:
* implement block Gibbs sampling
* implement gradient descent optimizer for better training, eg. adam, and add to all architectures
* shuffle samples in each epoch
* fix relative pickles issue

### RBM architecture
* attempt PCD-k with using previous iterations, figure out true difference between CD-k and PCD-k
* update RBM learning process based on newer techniques, improve from old method
* compute log-likelihood and output for idea of progress

### new architectures/features
* implement DBM training and sampling
* consider whether gibbs sampling is required for passing samples to next model
* implement wake-sleep algorithm for fine-tuning
* addition of supervised fine-tuning

### tf2 autograph integration
* add autograph call to DBN/DBM and test with Google GPU for optimization
* install tf2 on google colab with gpu argument
