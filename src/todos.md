## TODOs:

### current architecture
* attempt PCD-k with using previous iterations, figure out true difference between CD-k and PCD-k
* compute log-likelihood and output for idea of progress

### new features
* implement gradient descent optimizer for better training, eg. adam
* implement DBM training and sampling
* implement wake-sleep algorithm for fine-tuning
* addition of supervised fine-tuning

### tf2 autograph integration
* add autograph call to DBN/DBM and test with Google GPU for optimization
