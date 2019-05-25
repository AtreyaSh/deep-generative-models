## TODOs:

### quick-fixes:
* shuffle samples in each epoch
* fix relative pickles issue
* modify sample functions to allow plots with arbitrary number of samples

### RBM architecture
* compute log-likelihood and output for idea of progress

### new architectures/features
* implement wake-sleep algorithm for fine-tuning
* addition of supervised fine-tuning

### tf2 autograph integration
* add autograph call to DBN/DBM and test with Google GPU for optimization

### VAE
* generate smooth transition gif
* implement 2d-discriminator (requires labels)
* consider changing saved data dirname
* add NN schematics to github repo
* look into formulas here for log_pdf
