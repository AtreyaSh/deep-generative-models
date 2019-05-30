## TODOs:

### VAE
* implement 2d-discriminator (requires labels)
* consider changing saved data dirname
* add NN schematics to github repo
* look into formulas here for log_pdf and justifications for log distributions

### quick-fixes:
* fix relative pickles issue
* modify sample functions to allow plots with arbitrary number of samples

### RBM architecture
* output loss alongside gradient norm
* consider removing data storage, rather have data regenerated upon sampling (less optimal but more compact)

### new architectures/features
* implement wake-sleep algorithm for fine-tuning
* addition of supervised fine-tuning

### tf2 autograph integration
* add autograph call to DBN/DBM and test with Google GPU for optimization
