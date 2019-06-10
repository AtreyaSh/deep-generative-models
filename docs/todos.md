## TODOs:

### quick-fixes:
* fix relative pickles issue
* modify sample functions to allow plots with arbitrary number of samples

### RBM architecture
* output loss alongside gradient norm
* consider removing data storage, rather have data regenerated upon sampling (less optimal but more compact)

### VAE
* implement 2d-discriminator (requires labels)
* consider changing saved data dirname
* add NN schematics to github repo
* look into formulas here for log_pdf and justifications for log distributions

### CGAN
* add documentation, somehow lower learning rate for generator can add stability
* add profiler for relative learning rate adjustment ie. stacked adam for two optimizers

### tf2 autograph integration
* add autograph call to DBN/DBM and test with Google GPU for optimization
* improve RBM/DBM/DBN code with efficiencies for GPU computation, make feedback amendments for speed boost
