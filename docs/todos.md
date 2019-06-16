## TODOs:

### Quick-fixes:
* fix relative pickles issue
* modify sample functions to allow plots with arbitrary number of samples

### RBM/derivatives
* output loss alongside gradient norm
* for DBN/DBM: consider removing data storage, rather have data regenerated upon sampling (less optimal but more compact)
* modify input data storage from numpy to tensorflow batches
* remove python datatypes such as lists and replace with higher dimensional tensors

### CVAE
* implement 2d-discriminator (requires labels)
* consider changing saved data dirname
* add NN schematics to github repo
* look into formulas here for log_pdf and justifications for log distributions

### CGAN
* generate learning pathways from text files
* generate latent manifold for CGAN with some tricks
* add documentation on how lowering learning rate for generator can add stability
* add profiler for relative learning rate adjustment ie. stacked adam for two optimizers
* warning about unresolved checkpoints when re-loading weights, could be attributed to learning rate issue
* fix autograph implementation in colab related to segmenting training to different variables

### GPU implementation
* improve RBM/DBM/DBN code with efficiencies for GPU computation, make amendments from feedback
* add autograph calls to code and test with Google GPU for optimization
