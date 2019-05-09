## TODOs:

### current architecture
* offer two downward options: binary downward sampling, alternatively use probability for mean-field implementation
* attempt PCD-k with using previous iterations, figure out true difference between CD-k and PCD-k
* make auto class update script to update old classes to new classes

### new features
* implement wake-sleep algorithm for fine-tuning
* addition of supervised fine-tuning

### tf2 autograph integration
* add autograph call to DBM and test with Google GPU for optimization
* fix lens with shapes in autograph
* figure out progress log with tensorflow autograph
