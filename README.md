# Prototype of Differentiable Spiking Neural Net

## Explanation
Pending

## Check out `graph` branch for Graph Convolutional implementation.
## TODO:

### Algorithm
- [x] Backprop training
- [x] Gradient clipping
- [x] Deactivation of neurons after firing
- [ ] Weights between nodes (aka variable convolution filters)
- [ ] Vector-valued neurons
- [x] Thresholding
- [ ] Multiple input and output nodes (how to make sure the outputs all come in at the same time?)

### Infrastructure
- [ ] CLI
- [ ] train/test sets
- [ ] tensorboard and/or additional visualization. No one wants to look at command line matrices.

### Long-term
- [ ] Try to process images with DSNN
