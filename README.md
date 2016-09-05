### Code for "Learning to Reason with Adaptive Computation"
#### MSc Thesis, Mark Neumann

Before you can run the code in this repo, you need to download the SNLI data from [here](http://nlp.stanford.edu/projects/snli/). Additionally, you will need to alter run.sh to include the data_path argument to where you have saved the data. 

This code uses one of the daily binary tensorflow releases, which you can download [here](https://github.com/tensorflow/tensorflow). There's no reason why it won't work with the standard release, you might just have to do a bit of refactoring.

This code implements a couple of papers:

[Alternating Iterative Neural Attention for Machine Reading](http://arxiv.org/abs/1606.02245)
[A Decomposable Attention model for Natural Language Inference](https://arxiv.org/abs/1606.01933)
[Adaptive Computation Time for Recurrent Neural Networks](http://arxiv.org/abs/1603.08983)

These are then combined to make a couple of different models for Adaptive Natural Language Inference.

1. ACTDAModel.py : Decomposable Attention + ACT + Iterative Neural Attention inference module
2. AdaptiveIAAModel.py GRU Sentence encoders + ACT + Iterative Neural Attention inference module
3. DAModel.py : Decomposable Attention model
4. IAAModel.py : Iterative Neural Attention with fixed inference GRU steps

