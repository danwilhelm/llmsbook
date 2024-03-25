# Introduction

Our thesis is that understanding small circuits leads to techniques for understanding large circuits.

## What is "understanding"?

Unfortunately, machine learning educators often introduce neural networks as "black box models." As an educator myself, I've found this often dissuades students from believing an understanding _could_ exist. So, what do we mean by "understanding"?

- First, neural networks are well-known to (often poorly) approximate some "true" function. As we'll see in chapter 3, a six-neuron neural network is required to approximate a modified square wave. Although the neural net describes a complex non-linear equation, the "true" structure is a simple square wave. By reconstructing this human-understandable "true" structure, we represent the problem simply and gain an "understanding."

- Second, from engineering we know that complex systems work because they have guiding principles of design and operation. For example, in chapter 1 we will create a small circuit that solves cryptograms. There are only so many algorithms for solving cryptograms, namely by frequency analysis of the letters. Therefore, any circuit that does solve them must use one of these algorithms. By knowing which algorithm is used and the implementing mechanism, we have gained an "understanding" of the circuit.

- Finally, we ascribe to the adage that we only demonstrate true understanding if we can build something from scratch. Toward this, we often will analyze pre-trained circuits to discover their principles of operation. Using these, we'll attempt to reconstruct the weights by hand. If we obtain a similar output, then we will claim to have "understood" the circuit.


## Acknowledgements

The first chapter's Caesar cipher problem was originally posed by Callum McDougall as an interpretability challenge as part of his <a href="https://github.com/callummcdougall/ARENA_3.0">ARENA bootcamp</a>. For more practice, you can explore his <a href="https://colab.research.google.com/drive/1pW1qAd52ZRf6gU-fTORjVyVKHuoLzJzH">PyTorch-trained transformer model</a> of the same problem.
