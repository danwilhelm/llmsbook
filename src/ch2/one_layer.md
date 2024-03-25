# A Single-Layered Network

## The Math

First, let's examine the math behind a single-layered neural network.

Suppose we are given $k$ inputs (scalars) and $N_l$ neurons. Then, the output of neuron $i$ is a linear combination of all $k$-dimensional inputs $\mathbf{x}$ with $k$-dimensional weights $\mathbf{w_i}$, plus a scalar bias $b_i$. An **activation function** $\sigma: \mathbb{N}^k \mapsto \mathbb{N}$ is then applied to the resulting sum:

$$
o_i = \sigma(\mathbf{x} \cdot \mathbf{w_i} + b_i).
$$

Alternatively, we can compute all outputs simultaneously with a single matrix multiplication:

$$
\mathbf{o} = \sigma(\mathbf{X}\mathbf{W} + \mathbf{b}).
$$

As we saw in Chapter 1, multiple linear transforms applied back-to-back can be collapsed into a single linear transform. Therefore, non-terminal layers use a non-linear $\sigma$. Depending on the function, this allows us to warp the space (e.g. the logistic function squishes all outputs into the range $[-1, 1]$), apply gating (e.g. the ReLU function "turns off" an output if the weighted input sum is negative), and more. 


## The feedforward block of a transformer

The feedforward block consists of two neuron layers, where the first layer has some $L$ neurons. It takes as input the residual stream, which is an $N$x$k$ matrix. (That is, $N$ tokens/sequence positions, each represented by a $k$-dimensional vector.) The first layer consists of $L$ neurons, each with a non-linear activation function, followed by a second layer of $k$ neurons with an identity activation function.
