# Embeddings

A transformer takes as input a list of **tokens**. In our case, the vocabulary $V = \{\text{a}, \text{b}, \dots, \text{z}, \text{"space"}\}$ comprise our $|V| = 27$ tokens. That said, a token does not have to be a single character. It can represent chunks of characters or even more abstract concepts such as the start or end of text.

The first transformer operation -- the embedding -- takes each token and replaces it with a vector of $K$ numbers. These $K$ positions are often called **channels** since they convey information.

### The residual stream

The output of the embedding stage is a matrix with each row representing a token, and each column representing a channel. This gives the resulting matrix a **shape** of $(N_\text{toks}, K)$. This matrix begins the transformer's **residual stream**, which is read from and written to as it passes through each stage of the transformer.

### One-hot encoding

How should we represent each of the possible $|V|$ tokens? Recall our objective is to compute the ciphertext letter frequencies and that each letter is a token. So, a reasonable start would be to  dedicate a single channel per token. If the channel $i$ contains $1$, it indicates token $i$; if channel $i$ contains $0$, it indicates it is not token $i$.

This encoding scheme is called a **one-hot encoding**.

> For example, suppose we one-hot encode three tokens `a`, `b`, and `c` using $K=3$ channels. We'll map `a = [1, 0, 0]`, `b = [0, 1, 0]`, and `c = [0, 0, 1]`. (Stacked, these comprise the **embedding matrix**, which here is the identity matrix.)
> 
> The $(N_\text{toks}, K) = (3, 3)$ embedding of the string `abc` provides our input to the model $X$: $$X = \begin{vmatrix}0&1&0\\1&0&0\\0&1&0\end{vmatrix}.$$
> Note this has a convenient property -- we can count the letters by summing across the rows! This gives the vector $(N_a, N_b, N_c) = (1, 2, 0)$.

Typically embeddings are not one-hot encoded, since this requires $K \geq N_\text{tok}$. However, it also neglects an opportunity to store extra information per token. For example, we could dedicate a channel to indicate even vs. odd, or we could ensure that "similar" tokens are nearby in space.

At the end of this chapter, we'll show that nearly _any_ embedding matrix can solve our cryptograms! This is what makes interpretability so difficult. To us, a one-hot encoding makes the algorithm easy to understand. However, a computer is likely to choose an arbitrary embedding which obscures the underlying algorithm.
