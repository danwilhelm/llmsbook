# Embeddings

A transformer takes as input a list of **tokens**. In our case, the letters `a-z` and the space character comprise our 27 tokens. However, a token can represent chunks of characters or even more abstract concepts such as the start or end of text.

The first embedding stage takes each token and replaces it with a vector of $K$ numbers. These $K$ positions are often called **channels**, since they are read and written to as they pass through each stage of the transformer. Therefore, the output of the embedding stage is a matrix with each row representing a token, and each column representing a channel. We will write its shape as the tuple (n_toks, $K$).

### One-hot encoding

For this problem, we want to count the number of times each token occurs. The most convenient representation for this is to dedicate a channel per token, called a **one-hot encoding**. To count the number of tokens in the attention block, we can just sum across the tokens!

> For example, suppose we one-hot encode three tokens `a`, `b`, and `c` using $K=3$ channels as `a = [1, 0, 0]`, `b = [0, 1, 0]`, and `c = [0, 0, 1]`. We typically represent these embeddings as the **embedding matrix** $E = \begin{vmatrix}1&0&0\\0&1&0\\0&0&1\end{vmatrix}$, with each token corresponding to its respective row. We can then embed the three tokens `bab` as $X = \begin{vmatrix}0&1&0\\1&0&0\\0&1&0\end{vmatrix}$.

Embeddings are typically not one-hot encoded, since this disallows storing additional information. For example, we could dedicate a channel to even/odd, or we could ensure that "similar" tokens are nearby in space.

At the end of this chapter, we'll show that nearly _any_ embedding matrix can be used to solve this problem! This is what makes interpretability so difficult. To us, a one-hot encoding makes the algorithm clearest. However, a computer is likely to choose an arbitrary embedding which obscures the underlying algorithm.

#### Python Implementation

As implied above, the one-hot encoding is the identity matrix! After defining this, we embed each input token (recall **rotnums** are a list of indexes into `VOCAB`). This gives us the input to our transformer $X$:

```python
N_VOCAB = len(CaesarDataset.VOCAB)
K = n_vocab  # later we can experiment with larger K

embeds = np.identity(K)[:N_VOCAB]  # assuming K >= n_vocab
X = embeds[rotnums]
```
