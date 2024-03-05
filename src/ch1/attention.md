# Attention

Attention is often presented as the most complex block in a transformer. It is called **attention** because it applies a weight to each prior token vector. To pay more attention to some prior tokens, we can apply a larger weight. Hence, attention is best thought of as taking a weighted mean of the input $X$.

For this problem we want the token frequencies, which are proportional to the sum of each channel across the tokens. To find this we'll make the weights uniform, giving us the mean:

$$\overline{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$$

> We embed the three tokens `bab` as $X = \begin{vmatrix}0&1&0\\1&0&0\\0&1&0\end{vmatrix}$. In the attention step, we sum across the rows to get the token frequencies $\frac{1}{3}$`[1, 2, 0]` -- one `a` and two `b`s.

For simplicity, in this chapter we will not discuss how the weights are determined and just assume they are uniform.

### Residual stream

The input into attention is part of the **residual stream**. We can imagine that attention reads from and writes to this stream.

The information on the residual stream _always_ has dimensions (n_toks, $K$), i.e. the same as our embeddings $X$.

This seems to present a contradiction. If attention takes the mean across all tokens, why doesn't this result in a $(1,K)$-dimensional matrix?

In reality, transformer circuits make a prediction **for each token**. Hence in our case, attention will output a weighted mean for each **increasing subset of tokens**. So the first row will contain the mean of only the first token's embedding. The second row will contain the mean of the first two embeddings, and so on.

Let's see a concrete example:

> Suppose we have the input embeddings of `'bab'`: $$X = \begin{vmatrix}0&1&0\\1&0&0\\0&1&0\end{vmatrix}.$$ In attention, the mean of all three rows will be computed and output as the final row of $Y$. The first row of $Y$ will then be the mean of only the first embedding. And the second row of $Y$ will be the mean of only the first two embeddings, as follows: $$Y = \begin{vmatrix}0&1&0\\1/2&1/2&0\\1/3&2/3&0\end{vmatrix}.$$

> To compute this using linear algebra, we define a triangular (n_toks, n_toks) weight matrix $W$ containing uniform probabilities: $$W = \begin{vmatrix}1&0&0\\1/2&1/2&0\\1/3&1/3&1/3\end{vmatrix}.$$ Note that we are weighting each token equally (e.g. the final row weights all three tokens with probability 1/3 each). Then, we use matrix multiplication: $$Y = WX = \begin{vmatrix}1&0&0\\1/2&1/2&0\\1/3&1/3&1/3\end{vmatrix}\begin{vmatrix}0&1&0\\1&0&0\\0&1&0\end{vmatrix} = \begin{vmatrix}0&1&0\\1/2&1/2&0\\1/3&2/3&0\end{vmatrix}.$$

#### Python Implementation

```python
n_toks = X.shape[0]
W = np.tri(n_toks) * (1. / np.arange(1, n_toks+1)).reshape(-1,1)

# (n_toks, K) = (n_toks, n_toks) @ (n_toks, K)
Y = W @ X
```

### Technical Details

There are more details to attention, but they are not entirely needed to understand the rest of the chapter. If curious, below we'll show why in our case attention simply can be defined as the weighted inputs:

1. In attention, the inputs first undergo a linear projection $V$ (**value projection**) plus bias $b_v$. We'll define this as $X' = VX + b_v$.

2. $X'$ is then weighted by $W$ and subjected to a second linear projection $O$ (**output projection**) plus bias $b_o$. We'll call this the attention block output $Y = (WX')O + b_o$.

> In many attention implementations, there is no second (output) projection. In this text, however, we are preparing the reader for **multi-headed attention**. In this, the purpose of the output projection is to provide a final projection after concatenating together the outputs of numerous heads (although the math below still applies!). We'll look at this in more depth in later chapters.

Since the above is two linear transformations back-to-back, they can be combined into a single effective transformation. By using the associative and distributive properties of matrix multiplication, we can (pedantically) rewrite these two transformations as one in terms of the weighted inputs $WX$:

$$
\begin{align}
Y = && (WX')O + b_o            \\
= && (W(XV + b_v))O + b_o      \\
= && (W(XV) + Wb_v)O + b_o     \\
= && ((WX)V + Wb_v)O + b_o     \\
= && ((WX)V)O + (Wb_v)O + b_o  \\
= && (WX)(VO) + (b_vO + b_o).  \\
\end{align}
$$

In the final step, we use associativity then assert that $(Wb_v)O = b_vO$. To explain the latter, note that each row of $W$ sums to 1. Since $b_v$ is a $(1,K)$ matrix, it is broadcasted into a (n_toks,$K$) matrix, resulting in each row of $W$ being multiplied by a constant. For example, the first output position is $b_{v1} \cdot w_1 = b_{v1}(w_{11} + w_{12} + \dots + w_{1n}) = b_{v1}$. Therefore, $Wb_v = b_v$.

A few important observations:

1. $W$ is typically applied to the _transformed_ inputs $WX'$. However, we showed it can be identically thought of as directly weighting the inputs $WX$.

2. Interestingly, we showed that the value biases are not needed! We can identically "fold" them into the output biases. Just let $b_v' = 0$, and $b_o' = b_vO + b_o$, giving a single transform $Y = (WX)(VO) + b_o'$. This identity is useful both for ease of interpretability and for reducing multiplications.

3. In this way, it becomes clear the output of attention simply can be the weighted inputs $Y = WX$. Just set $V$ and $O$ such that $VO = I$ (the identity), then set the effective output bias to 0s ($b_o' = \mathbf{0}$).


#### Python Implementation

```python
K = X.shape[1]    # num channels
VO = np.identity(K)
bo = np.zeros(K)

# (n_toks, K) = (n_toks, n_toks) @ (n_toks, K) @ (K, K) + (K,)
Y = W @ X @ VO + bo
```
