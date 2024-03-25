# Attention

Attention is often presented as the most complex block in a transformer. It is called **attention** because it applies a weight to each prior token vector. To pay more attention to some prior tokens, we can apply a larger weight. Hence, attention is effectively a weighted mean of the input $X$.

For this problem we must compute the token frequencies. Luckily for us, these are found by taking the (uniformly-weighted) mean of the input $X$, where $X_i$ is the $i$th row:

$$\overline{X} = \sum_i{\frac{1}{N_\text{toks}} X_i}$$

> In the last section, we embedded the tokens `bab` as: $$X = \begin{vmatrix}0&1&0\\1&0&0\\0&1&0\end{vmatrix}.$$
> Now in the attention step, we take the weighted mean across the rows to compute the token frequencies: $\overline{X} = \frac{1}{3}(1, 2, 0)$.

For simplicity, we will assume the attention weights are uniform. In later chapters, we'll discuss the math behind computing the weights and show that uniform weights are possible.

### The residual stream

We mentioned in the last section that the residual stream always has shape $(N_\text{toks}, K)$ in both the input and output of attention.

Yet if attention takes the mean across all tokens, why doesn't this result in a $(1,K)$-dimensional output?

It turns out that attention takes the mean of each **increasing subset of tokens**. So the first output row will contain the mean of only the first token's embedding (i.e. itself). The second row will contain the mean of the first two embeddings, and so on.

Let's see a concrete example:

> Suppose we have the input embeddings of `'bab'`: $$X = \begin{vmatrix}0&1&0\\1&0&0\\0&1&0\end{vmatrix}.$$
> Then, the output of attention $Y$ is: $$Y = \begin{vmatrix}0&1&0\\1/2&1/2&0\\1/3&2/3&0\end{vmatrix}.$$
> The first row is the first row of $X$. The second row is the mean of the first _two_ rows of $X$. And the final row is the mean of all _three_ rows $\overline{X}$.

It is also possible to compute this efficiently using linear algebra:

> We define a triangular $(N_\text{toks}, N_\text{toks})$-shape weight matrix $W$. Row $i$ specifies how to weight the first $i$ token embeddings to compute the $i$th row of the output matrix $Y$: $$W = \begin{vmatrix}1&0&0\\1/2&1/2&0\\1/3&1/3&1/3\end{vmatrix}.$$ Note that row $i$ weights the first $i$ tokens equally with probability $1/i$. The final row of $W$ corresponds to $\overline{X}$, where each token embedding is weighted uniformly as $1/N_\text{toks}$.
>
> To apply the weight matrix to the token emebddings $X$, we use matrix multiplication: $$Y = WX = \begin{vmatrix}1&0&0\\1/2&1/2&0\\1/3&1/3&1/3\end{vmatrix}\begin{vmatrix}0&1&0\\1&0&0\\0&1&0\end{vmatrix} = \begin{vmatrix}0&1&0\\1/2&1/2&0\\1/3&2/3&0\end{vmatrix}.$$

In attention, each row of the weight matrix $W$ will always sum to 1. This is a consequence of how the matrix is computed. We'll use this fact below to simplify some of the math.

#### Python Implementation

```python
n_toks = X.shape[0]
W = np.tri(n_toks) * (1. / np.arange(1, n_toks+1)).reshape(-1,1)

# (n_toks, K) = (n_toks, n_toks) @ (n_toks, K)
Y = W @ X
```

### Technical Details

The attention block can do more than merely take an average of the inputs. However, this is a convenient first-approximation of its workings.

Below, we'll walk through under what conditions the attention block can be reduced to a direct weighted mean of its inputs.

1. In attention, the inputs first undergo a linear projection $V$ (**value projection**) plus bias $b_v$. We'll define this as $X' = VX + b_v$.

2. $X'$ is then weighted by $W$ and subjected to a second linear projection $O$ (**output projection**) plus bias $b_o$. We'll call this the attention block output $Y = (WX')O + b_o$.

> In many attention implementations, there is no second (output) projection. In this text, however, we are preparing the reader for **multi-headed attention**. In this, the purpose of the output projection is to provide a final projection after concatenating together the outputs of numerous heads (although the math below still applies!). We'll look at this in more depth in later chapters.

The two linear projections above are back-to-back, so they can be combined into a single effective transformation. By using the associative and distributive properties of matrix multiplication, we can (step-by-step) rewrite these two transformations as one in terms of the weighted inputs $WX$:

> $$
\begin{align}
Y = && (WX')O + b_o            \\
= && (W(XV + b_v))O + b_o      \\
= && (W(XV) + Wb_v)O + b_o     \\
= && ((WX)V + Wb_v)O + b_o     \\
= && ((WX)V)O + (Wb_v)O + b_o  \\
= && (WX)(VO) + (b_vO + b_o).  \\
\end{align}
$$
> The attention output is now in terms of the weighted _inputs_ $X$ rather than the weighted _transformed inputs_ $X'$!
> 
> Note we combined the two transforms above into a single linear transformation with bias (with projection $VO$ and bias $b_vO + b_o$).

To obtain the final step, we assert that $(Wb_v)O = b_vO$. To explain this, note that each row of $W$ sums to 1. Since $b_v$ is a $(1,K)$ matrix, it is broadcasted into a $(N_\text{toks},K)$ matrix, resulting in each row of $W$ being multiplied by a constant. For example, the first output position is $b_{v1} \cdot w_1 = b_{v1}(w_{11} + w_{12} + \dots + w_{1n}) = b_{v1}$. Therefore, $Wb_v = b_v$.

A few important observations:

1. Attention is typically described as weighting the _transformed_ inputs ($WX'$). However, we showed it can just as accurately be described as directly weighting the original inputs ($WX$).

2. Interestingly, we showed that the value biases are not needed! We equivalently can "fold" them into the output biases. Just let $b_v' = 0$ and $b_o' = b_vO + b_o$, giving a single transform $Y = (WX)(VO) + b_o'$. This identity is useful both for ease of interpretability and for reducing multiplications.

3. In this way, it becomes clear exactly how we can choose $V$, $O$, and $b_o$ to return solely uniformly-weighted inputs. Just set $V$ and $O$ such that $VO = I$ (the identity), then zero the effective output bias ($b_o' = \mathbf{0}$).


#### Python Implementation

```python
K = X.shape[1]    # num channels
VO = np.identity(K)
bo = np.zeros(K)

# (n_toks, K) = (n_toks, n_toks) @ (n_toks, K) @ (K, K) + (K,)
Y = W @ X @ VO + bo
```
