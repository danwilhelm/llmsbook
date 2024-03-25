# Unembeddings

Finally, we must predict the rotation of the entire ciphertext. To do this, we'll use the technique from the first section.

1. First, we'll take the dot product of the final residual stream row ($\overline{X}$) with each rotation's expected frequency distribution. 
2. Then, the largest dot product (called a **logit**) will indicate the most likely rotation class.

Let's continue the example from the first section:

> Suppose for rotation 0 we expect frequencies for the letters `e` and `b` to be $(0.127, 0.015)$, respectively. As we discussed in the first section, for rotation 5 we expect the `e` and `b` frequencies to be $(0.0015, 0.061)$.
> 
> In our ciphertext, the respective frequencies are $(e, b)$. This makes the dot product for rotation 0: $0.127e + 0.015b$ and for rotation 5: $0.0015e + 0.061h$.
> 
> Hence, any given rotation comparatively is largest when the letter frequencies follow its expected distribution.

As before, transformers use linear algebra to compute these as follows:

> Similar to the embedding matrix, we combine each class's expected frequencies into an **unembedding matrix** where each _column_ represents a rotation class: $$U = \begin{vmatrix}0.127&0.0015\\0.015&0.061\end{vmatrix}.$$
> To compute the final logits, we use matrix multiplication: $$L = YU = \begin{vmatrix}e&b\end{vmatrix} \begin{vmatrix}0.127&0.0015\\0.015&0.061\end{vmatrix} = \begin{vmatrix}0.127e + 0.015b&0.0015e + 0.061b\end{vmatrix}.$$
> 
> Typically there will also be unembed biases $b_u$ (one per rotation class), making the final logits: $$L = YU + b_u.$$

Note there are two conspicuous ways to make a large dot product:
1. Align the signs/magnitudes so that large positives match with large positives and large negatives match with large negatives.
2. Use a few particularly large outliers to influence the result, similarly to a gate.

In LLMs, the first method seems much more common. However, there are some examples of trained models using outliers as gates.


#### Python Implementation

```python
# (n_toks, n_classes) = (n_toks, K) @ (K, n_classes) + (n_classes,)
logits = Y @ unembeds + bu
pred_class = np.argmax(self.logits[-1])  # final row incl all tokens
```
