# Unembeddings

Finally, we must compare our ciphertext token frequencies to the expected frequencies given each possible rotation (henceforth called the **output class**). We obtain a score for each output class (called a **logit**). Then the largest score will indicate the most likely rotation.

To compute the scores, we will use the dot product. There are two conspicuous ways to make a large dot product:
1. Align the signs/magnitudes so that large positives match with large positives and large negatives match with large negatives.
2. Use a few particularly large outliers to influence the result, similarly to a gate.

In LLMs, the first method is much more common. So, we'll use that here:

> Suppose for rotation 0 we expect frequencies for the letters `e` and `h` to be $(0.12, 0.01)$, respectively. For rotation 3 this is reversed at $(0.01, 0.12)$.
> 
> In our ciphertext, the respective frequencies are $(e, h)$. This makes the dot product for rotation 0: $0.12e + 0.01h$ and for rotation 3: $0.01e + 0.12h$.
> a
> Hence, rotation 0 comparatively is largest when $e$ and $h$ follow its distribution and smallest when they do not. (This largely works because all frequencies must sum to 1.)

As before, transformers use linear algebra to compute these as follows:

> Similar to the embedding matrix, we combine each class's expected frequencies into an **unembedding matrix** where each _column_ represents a class: $$U = \begin{vmatrix}0.12&0.01\\0.01&0.12\end{vmatrix}.$$
> To compute the final logits, we use matrix multiplication: $$L = YU = \begin{vmatrix}e&h\end{vmatrix} \begin{vmatrix}0.12&0.01\\0.01&0.12\end{vmatrix} = \begin{vmatrix}0.12e + 0.01h&0.01e + 0.12h\end{vmatrix}.$$


#### Python Implementation

```python
# (n_toks, n_classes) = (n_toks, K) @ (K, n_classes) + (n_classes,)
logits = Y @ unembeds + bu
pred_class = np.argmax(self.logits[-1])  # final row incl all tokens
```


That said, in many cases it is fairly safe to think of the dot product as related to the "angle" between the vectors:

> Let's interpret our example above in terms of vector angles. We can imagine the $e$ embedding as $(1, 0)$ (the $x$-axis) and the $h$ embedding as $(0, 1)$ (the $y$-axis). Then, the rotation 0 class vector would point toward $(0.12, 0.01)$ and the rotation 1 class vector toward $(0.01, 0.12)$.
> 
> Our ciphertext token frequencies $(e, h)$ will then also be a vector. We compare its angle to that of each of the two rotation classes, and the class vector with largest dot product (typically smallest angle) determines the predicted class.

<div class="canvas-figure">
    <canvas id="unembedding-vectors">
        shows the x,y axes representing the frequencies of es and hs.
        shows the rotation 0 and rotation 3 class vectors in relation.
    </canvas>
</div>
