# Letter Frequency Solution

In this section, we'll solve cryptograms by comparing the ciphertext letter frequencies to those expected with each rotation. We'll prove that in this particular problem, comparing the distributions using dot products gives the same result as the sum of squared errors (SSE).


## Letter Frequency

Humans solve cryptograms by analyzing letter frequency.

> Since `e` is the most common English letter, we can assume the most common ciphertext letter is a rotated `e`.

More generally, we can match the letter frequencies of our ciphertext to the letter frequencies we'd expect to see for each rotation. The "best match" will indicate the most likely rotation.

> Suppose the most common English letters are `etainos`. In our ciphertext, the most common letters are `fubjopt` -- one letter higher. Hence, we can predict a rotation of 1.

From Wikipedia, here are the letter frequencies for "Texts":

```python
# English language frequencies from "Texts" for the letters a-z
base_freq = [0.082, 0.015, 0.028, 0.043, 0.127, 0.022, 0.020, 0.061, 0.070, 
             0.0015, 0.0077, 0.04, 0.024, 0.067, 0.075, 0.019, 0.00095, 0.060, 
             0.063, 0.091, 0.028, 0.0098, 0.024, 0.0015, 0.020, 0.00074]
```

For a rotation of 2 where A=>C, B=>D, etc. these are cyclically rotated:

```python
# English language frequencies from "Texts" for the letters a-z
rot_freqs = np.array([base_freq[-rot:] + base_freq[rot:] for rot in range(N_LETTERS)])
```

### Formalizing the letter frequency

Let's make this more formal. Suppose we have a vocabulary $V$ containing $|V|$ tokens. In this problem, $V = \{\text{a}, \text{b}, \dots, \text{z}, \text{"space"}\}$ and $|V| = 27$. We will only rotate a-z, so we'll define $N_\text{rot} = 26$.

We define the rotation letter frequency vector $\mathbf{r}_0$ as the base English letter frequencies (above called **base_freqs**). We define $\mathbf{r}_i$ as the letter frequencies when rotated by $i$ (above called **rot_freqs**).

For the $k$th letter, we define $r_{ik} := r_{0(i + k \mod N_\text{rot})}$. This defines a one-to-one map between $\mathbf{r}_i$ and $\mathbf{r}_0$. Hence, the set of values in each vector are identical, making $|\mathbf{r}_i| = |\mathbf{r}_j|$ for any $0 \leq i,j < N_\text{rot}$.

As we'll see below, because these magnitudes are equal they will cancel out when we compare the distance between vectors.

## Distribution distances

To solve the cryptogram, we must find which of the $N_\text{rot}$ rotation letter distributions is "closest" to that of the ciphertext. To do so, we will compute the "distance" between the ciphertext letter frequencies $\mathbf{c}$ and each of the $N_\text{rot}$ expected frequency distributions.

Although there are several methods for comparing frequency distributions, in this chapter we'll start with the simplest -- sum of squared errors (SSE).

### Sum of Squared Errors (SSE)

We will compare each letter frequency individually, penalizing larger errors more. We'll do this by _squaring_ these errors and summing the squared errors.

Based on this, here is our definition:

$$
\text{SSE}(\mathbf{c}, {\mathbf{r}}) := \sum_i{(c_{i} - r_{i})^2} = |\mathbf{c} - \mathbf{r}|^2.
$$

Expanded:

$$
\begin{align}
\text{SSE}(\mathbf{c}, {\mathbf{r}}) = && \sum_i{c_{i}^2} - 2\sum_i{c_{i}r_{i}} + \sum_i{r_{i}^2} \\
= && |\mathbf{c}|^2 - 2\mathbf{c} \cdot \mathbf{r} + |\mathbf{r}|^2.
\end{align}
$$

Squaring the errors gives several benefits: First, it makes all differences positive. Second, it magnifies large differences in our total error. Third, compared to absolute value, it is differentiable (for potential minimization) and easier to prove things about!


#### Comparing SSEs

Although we could compute all $N_\text{rot}$ SSEs, there is a faster and easier way. It turns out we can simply take the dot product of $\mathbf{c}$ with each $\mathbf{r}$:

> **Theorem 1.** For a given ciphertext letter distribution $\mathbf{c}$ and rotation distributions $\mathbf{r_1}$ and $\mathbf{r_2}$, then $\text{SSE}(\mathbf{c}, {\mathbf{r_1}}) < \text{SSE}(\mathbf{c}, {\mathbf{r_2}})$ if and only if $\mathbf{c} \cdot \mathbf{r_1} > \mathbf{c} \cdot \mathbf{r_2}$.
> 
> **Proof.** We showed earlier that $|\mathbf{r}_i| = |\mathbf{r}_j|$ for any $0 \leq i,j < N_\text{rot}$. Then: $$
\begin{align}
&& \text{SSE}(\mathbf{c}, {\mathbf{r_1}}) < && \text{SSE}(\mathbf{c}, {\mathbf{r_2}}) \\
\iff && |\mathbf{c}|^2 - 2\mathbf{c} \cdot \mathbf{r_1} + |\mathbf{r_1}|^2 < && |\mathbf{c}|^2 - 2\mathbf{c} \cdot \mathbf{r_2} + |\mathbf{r_2}|^2 \\
\iff && \mathbf{c} \cdot \mathbf{r_1} > && \mathbf{c} \cdot \mathbf{r_2}.
\end{align}
$$

Let's define a score function $S(\mathbf{c}, \mathbf{r}) = \mathbf{c} \cdot \mathbf{r}$. By definition, the rotation with _smallest_ SSE has a smaller SSE than any other rotation. From Theorem 1, the rotation with the _largest_ score has the _smallest_ SSE (and vice versa). For more discussion of score vs. distance vs. similarity, see Appendix B.

<div class="warning">
Note this does not hold in general -- only when all class vectors have the same magnitude.
</div>

Note that we can simultaneously compute the dot products using matrix algebra. Given a matrix $R$ where column $i$ is $\mathbf{r}_i$ and a row vector of ciphertext frequencies $\mathbf{c}$, then $\mathbf{c}R$ gives a vector of all dot products.


#### Visualizing the dot product

Interestingly, the SSE can now be interpreted as related to the "angle" between vectors. This is because the dot product of unit vectors is the cosine of the angle between them -- see Appendix B for more details!

> Imagine the x-axis depicting the letter $e$ frequency and the y-axis the frequency of $b$. Then, the rotation 0 class vector is $(r_{04}, r_{02}) = (0.127, 0.015)$ and the rotation 5 class vector $(r_{54}, r_{52}) = (r_{09}, r_{07}) = (0.0015, 0.061)$.
> 
> Our ciphertext token frequencies $(e, b)$ will then also be a vector. We compare its angle to that of each of the two rotation classes, and the class vector with largest dot product (typically smallest angle) determines the predicted class.

<div class="canvas-figure">
    <canvas id="unembedding-vectors">
        shows the x,y axes representing the e,b letter frequencies.
        shows the rotation 0 and rotation 3 class vectors in relation.
    </canvas>
</div>

