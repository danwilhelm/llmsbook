# Letter Frequencies

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


## What is a "Best Match"?

Our strategy is to compute the expected letter frequency distribution for each rotation. Given ciphertext, we'll calculate its letter frequencies and score how "close" they are to each rotation's expected distribution. The highest score will then be the expected rotation.

Although there are several methods for comparing probability distributions, in this chapter we'll start with the simplest -- sum of squared errors (SSE).

### Sum of Squared Errors (SSE)

For each letter, we'll determine how "close" we are by squaring the difference between the expected and observed frequencies. Squaring gives several benefits: First, it makes all differences positive. Second, it magnifies large differences in our total error. Third, compared to absolute value, it is differentiable (for potential minimization) and easier to prove things about!

#### SSE definition

Given a vocabulary $V$ containing $|V|$ letters, suppose we have $|V|$-length ciphertext frequencies $\mathbf{c}$ and rotation frequencies $\mathbf{r}$. Then the formula looks like this: 

$$\text{SSE}(\mathbf{c}, {\mathbf{r}}) = \sum_{i=1}^{|V|}{(c_{i} - r_{i})^2} = |\mathbf{c} - \mathbf{r}|^2.$$

Due to linearity, we can break this into individual sums of terms:

$$\text{SSE}(\mathbf{c}, {\mathbf{r}}) = \sum_{i=1}^{|V|}{c_{i}^2} - 2\sum_{i=1}^{|V|}{c_{i}r_{i}} + \sum_{i=1}^{|V|}{r_{i}^2}.$$

Using the definitions of vector magnitude and dot product, we can express it more compactly:

$$\text{SSE}(\mathbf{c}, {\mathbf{r}}) = |\mathbf{c}|^2 - 2\mathbf{c} \cdot \mathbf{r} + |\mathbf{r}|^2.$$

#### Comparing SSEs

Unlike in machine learning, here we are not trying to minimize the SSE. Instead, we are looking for which rotation's frequency distribution gives the smallest SSE. Regardless of rotation:

- Each $\text{SSE}(\mathbf{c}, {\mathbf{r}})$ is applied to same ciphertext vector $\mathbf{c}$, making $|\mathbf{c}|$ constant.
- For any two rotation frequencies $\mathbf{r_1}$ and $\mathbf{r_2}$, $|\mathbf{r}_1| = |\mathbf{r_2}|$. This is because rotation changes the position of each frequency value, but the frequencies themselves are the same.

Then we conclude:

$$
\begin{align}
&& \text{SSE}(\mathbf{c}, {\mathbf{r_1}}) < && \text{SSE}(\mathbf{c}, {\mathbf{r_2}}) \\
\iff && |\mathbf{c}|^2 - 2\mathbf{c} \cdot \mathbf{r_1} + |\mathbf{r_1}|^2 < && |\mathbf{c}|^2 - 2\mathbf{c} \cdot \mathbf{r_2} + |\mathbf{r_2}|^2 \\
\iff && \mathbf{c} \cdot \mathbf{r_1} > && \mathbf{c} \cdot \mathbf{r_2}.
\end{align}
$$

This gives us a score/ranking function $S_{\mathbf{r}}(\mathbf{c}) = \mathbf{c} \cdot \mathbf{r}$. By applying the above equivalence, we conclude the rotation with _largest_ score has the _smallest_ SSE.
