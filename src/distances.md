# Comparing Vectors

<div class="warning">
In this book, we'll use the terms distance, score, and similarity. For any $k$-dimensional vectors $\mathbf{x}$ and $\mathbf{y}$:

- A **distance measure** such as Euclidean distance $d(\mathbf{x},\mathbf{y})$ requires smaller values to indicate "more similar". We assume for any distance measure, $d(\mathbf{x},\mathbf{y}) = 0$ if and only if $\mathbf{x} = \mathbf{y}$, $d(\mathbf{x},\mathbf{y}) = d(\mathbf{y},\mathbf{x})$, and $d(\mathbf{x},\mathbf{y}) \geq 0$.

- A **similarity measure** such as **cosine similarity** $c(\mathbf{x},\mathbf{y})$ requires larger values to indicate "more similar", but no upper bound is required. Often, similarity measures can be converted to distance measures, and vice versa. For example, since cosine similarity is bounded by -1 and 1, we can define **cosine distance** as $1 - c(\mathbf{x}, \mathbf{y})$.

- A **score** $s(\mathbf{x},\mathbf{y})$ is an arbitrary function where a larger score indicates a "better" match. All similarity measures are scores, and all negated distance metrics are scores. This terminology was invented to avoid misunderstandings that may arise when comparing, say, Euclidean distance with cosine similarity.
</div>

## The Euclidean distance

In machine learning, we frequently must compare how far one vector is from another. There are many ways of doing this. Among the simplest (and most common) is the **Euclidean distance**. This is the "straight line" distance we're familiar with in the real world.

Its definition is based on the Pythagorean Theorem. Given a right-angled triangle with side lengths $a,b$ and hypotenuse length $c$, then $c^2 = a^2 + b^2$. We can use this to compute the distance ("hypotenuse") between two two-dimensional points. 

> Suppose we want to know the "straight-line" distance between two points $\mathbf{x} = (x_1, x_2)$ and $\mathbf{y} = (y_1, y_2)$. We can connect the points with a right triangle with sides $|x_1 - y_1|$ and $|x_2 - y_2|$. Then, by the Pythagorean Theorem, the "straight-line" distance between them (the hypotenuse) is $d(\mathbf{x}, \mathbf{y}) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}$.

Applying the Pythagorean Theorem a second time, we can derive the formula for three-dimensional vectors $d(\mathbf{x}, \mathbf{y}) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + (x_3 - y_3)^2}$.

Noticing a pattern, we can define the general $k$-dimensional Euclidean distance as follows:

$$
d(\mathbf{x}, \mathbf{y}) := \sqrt{\sum_{i=1}^k{(x_i - y_i)^2}}.
$$

> **Definition.** Given a $k$-dimensional vector $\mathbf{x}$, the **magnitude of $\mathbf{x}$** is $$|\mathbf{x}| := \sqrt{\sum_{i=1}^{k}{x_i^2}}.$$

> **Definition.** Given $k$-dimensional vectors $\mathbf{x}$ and $\mathbf{y}$, the **Euclidean distance** is $$d(\mathbf{x}, \mathbf{y}) := |\mathbf{x} - \mathbf{y}|.$$

We will see that some distance measures (including Euclidean distance) are referred to as **metrics**:

> **Definition.** A **distance metric $d$** satisfies the following properties. For any $k$-dimensional vectors $\mathbf{x}$, $\mathbf{y}$, and $\mathbf{z}$:
> - $d(\mathbf{x},\mathbf{x}) = 0$;
> - **Positivity.** $d(\mathbf{x},\mathbf{y}) > 0$;
> - **Symmetry.** $d(\mathbf{x},\mathbf{y}) = d(\mathbf{y},\mathbf{x})$;
> - **Triangle inequality.** $d(\mathbf{x},\mathbf{z}) \leq d(\mathbf{x},\mathbf{y}) + d(\mathbf{y},\mathbf{z})$

The triangle inequality ensures that the shortest distance between two points is a straight line.

## Dot product

> **Definition.** Given two $k$-dimensional vectors $\mathbf{x}$ and $\mathbf{y}$, the **dot product** is their elementwise product: $\mathbf{x} \cdot \mathbf{y} := \sum_{i=1}^{k}{x_iy_i}$.

In machine learning, the dot product often is used as a similarity measure. However, as we'll see below it is somewhat imperfect. It is affected both by the angle between the vectors and their magnitudes.

> **Definition.** Given a $k$-dimensional vector $\mathbf{x}$, the **unit vector of $\mathbf{x}$** is $\mathbf{\hat{x}} = \mathbf{x}/|\mathbf{x}|$. This vector is in the direction of $\mathbf{x}$ but has magnitude $1$, on the unit circle.

In many textbooks, the dot product is defined directly in terms of the angle between two vectors. Where $\mathbf{u},\mathbf{v}$ are vectors, $|\mathbf{u}|,|\mathbf{v}|$ are their magnitudes, and $\theta$ is the angle between the vectors:

$$
\begin{align}
&& \mathbf{u} \cdot \mathbf{v} := && |\mathbf{u}||\mathbf{v}| \cos{\theta} \\
\iff && \hat{\mathbf{u}} \cdot \hat{\mathbf{v}} := && \cos{\theta}. \\
\end{align}
$$

Hence, if the vectors are unit vectors, the dot product by itself is the cosine of the angle between them.


## Cosine similarity

The **cosine similarity** is the cosine of the angle between two vectors. It is useful when direction is more important than magnitude. For example, someone who always rates movies $4/5$ is similar to someone who always rates them $5/5$ (perhaps the origin represents all movies rated $2.5/5$) -- their ratings convey no information about the movies!

> The cosine similarity $s$ is bounded between [-1, 1], with a larger score indicating more similar.
> - **s = 1**. Same direction, since: $\cos{0} = 1$, e.g. $[1,0] \cdot [1,0] = 1$.
> - **s = -1**. Opposite direction, since: $\cos \pi = -1$, e.g. $[1,0] \cdot [-1,0] = -1$.
> - **s = 0**. Orthogonal (e.g. a 90-degree angle), since: $\cos \pi/2 = 0$, e.g. $[1,0] \cdot [0,1] = 0$.

This measure can be efficiently computed, since it is based on the **dot product**.

Hence, the dot product by itself is effectively an "un-normalized" cosine similarity. This means it can be affected by magnitude!

> This is especially apparent in modern LLMs if the embeddings are unembedded by themselves. Using cosine similarity, each token will match best with itself. However, merely using the dot product will cause many tokens to match with other tokens with larger magnitude.


## Exercises

1. By applying the Pythagorean Theorem twice, derive the three-dimensional Euclidean distance formula. Prove that for any three-dimensional vectors $\mathbf{x}$ and $\mathbf{y}$,

$$d(\mathbf{x}, \mathbf{y}) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + (x_3 - y_3)^2}.$$
