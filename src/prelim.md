# Mathematical Preliminaries

This book is intended to be entirely self-contained. So, we are providing a minimal set of definitions needed to understand the text.

- We indicate a **definition** by $:=$. This often defines a new notation or operator that cannot be derived from other statements.

Given statements $A$ and $B$:
- $A \implies B$ ("$A$ implies $B$") indicates that $B$ logically follows from $A$ (but not necessarily in the opposite direction).
- $A \iff B$ ("$A$ if and only if $B$") indicates that $A$ follows from $B$ and $B$ follows from $A$ (in both directions).

## Sets

A **set** V is a collection of unique elements. We denote membership by $x \in V$ ("$x$ is an element of $V$" or "$x$ is in $V$"). We denote the size of the set (its **cardinality**) by $|V|$. The empty set is denoted $\phi$.

> For example, we may define a vocabulary $V = \{\text{"and"}, \text{"or"}\}$ which has cardinality $|V| = 2$. Then, $\text{"and"} \in V$ while $\text{"but"} \not \in V$.

**Set operations.** Given sets $A$ and $B$, we define:

- The **union** $A \cup B := \{x \mid x \in A \text{ or } x \in B \}$.
- The **intersection** $A \cap B := \{x \mid x \in A \text{ and } x \in B \}$.
- The **set difference** $A - B := \{x \mid x \in A \text{ and } x \not \in B \}$.

> The above uses **set-builder notation** to define each set in terms of $A$ and $B$. We read the **union** definition as "the set of all $x$ such that $x$ is in $A$ or $x$ is in $B$".

**Special sets.** We denote $\mathbb{R}$ as the set of all real numbers. For example: $1, 1.5, \pi \in \mathbb{R}$. We define $\mathbb{Z}$ as the set of all integers and $\mathbb{Z}^+$ as the set of all positive integers. From these, we define the **natural numbers** $\mathbb{N} = \mathbb{Z}^+ \cup \{0\}$.

## Tuples and the Cartesian product

**Tuples** are often used in conjunction with sets and vectors. An **$n$-tuple** is an ordered collection of $n$ elements. The $1$-tuple is a **singleton**, and the $2$-tuple an **ordered pair**. For example, the point $\mathbf{p} = (-1, 3)$ is an ordered pair.

**Cartesian product.** We can redundantly define the set of all reals as:

$$\mathbb{R} = \{x \mid x \in \mathbb{R}\}.$$

Let's generalize this to the set of all points (ordered pairs) in 2D space by using the **Cartesian product $\times$ of $\mathbb{R}$**: $$\mathbb{R}^2 := \mathbb{R} \times \mathbb{R} := \{(x_1, x_2) \mid x_1,x_2 \in \mathbb{R} \}.$$

For any set $A$, we can generally define the **$k$-fold Cartesian product of A** to produce the set of all $k$-tuples with elements in $A$: $$A^k := A \times \dots \times A := \{(x_1, \dots, x_k) \mid x_1,\dots,x_k \in A \}.$$

So the ordered pair $\mathbf{p} = (-1, 3) \in \mathbb{Z}^2$ and $(1, 2, \pi, 4, 5, 6) \in \mathbb{R}^6$.


## Vectors

In this book, we will only work with vectors defined on the reals. So, we refer to a real number as a **scalar**. Vectors will be written in lowercase boldface, e.g. $\mathbf{x}$, while elements of vectors (scalars) will be lowercase non-boldface, e.g. $a$.

> **Definition.** A $k$-dimensional **vector** is a $k$-tuple with scalar elements: $$(x_1, \dots, x_k) \in \mathbb{R}^k.$$

For any two $k$-dimensional vectors $\mathbf{x}$ and $\mathbf{y}$ and scalar $a$, we define the following:

> **Definition.** The mathematical operator $+$ is defined **elementwise** such that $$\mathbf{x} + \mathbf{y} := (x_1 + y_1, \dots, x_k + y_k).$$

> **Definition.** Scalar-vector multiplication is defined as $$a\mathbf{x} := (ax_1, \dots, ax_k).$$

> **Definition.** The mathematical operator $-$ is defined elementwise as $$\mathbf{x} - \mathbf{y} := \mathbf{x} + -1 \cdot \mathbf{y} = (x_1 - y_1, \cdots, x_k - y_k).$$


