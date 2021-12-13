# Hilbert Space Theory

Complex random variables form a Hilbert space with inner product
$$\langle X, Y \rangle  = \mathbb{E}\left[XY^*\right] $$. If we have a
random complex vector, then we can use Hilbert Theory in a more
efficient manner by looking at the matrix of inner products. For
simplicity, we will call this the “inner product” of two complex
vectors.

{% hint style="info" %}

### Definition 1

Let the inner product between two random, complex vectors $$\boldsymbol{Z_1}, \boldsymbol{Z_2}$$ be defined as 

$$ \langle \boldsymbol{Z_1}, \boldsymbol{Z_2} \rangle  = \mathbb{E}\left[\boldsymbol{Z_1}\boldsymbol{Z_2}^*\right]  $$

{% endhint %}

The ij-th entry of the matrix is simply the scalar inner product
$$\mathbb{E}\left[X_iY_j^*\right] $$ where $$X_i$$ and $$Y_j$$ are the
ith and jth entries of $$\boldsymbol{X}$$ and $$\boldsymbol{Y}$$
respectively. This means the matrix is equivalent to the cross
correlation $$R_{XY}$$ between the two vectors. We can also specify the
auto-correlation
$$R_X = \langle \boldsymbol{X}, \boldsymbol{X} \rangle $$ and
auto-covariance
$$\Sigma_X = \langle \boldsymbol{X} - \mathbb{E}\left[\boldsymbol{X}\right] , \boldsymbol{X} - \mathbb{E}\left[\boldsymbol{X}\right]  \rangle $$.
One reason why we can think of this matrix as the inner product is
because it also satisfies the properties of inner products. In
particular, it is

1.  Linear:
    $$\langle \alpha_1\boldsymbol{V_1}+\alpha_2\boldsymbol{V_2}, \boldsymbol{u} \rangle  = \alpha_1\langle \boldsymbol{V_1}, \boldsymbol{u} \rangle  + \alpha_2\langle \boldsymbol{V_2}, \boldsymbol{u} \rangle $$.

2.  Reflexive:
    $$\langle \boldsymbol{U}, \boldsymbol{V} \rangle  = \langle \boldsymbol{V}, \boldsymbol{U} \rangle ^*$$.

3.  Non-degeneracy:
    $$\langle \boldsymbol{V}, \boldsymbol{V} \rangle  = \boldsymbol{0} \Leftrightarrow \boldsymbol{V} = \boldsymbol{0}$$.

Since we are thinking of the matrix as an inner product, we can also
think of the norm as a matrix.

{% hint style="info" %}

### Definition 2

The norm of a complex random vector is given by $$\|\boldsymbol{Z}\|^2 = \langle \boldsymbol{Z}, \boldsymbol{Z} \rangle $$.

{% endhint %}

When thinking of inner products as matrices instead of scalars, we must
rewrite the Hilbert Projection Theorem to use matrices instead.

{% hint style="info" %}

### Theorem 1 (Hilbert Projection Theorem) {#theorem-1}

The minimization problem $$\min_{\hat{\boldsymbol{X}}(\boldsymbol{Y})}\|\hat{\boldsymbol{X}}(\boldsymbol{Y}) - \boldsymbol{X}\|^2$$ has a unique solution which is a linear function of $$\boldsymbol{Y}$$. The error is orthogonal to the linear subspace of $$\boldsymbol{Y}$$ (i.e $$\langle \boldsymbol{X} - \hat{\boldsymbol{X}}, \boldsymbol{Y} \rangle  = \boldsymbol{0}$$)

{% endhint %}

When we do a minimization over a matrix, we are minimizing it in a PSD
sense, so for any other linear function $$\boldsymbol{X}'$$,

$$\|\boldsymbol{X}-\hat{\boldsymbol{X}}\|^2  \preceq \|\boldsymbol{X} - \boldsymbol{X}'\|^2.$$

## Innovations

Suppose we have jointly distributed random variables
$$Y_0, Y_1,\cdots,Y_n$$. Ideally, we would be able to “de-correlate”
them so each new vector $$E_0$$ captures the new information which is
orthogonal to previous random vectors in the sequence. Since vectors of
a Hilbert Space operate like vectors in $$\mathbb{R}^n$$, we can simply
do Gram-Schmidt on the $$\{Y_i\}_{i=0}^n$$.

{% hint style="info" %}

### Definition 3

Given jointly distributed random vectors $$\{Y_i\}_{i=0}^n$$ with $$\mathcal{L}_i = \text{span}\{Y_j\}_{j=0}^i$$, the ith innovation $$E_i$$ is given by


$$ E_i = Y_i - \text{proj}(Y_i|\mathcal{L}_{i-1}) = Y_i - \sum_{j=0}^{i-1}\frac{\langle Y_i, E_j \rangle }{\|E_j\|^2}E_j $$

{% endhint %}

Innovations have two key properties.

1.  $$\forall i\neq j,\ \langle E_i, E_j \rangle =0$$

2.  $$\forall i,\ \text{span}\{Y_j\}_{j=0}^i = \text{span}\{E_j\}_{j=0}^i$$

We can also write innovations in terms of a matrix where
$$\boldsymbol{\varepsilon} = A\boldsymbol{Y}$$ where
$$\boldsymbol{\varepsilon} = \begin{bmatrix}E_0 & E_1 & \cdots & E_n\end{bmatrix}^T$$
and
$$\boldsymbol{Y} = \begin{bmatrix}Y_0 & Y_1 & \cdots & Y_n\end{bmatrix}^T$$.
Since each $$E_i$$ only depends on the previous $$Y_i$$, then A must be
lower triangular, and because we need each $$E_i$$ to be mutually
orthogonal, $$R_{\varepsilon}$$ should be diagonal.
$$R_{\varepsilon} = AR_YA^*$$, so if $$R_Y \succ 0$$, then we can use
its unique LDL decomposition $$R_Y = LDL^*$$ and let $$A = L^{-1}$$.

