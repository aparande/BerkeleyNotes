# Linear Estimation

In Linear Estimation, we are trying to estimate a random variable
$$\boldsymbol{X}$$ using an observation $$\boldsymbol{Y}$$ with a linear
function of $$\boldsymbol{Y}$$. If $$\boldsymbol{Y}$$ is finite
dimensional, then we can say
$$\hat{\boldsymbol{X}}(\boldsymbol{Y}) = W\boldsymbol{Y}$$ where $$W$$
is some matrix. Using Theorem 1 and the orthogonality principle, we know
that

$$\langle \boldsymbol{X}-W\boldsymbol{Y}, \boldsymbol{Y} \rangle  = \boldsymbol{0} \Leftrightarrow R_{XY} = W\boldsymbol{R}_Y$$

This is known as the **Normal Equation**. If $$R_Y$$ is invertible, then
we can apply the inverse to find $$W$$. Otherwise, we can apply the
pseudoinverse $$R_Y^\dagger$$ to find $$W$$, which may not be unique. If
we want to measure the quality of the estimation, since
$$\boldsymbol{X} = \boldsymbol{X}+(\boldsymbol{X}-\hat{\boldsymbol{X}})$$,

$$\begin{aligned}     \|\boldsymbol{X}\|^2 &= \|\hat{\boldsymbol{X}}\|^2 + \|\boldsymbol{X} - \hat{\boldsymbol{X}}\|^2 \implies \\     \|\boldsymbol{X}-\hat{\boldsymbol{X}}\|^2 &= \|\boldsymbol{X}\|^2 - \|\hat{\boldsymbol{X}}\|^2 = R_X - R_{XY}R_Y^{-1}R_{YX}\end{aligned}$$

## Affine Estimation

If we allow ourselves to consider an affine function for estimation
$$\hat{\boldsymbol{X}}(\boldsymbol{Y}) = W\boldsymbol{Y}+b$$, then this
is equivalent to instead finding an estimator

$$\hat{\boldsymbol{X}}(\boldsymbol{Y}') = W\boldsymbol{Y}' \qquad \text{ where } \boldsymbol{Y}' = \begin{bmatrix} \boldsymbol{Y} \\ 1 \end{bmatrix}$$

This is equivalent to the following orthogonality conditions:

1.  $$\langle \boldsymbol{X}-\hat{\boldsymbol{X}}, \boldsymbol{Y} \rangle $$

2.  $$\langle \boldsymbol{X}-\hat{\boldsymbol{X}}, 1 \rangle $$

Solving gives us

$$\hat{\boldsymbol{X}}(\boldsymbol{Y}) = W(\boldsymbol{Y}-\boldsymbol{\mu}_Y) + \mu_x \qquad \text{ where } W\Sigma_Y=\Sigma_{XY}.$$

$$\Sigma_Y$$ and $$\Sigma_{XY}$$ are the auto-covariance and
cross-covariance respectively. Recall that if

$$\begin{bmatrix} \boldsymbol{X} \\ \boldsymbol{Y} \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \boldsymbol{\mu}_X \\ \boldsymbol{\mu}_Y \end{bmatrix}, \begin{bmatrix} \Sigma_X & \Sigma_{XY}\\ \Sigma_{YX} & \Sigma_Y \end{bmatrix}\right)$$

then

$$\boldsymbol{X}|\boldsymbol{Y} \sim \mathcal{N}\left(\boldsymbol{\mu}_X + \Sigma_{XY}\Sigma_Y^{-1}(\boldsymbol{Y}-\boldsymbol{\mu}_Y), \Sigma_X-\Sigma_{XY}\Sigma_Y^{-1}\Sigma_{YX} \right)$$

Thus in the Joint Gaussian case, the mean of the conditional
distribution is the best affine estimator of $$\boldsymbol{X}$$ using
$$\boldsymbol{Y}$$, and the covariance is the estimation error. This has
two interpretations.

1.  Under the Gaussian assumption, the best nonlinear estimator
    $$\mathbb{E}\left[\boldsymbol{X}|\boldsymbol{Y}\right] $$ is affine

2.  Gaussian random variables are the hardest predict because
    nonlinearity should improve our error, but it does not in the
    Gaussian case. This means if affine estimation works well, we
    shouldn’t try and find better non-linear estimators.

## Least Squares

The theory of linear estimation is very closely connected with the
theory behind least squares in linear algebra. In least squares, we have
a deterministic $$\boldsymbol{x}$$ and assume nothing else about it,
meaning we are looking for an unbiased estimator. Theorem 2 tells us how
to find the best linear unbiased estimator in a linear setting.

{% hint style="info" %}

### Theorem 2 (Gauss Markov Theorem) {#theorem-2}

Suppose that $$\boldsymbol{Y}=H\boldsymbol{x}+\boldsymbol{Z}$$ and $$Z$$ is zero-mean with $$\langle \boldsymbol{Z}, \boldsymbol{Z} \rangle  = \boldsymbol{I}$$, $$H$$ is full-column rank, then $$\hat{\boldsymbol{x}_b} = (H^*H)^{-1}H^*\boldsymbol{Y}$$is the best linear unbiased estimator.

{% endhint %}

### Recursive Least Squares

Suppose we extend the least squares setup to allow a stochastic, but
fixed, $$\boldsymbol{X}$$ where
$$\langle \boldsymbol{X}, \boldsymbol{X} \rangle  = \Pi_0$$. At each
timestep, we receive observations of $$\boldsymbol{X}$$ such that
$$\boldsymbol{Y}_i = h_i^* \boldsymbol{X} + \boldsymbol{V}_i$$ where
$$\langle \boldsymbol{V}_i, \boldsymbol{V}_j \rangle  = \delta[i, j]$$
and $$\langle \boldsymbol{X}, \boldsymbol{V} \rangle $$. Define

$$\boldsymbol{Y}^i = \begin{bmatrix} \boldsymbol{Y}_0 \\ \boldsymbol{Y}_1 \\ \cdots \\ \boldsymbol{Y}_i \end{bmatrix}     \qquad      H_i = \begin{bmatrix}          h_0^*\\         h_1^*\\         \vdots\\         h_i^*\\     \end{bmatrix}     \qquad     \boldsymbol{V}^i = \begin{bmatrix} \boldsymbol{V}_0 \\ \boldsymbol{V}_1 \\ \cdots \\ \boldsymbol{V}_i \end{bmatrix}$$

Then our setup becomes
$$\boldsymbol{Y}^i= H_i \boldsymbol{X} + \boldsymbol{V}^i$$.

$$R_{XY^i} = \Pi_0 H_i^* \qquad R_{Y^i} = (H_i\Pi_0H_i^* + I)$$

Applying Theorem 1 and solving the normal equation, we see

$$\begin{aligned}     W &= \Pi_0 H_i^*(H_i\Pi_0H_i^* + I)^{-1} = \Pi_0 H_i^* (I - H_i(\Pi_0^{-1} + H_i^*H_i)^{-1}H_i^*)\\     &= \Pi_0 (I - H_i^*H_i(\Pi_0^{-1} + H_i^*H_i)^{-1})H_i^* \\     &= \Pi_0 ((\Pi_0^{-1} + H_i^*H_i)(H_i^*H_i)^{-1}(H_i^*H_i)(\Pi_0^{-1} + H_i^*H_i)^{-1}- H_i^*H_i(\Pi_0^{-1} + H_i^*H_i)^{-1})H_i^*\\     &= \Pi_0 \Pi_0^{-1}(H_i^*H_i)^{-1}H_i^*H_i(\Pi_0^{-1}+H_i^*H_i)^{-1}H_i^*\\     &= (\Pi_0^{-1} + H_i^* H_i)^{-1}H_i^*\end{aligned}$$

Suppose we want to do this in an online fashion where at each timestep
$$i$$, we only use the current $$h_i, \boldsymbol{Y}_i$$ and our
previous estimate $$\boldsymbol{X}_{i-1}$$. Let
$$P_i = (\Pi_0^{-1} + H_i^*H_i)^{-1}$$. Then

$$P_i^{-1} = \Pi_0 + \sum_{k=0}^i h_k h_k^* = P_{i-1}^{-1} + h_ih_i^*.$$

By applying the Sherman-Morrison-Woodbury identity, we can see that

$$P_i = P_{i-1} = P_{i-1} \frac{h_ih_i^*}{1 + h_i^*P_{-1}h_i} P_{i-1}$$

{% hint style="info" %}

### Theorem 3 (Recursive Least Squares Update) {#theorem-3}

The best least squares estimate using $$i+1$$ data points can be found by updating the best least squares estimate using $$i$$ data points using


$$ \hat{\boldsymbol{X}}_i = \hat{\boldsymbol{X}}_{i-1} + \frac{P_{i-1}h_i}{1 + h_i^*P_{i-1}h_i}(\boldsymbol{Y}_i - h_i^* \hat{\boldsymbol{X}}_{i-1}) $$

{% endhint %}

Notice that this formula scales an innovation in order to improve the
current estimate of $$\boldsymbol{X}$$.

Just as we could compute a recursive update, we can also compute a
“downdate” where we forget a particular observation. More concretely, we
want to use $$\hat{\boldsymbol{X}}_i$$ to find
$$\hat{\boldsymbol{X}}_{i|k}$$, the best linear estimator of
$$\boldsymbol{X}$$ using
$$\boldsymbol{Y}_0,\boldsymbol{Y}_1,\cdots,\boldsymbol{Y}_{k-1},\boldsymbol{Y}_{k+1},\cdots,\boldsymbol{Y}_i$$.
Defining $$P_{i|k} = (\Pi_0^{-1} + H_{i|k}^*H_{i|k})^{-1}$$,

$$P_{i|k}^{-1} = \Pi_0^{-1} + \sum_{j=0,j\neq k}^i h_jh_j^* = P_i^{-1} - h_kh_k^{-1}.$$

Applying the Sherman-Morrison-Woodbury identity,

$$P_{i|k} = P_i + P_i \frac{h_kh_k^*}{h_k^*P_ih_k - 1}P_i$$

{% hint style="info" %}

### Theorem 4 (Recursive Least Squares Downdate) {#theorem-4}

The best least squares estimate using all but the kth observation can be found by updating the best least squares estimate using all data points using


$$ \hat{\boldsymbol{X}}_{i|k} = \hat{X}_i + \frac{P_ih_k}{h_k^*P_ih_k - 1}(Y_k - h_k^*\hat{\boldsymbol{X}}_i) $$

{% endhint %}

