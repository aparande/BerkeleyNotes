# Discrete Time Random Processes

{% hint style="info" %}

### Definition 4

A Discrete-Time Random Process is a countably infinite collection of random variables on the same probability space $$\{X_n: n\in\mathbb{Z}\}$$.

{% endhint %}

Discrete Time Random Processes have a mean function
$$\mu_n = \mathbb{E}\left[X_n\right] $$ and an auto-correlation function
$$R_X(n_1, n_2) = \mathbb{E}\left[X_{n_1}X_{n_2}^*\right] $$

## Wide-Sense Stationary Random Processes

{% hint style="info" %}

### Definition 5

A Wide-Sense Stationary Random Process is a disrete-time random process with constant mean, finite variance, and an autocorrelation function that can be re-written to only depend on $$n_1-n_2$$.

{% endhint %}

We call this wide-sense stationary because the mean and covariance do
not change as the process evolves. In a strict-sense stationary process,
the distribution of each random variable in the process would not
change.

{% hint style="info" %}

### Definition 6

A WSS process $$Z\sim \mathcal{WN}(0, \sigma^2)$$ is a white noise process with variance $$\sigma^2$$ if and only if $$\mathbb{E}\left[Z_n\right]  = 0$$ and $$\mathbb{E}\left[Z_nZ_m^*\right]  = \sigma^2\delta[n, m]$$.

{% endhint %}

### Spectral Density

Recall that the Discrete Time Fourier Transform is given by

$$X(e^{j\omega}) = \sum_{n=-\infty}^{\infty}x[n]e^{-j\omega n}.$$

The Inverse Discrete Time Fourier Transform is given by

$$x[n] = \frac{1}{2\pi}\int_{-\pi}^{\pi}X(e^{j\omega})e^{j\omega n}d\omega.$$

Since the DTFT is an infinite summation, it may or may not converge.

{% hint style="info" %}

### Definition 7

A signal $$x[n]$$ belongs to the $$l^1$$ class of signals if the series converges absolutely. In other words,


$$ \sum_{k=-\infty}^{\infty}|x[k]| < \infty. $$

{% endhint %}

This class covers most real-world signals.

{% hint style="info" %}

### Theorem 5

If $$x[n]$$ is a $$l^1$$ signal, then the DTFT $$X(e^{j\omega})$$ converges uniformly and is well-defined for every $$\omega$$. $$X(e^{j\omega})$$is also a continuous function.

{% endhint %}

{% hint style="info" %}

### Definition 8

A signal $$x[n]$$ belongs to the $$l^2$$ class of signals if it is square summable. In other words,


$$ \sum_{k=-\infty}^{\infty}|x[k]|^2 < \infty. $$

{% endhint %}

The $$l^2$$ class contains important functions such as $$\text{sinc}$$.

{% hint style="info" %}

### Theorem 6

If $$x[n]$$ is a $$l^2$$ signal, then the DTFT $$X(e^{j\omega})$$ is defined almost everywhere and only converges in the mean-squared sense:


$$ \lim_{N\to\infty} \int_{-\pi}^{\pi}\left|\left(\sum_{k=-N}^N x[k]e^{-j\omega n}\right) - X(\omega)\right|^2d\omega = 0 $$

{% endhint %}

Tempered distributions like the Dirac Delta function are other functions
which are important for computing the DTFT, and they arise from the
theory of generalized functions.

Suppose we want to characterize the signal using its DTFT.

{% hint style="info" %}

### Definition 9

The energy of a deterministic, discrete-time signal $$x[n]$$ is given by 

$$ \sum_{n\in\mathbb{Z}}|x[n]|^2. $$

{% endhint %}

The autocorrelation of $$x[n]$$, given by $$a[n] = x[n] * x^*[-n]$$, is
closely related to the energy of the signal since
$$a[0] = \sum_{n\in\mathbb{Z}}|x(n)|^2$$.

{% hint style="info" %}

### Definition 10

The Energy Spectral Density $$x[n]$$ with auto-correlation $$a[n]$$ is given by 

$$ A(e^{j\omega}) = \sum_{n\in\mathbb{Z}}a[n]e^{-j\omega n} $$

{% endhint %}

We call the DTFT of the autocorrelation the energy spectral density
because, by the Inverse DTFT,

$$a[0] = \frac{1}{2\pi}\int_{-\pi}^{\pi}A(e^{j\omega})d\omega.$$

Since summing over each frequency gives us the energy, we can think of
$$A(e^{j\omega})$$ as storing the energy density of each spectral
component of the signal. We can apply this same idea to wide-sense
stationary stochastic processes.

{% hint style="info" %}

### Definition 11

The Power Spectral Density of a Wide-Sense Stationary random process is given by 

$$ S_X(e^{j\omega}) = \sum_{k\in\mathbb{Z}}R_X(k)e^{-j\omega k}. $$

{% endhint %}

Note that when considering stochastic signals, the metric changes from
energy to power. This is because if $$X_n$$ is Wide-Sense Stationary,
then

$$\mathbb{E}\left[\sum_{n\in\mathbb{Z}}|X_n|^2\right]  = \infty,$$

so energy doesn’t even make sense. To build our notion of power, let
$$A_T(\omega)$$ be a truncated DTFT of the auto-correlation of a
wide-sense stationary process, then

$$\begin{aligned}     \lim_{T\to\infty} \frac{\mathbb{E}\left[A_T(e^{j\omega})\right] }{2T + 1} &= \lim_{T\to\infty}\frac{1}{2T+1}\left(\sum_{n=-T}^Tx[n]e^{-j\omega n}\right)\left(\sum_{m=-T}^Tx^*[m]e^{j\omega m}\right)\\     &= \lim_{T\to\infty}    \frac{1}{2T+1} \sum_{n,m \in [-T,T]}\mathbb{E}\left[x[n]x^*[m]\right] e^{-j\omega(n-m)}\\     &= \lim_{T\to\infty}    \frac{1}{2T+1} \sum_{n,m \in [-T,T]}R_x(n-m)e^{-j\omega(n-m)}\\     &= \lim_{T\to\infty} \sum_{k=-2T}^{2T}R_X(k)e^{-j\omega k}\left(1 - \frac{|k|}{2T+1}\right)\\     &= \sum_{k=-\infty}^{\infty}R_X(k)e^{-j\omega k}\end{aligned}$$

The DTFT of the auto-correlation function naturally arises out of taking
the energy spectral density and normalizing it by time (the truncated
sequence is made of $$2T+1$$ points). In practice, this means to measure
the PSD, we need to either use the distribution of the signal to compute
$$R_X$$, or estimate the $$PSD$$ by averaging multiple realizations of
the signal.

The inverse DTFT formula tells us that we can represent a deterministic,
discrete-time signal $$x[n]$$ as a sum of complex exponentials weighted
by $$\frac{X(e^{j\omega})d\omega}{2\pi}$$. This representation has an
analog for stochastic signals as well.

{% hint style="info" %}

### Theorem 7 (Cramer-Khinchin) {#theorem-7}

For a complex-valued WSS stochastic process $$X_n$$ with power spectral density $$S_X(\omega)$$, there exists a unique right-continuous stochastic process $$F(\omega), \omega\in(-\pi,\pi]$$ with square-integrable, orthogonal increments such that 

$$ X_n = \int_{-\pi}^{\pi}e^{j\omega n}dF(\omega) $$

 where for any interval $$[\omega_1,\omega_2], [\omega_3, \omega_4]\subset [-\pi,\pi]$$, 

$$ \mathbb{E}\left[(F(\omega_2)-F(\omega_1))(F(\omega_4) - F(\omega_3))^*\right]  = f((\omega_1,\omega_2] \cap (\omega_3, \omega_4]) $$

 where $$f$$ is the structural measure of the stochastic process and has Radon-Nikodym derivative $$\frac{S_X(e^{j\omega})}{2\pi}$$.

{% endhint %}

Besides giving us a decomposition of a WSS random process, Theorem 7
tells a few important facts.

1.  $$\omega_1\neq\omega_2 \implies \langle dF(\omega_1), dF(\omega_2) \rangle = 0$$
    (i.e different frequencies are uncorrelated).

2.  $$\mathbb{E}\left[|dF(\omega)|^2\right]  = \frac{S_X(e^{j\omega})d\omega}{2\pi}$$

### Z-Spectrum

Recall that the Z-transform converts a discrete-time signal into a
complex representation. It is given by

$$X(z) = \sum_{n=-\infty}^{\infty}x[n]z^{-n}.$$

It is a special type of series called a **Laurent Series**.

{% hint style="info" %}

### Theorem 8

A Laurent Series will converge absolutely on an open annulus 

$$ A = \{z | r < |z| < R \} $$

 for some $$r$$ and $$R$$.

{% endhint %}

We can compute $$r$$ and $$R$$ using the signal $$x[n]$$.

$$r = \limsup_{n\to\infty} |x[n]|^{\frac{1}{n}}, \qquad \frac{1}{R} = \limsup_{n\to\infty}|x[-n]|^{\frac{1}{n}}.$$

In some cases, it can be useful to only compute the Z-transform of the
right side of the signal.

{% hint style="info" %}

### Definition 12

The unilateral Z-transform of a sequence $$x[n]$$ is given by 

$$ \left[X(z)\right]_+  = \sum_{n=0}^\infty x[n]z^{-n} $$

{% endhint %}

If the Z-transform of the sequence is a rational function, then we can
quickly compute what the unilateral Z-transform will be by leveraging
its partial fraction decomposition.

{% hint style="info" %}

### Theorem 9

Any arbitrary rational function $$H(z)$$ with region of convergence including the unit circle corresponds with the unilateral Z-transform 

$$ \left[H(z)\right]_+  = r_0 + \sum_{i=1}^m\sum_{k=1}^{l_i}\frac{r_{ik}}{(z+\alpha_i)^k} + \sum_{i=m+1}^n\sum_{k=1}^{l_i}\frac{r_{ik}}{\beta_i^k} $$


where $$|\alpha_i| < 1 < |\beta_i|$$.

{% endhint %}

{% hint style="info" %}

### Definition 13

For two jointly WSS processes $$X_n, Y_n$$, the z-cross spectrum is the Z-Transform of the correlation function $$R_{YX}(k) = \mathbb{E}\left[Y_nX^*_{n-k}\right] $$.


$$ S_{YX}(z) = \sum_{k\in\mathbb{Z}}R_{YX}(k)z^{-k} $$

{% endhint %}

Using this definition, we can see that

$$S_{XY}(z) = S^*_{YX}(z^{-*}).$$

We can also look at the Z-transform of the auto-correlation function of
a WSS process $$X$$ to obtain $$S_X(z)$$.

{% hint style="info" %}

### Definition 14

For a rational function $$S_X(z)$$ with finite power $$\left(\int_{-\pi}^\pi S_X(e^{j\omega})d\omega < \infty \right)$$ and is strictly positive on the unit circle, the canonical spectral factorization decomposes $$S_X(z)$$ into a product of a $$r_e>0$$ and the transfer function of a minimum phase system $$L(z)$$ with $$L(\infty) = 1$$ 

$$ S_X(z) = L(z)r_eL^*(z^{-*}) $$

{% endhint %}

Because $$L(z)$$ is minimum phase and $$L(\infty)=1$$, it must take the
form

$$L(z) = 1 + \sum_{i=1}^\infty l[i]z^{-i}$$

since minimum phase systems are causal. Using Definition 14, we can
express $$S_X(z)$$ as the product of a right-sided and left-sided
process.

$$S_X(z) = (\sqrt{r_e}L(z))(\sqrt{r_e}L^*(z^{-*})) = S_X^+(z)S_X^-(z)$$

Note that $$S_X^-(e^{j\omega}) = \left(S_X^+(e^{j\omega})\right)^*$$.
Using the assumptions built into Definition 14, we can find a general
form for $$L(z)$$ since we know $$S_Y(z)$$ takes the following form

$$S_Y(z) = r_e \frac{\prod_{i=1}^m(z-\alpha_i)(z^{-1}-\alpha_i^*)}{\prod_{i=1}^n(z-\beta_i)(z^{-1}-\beta_i^*)}\quad |\alpha_i| < 1, |\beta_i| < 1, r_e > 0.$$

If we let the $$z - \alpha_i$$ and $$z-\beta_i$$ terms be part of
$$L(z)$$, then

$$L(z) = z^{n-m}\frac{\prod_{i=1}^m(z-\alpha_i)}{\prod_{i=1}^n(z-\beta_i)}.$$

## Markov Processes

{% hint style="info" %}

### Definition 15

We say that random variables $$X, Y, Z$$ form a Markov Triplet $$X \textemdash Y \textemdash Z$$ if and only if $$X$$ and $$Z$$ are conditionally independent on $$Y$$

{% endhint %}

Mathematically, Markov triplets satisfy three properties.

1.  $$p(x, z | y) = p(x|y)p(z|y)$$

2.  $$p(z|x, y) = p(z|y)$$

3.  $$p(x|y, z) = p(x|y)$$

Because of these rules, the joint distribution can be written as
$$p(x, y, z) = p(x)p(y|x)p(z|y)$$.

{% hint style="info" %}

### Theorem 10

Random variables $$X,Y,Z$$ form a Markov triplet if and only if there exist $$\phi_1, \phi_2$$ such that $$p(x, y, z) = \phi_1(x, y)\phi_2(y, z)$$.

{% endhint %}

To simplify notation, we can define
$$X_m^n = \left(X_m,X_{m+1},\cdots, X_n\right)$$ and $$X^n=X_1^n$$.

{% hint style="info" %}

### Definition 16

A Markov Process is a Discrete Time Random Process $$\{X_n\}_{n\geq1}$$ where $$X_n \textemdash X_{n-1} \textemdash X^{n-2}$$ for all $$n\geq 2$$

{% endhint %}

Because of the conditional independence property, we can write the joint
distribution of all states in the Markov process as

$$p(x^n) = \prod_{t=1}^n p(x_t|x^{t-1}) = \prod_{t=1}^np(x_t|x_{t-1}).$$

The requirement for $$X \textemdash Y \textemdash Z$$ to satisfy
$$p(x, y, z) = p(x)p(y|x)p(z|y)$$ is a very strict requirement. If we
wanted to create a “wider” requirement of Markovity, then we could
settle for $$\hat{X}(Y) = \hat{X}(Y, Z)$$ where $$\hat{X}$$ is the best
linear estimator of $$X$$ since this property is satisfied by all Markov
triplets, but does not imply a Markov Triplet.

{% hint style="info" %}

### Definition 17

Random variables $$X, Y, Z$$ form a Wide Sense Markov Triplet $$X \textemdash Y \textemdash Z$$ if and only if the best linear estimator of X given Y is identical to the best linear estimator of X given Y and Z.


$$ \hat{X}(Y) = \hat{X}(Y, Z) $$

{% endhint %}

{% hint style="info" %}

### Definition 18

A stochastic process $$\{Y_i\}_{i=0}^n$$ is a Wide-Sense Markov Process if and only if for any $$1 \leq i \leq n - 1$$, $$Y_{i+1} \textemdash Y_i \textemdash Y^{i-1}$$forms a Wide-Sense Markov Triplet.

{% endhint %}

All Wide-Sense Markov models have a very succint representation.

{% hint style="info" %}

### Theorem 11

A process $$\boldsymbol{X}$$ is Wide-Sense Markov if and only if $$\boldsymbol{X}_{i+1} = F_i \boldsymbol{X}_i + G_i \boldsymbol{U}_i$$ and 

$$ \langle \begin{bmatrix} U_i \\ \boldsymbol{X}_0 \end{bmatrix}, \begin{bmatrix} U_j \\ \boldsymbol{X}_0 \end{bmatrix} \rangle  = \begin{bmatrix} Q_i \delta[i-j] & 0\\ 0 & \Pi_0 \end{bmatrix} $$

{% endhint %}

### Hidden Markov Processes

{% hint style="info" %}

### Definition 19

If $$\{X_n\}_{n\geq1}$$ is a Markov Process, then $$\{Y_n\}_{n\geq1}$$ is a Hidden Markov Process if we can factorize the conditional probability density


$$ p(y^n, x^n) = \prod_{i=1}^np(y_i|x_i) $$

{% endhint %}

We can think of $$Y$$ as a noisy observation of an underlying Markov
Process. The joint distribution of $$\{X_n\}_{n\geq1}$$ and
$$\{Y_n\}_{n\geq1}$$ can be written as

$$p(x^n, y^n) = p(x^n)p(y^n|x^n) = \prod_{t=1}^np(x_t|x_{t-1})\prod_{i=1}^np(y_i|x_i).$$

Hidden Markov Models can be represented by undirected graphical models.
To create an undirected graphical model,

1.  Create a node for each random variable.

2.  Draw an edge between two nodes if a factor of the joint distribution
    contains both nodes.

Undirected graphical models of Hidden Markov Processes are useful
because they let us derive additional Markov dependepencies between
groups of variables.

{% hint style="info" %}

### Theorem 12

For 3 disjoint sets $$S_1, S_2, S_3$$ of notes in a graphical model, if any path from $$S_1$$ to $$S_3$$ passes through a node in $$S_2$$, then $$S_1 \textemdash S_2 \textemdash S_3$$.

{% endhint %}

### State-Space Models

Suppose we have a discrete-time random process which evolves in a
recursive fashion, meaning the current state depends in some way on the
previous state. We can express this recursion with a set of equations.

{% hint style="info" %}

### Definition 20

The standard state space model describes random processes which describe the evolution of state vectors $$\boldsymbol{X}_i$$ and observation vectors $$\boldsymbol{Y}_i$$ according to the equations


$$ \begin{cases} \boldsymbol{X}_{i+1} = F_i \boldsymbol{X}_i + G_i \boldsymbol{U}_i\\ \boldsymbol{Y}_{i} = H_i\boldsymbol{X}_i + \boldsymbol{V}_i \end{cases} $$


with initial condition


$$ \langle \begin{bmatrix}\boldsymbol{X}_0 \\ \boldsymbol{U}_i \\ \boldsymbol{V}_i\end{bmatrix}, \begin{bmatrix}\boldsymbol{X}_0 \\ \boldsymbol{U}_j \\ \boldsymbol{V}_j\end{bmatrix} \rangle  = \begin{bmatrix} \Pi_0 & 0 & 0\\ 0 & Q_i\delta[i-j] & S_i\delta[i-j]\\ 0 & S_i^*\delta[i-j] & R_i\delta[i-j] \end{bmatrix} $$

{% endhint %}

From Theorem 11, we can easily see that state space models are
Wide-Sense Markov. Note that $$U_i$$ and $$V_i$$ are white noise, and
that the dynamics of the system can change at every time step. From
these equations, we can derive six different properties. Let
$$\Pi_i = \langle \boldsymbol{X}_i, \boldsymbol{X}_i \rangle $$ and
$$\Phi_{i,j} = \prod_{k=j}^{i-1}F_k$$ and $$\Phi_{i,u} = I$$.

1.  $$\forall i \geq j,\ \langle \boldsymbol{U}_i, \boldsymbol{X}_j \rangle  = 0,\ \langle \boldsymbol{V}_i, \boldsymbol{X}_j \rangle  = 0$$

2.  $$\forall i > j,\ \langle \boldsymbol{U}_i, \boldsymbol{Y}_j \rangle  = 0,\ \langle \boldsymbol{V}_i, \boldsymbol{Y}_j \rangle  = 0$$

3.  $$\forall i,\ \langle \boldsymbol{U}_i, \boldsymbol{Y}_i \rangle  = S_i,\ \langle \boldsymbol{V}_i, \boldsymbol{Y}_i \rangle  = R_i$$

4.  $$\Pi_{i+1} = F_i\Pi_iF_i^* + G_iQ_iG_i^*$$

5.  

    $$\langle \boldsymbol{X}_i, \boldsymbol{X}_j \rangle  = \begin{cases}                 \Phi_{i,j}\Pi_j & i \geq j \\                 \Pi_i \Phi_{j,i}^* & i \leq j             \end{cases}$$

6.  

    $$\langle \boldsymbol{Y}_i, \boldsymbol{Y}_j \rangle  = \begin{cases}                  H_i \Phi_{i,j+1}N_j & i > j\\                  R_i + H_i\Pi_iH_i^* & i=j \\                  N_i^*\Phi^*_{j,i+1}H_j^* & i < j             \end{cases} \text{ where } N_i=F_i\Pi_iH_i^*+G_iS_i$$

