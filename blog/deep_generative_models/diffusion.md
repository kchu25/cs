@def title = "Introduction to Diffusion Models"
@def published = "11 October 2025"
@def tags = ["deep-generative-models"]


# Introduction to Diffusion Models

## The Core Idea

Diffusion models are built on a beautifully simple insight: **it's easier to gradually destroy structure than to create it directly**. So we learn to reverse the destruction process.

Think of it this way: if I asked you to generate a realistic image from scratch, that's hard. But if I showed you a slightly noisy image and asked you to denoise it one step, that's much more tractable. Diffusion models exploit this asymmetry by learning to iteratively denoise.

## The Forward Process (Destruction)

We start with data $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ and gradually add Gaussian noise over $T$ timesteps:

> **Notation note**: $q(\mathbf{x}_0)$ is the *data distribution* (e.g., distribution of natural images). The $\mathbf{x}_0$ in parentheses just indicates "this is a distribution over the variable $\mathbf{x}_0$"—it's not a parameter. When we write $q(\mathbf{x}_t | \mathbf{x}_{t-1})$, the part after the bar is the actual conditioning/parameter.

$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$

where $\{\beta_t\}_{t=1}^T$ is a variance schedule with $\beta_t \in (0,1)$.

> **Variance schedule**: The sequence $\{\beta_1, \beta_2, \ldots, \beta_T\}$ controls how much noise we add at each step. It's called a "schedule" because we're scheduling the noise levels over time. Common choices: linear schedule ($\beta_t$ increases linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$), or cosine schedule (smoother, often works better). Small $\beta_t$ means we add a little noise; large $\beta_t$ means we add a lot. The schedule is typically designed so that $\mathbf{x}_T \approx \mathcal{N}(0, \mathbf{I})$ (pure noise).

### The Direct Sampling Trick

Here's the magic: thanks to the reparameterization trick and properties of Gaussians, we can jump to *any* timestep directly without iterating through all intermediate steps.

Why? Because adding Gaussian noise sequentially is just adding independent Gaussians, and the sum of independent Gaussians is itself Gaussian. Let's see how this works step by step:

**Important relationship:** We define $\alpha_t = 1 - \beta_t$ for convenience. This means:
- If $\beta_t$ is small (little noise added), then $\alpha_t \approx 1$ (signal mostly preserved)
- If $\beta_t$ is large (lots of noise added), then $\alpha_t$ is small (signal heavily corrupted)
- Since $\beta_t \in (0,1)$, we always have $\alpha_t \in (0,1)$ as well

**Why the square roots?** You might wonder why we write $\sqrt{\alpha_t}\mathbf{x}_{t-1}$ instead of just $\alpha_t\mathbf{x}_{t-1}$. This is crucial for variance preservation. Here's the mathematical reasoning:

Starting with the forward step: $\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}$ where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$

We compute the variance of $\mathbf{x}_t$:

$\text{Var}(\mathbf{x}_t) = \text{Var}(\sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon})$

Using the property that for independent random variables $X$ and $Y$: $\text{Var}(aX + bY) = a^2\text{Var}(X) + b^2\text{Var}(Y)$

$\text{Var}(\mathbf{x}_t) = (\sqrt{\alpha_t})^2 \text{Var}(\mathbf{x}_{t-1}) + (\sqrt{\beta_t})^2 \text{Var}(\boldsymbol{\epsilon})$

$= \alpha_t \cdot \text{Var}(\mathbf{x}_{t-1}) + \beta_t \cdot \text{Var}(\boldsymbol{\epsilon})$

Now, if we normalize our data so that $\text{Var}(\mathbf{x}_0) = 1$ initially, and note that $\text{Var}(\boldsymbol{\epsilon}) = 1$ by definition:

$\text{Var}(\mathbf{x}_t) = \alpha_t \cdot 1 + \beta_t \cdot 1 = \alpha_t + \beta_t$

Since we defined $\alpha_t = 1 - \beta_t$:

$\text{Var}(\mathbf{x}_t) = (1 - \beta_t) + \beta_t = 1$

**Key insight:** The square roots ensure that variance is preserved at every step! Without them (if we used $\alpha_t\mathbf{x}_{t-1}$), we'd have $\text{Var}(\mathbf{x}_t) = \alpha_t^2 + \beta_t \neq 1$, causing the signal to decay too quickly.

**Step 1:** Starting from $\mathbf{x}_0$, we add noise to get $\mathbf{x}_1$
- Using the reparameterization: $\mathbf{x}_1 = \sqrt{\alpha_1}\mathbf{x}_0 + \sqrt{1-\alpha_1}\boldsymbol{\epsilon}_1$
- Where $\alpha_1 = 1-\beta_1$ and $\boldsymbol{\epsilon}_1 \sim \mathcal{N}(0, \mathbf{I})$

**Step 2:** From $\mathbf{x}_1$, we add more noise to get $\mathbf{x}_2$
- Starting formula: $\mathbf{x}_2 = \sqrt{\alpha_2}\mathbf{x}_1 + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_2$
- Substitute the expression for $\mathbf{x}_1$:
  - $\mathbf{x}_2 = \sqrt{\alpha_2}(\sqrt{\alpha_1}\mathbf{x}_0 + \sqrt{1-\alpha_1}\boldsymbol{\epsilon}_1) + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_2$
- Distribute $\sqrt{\alpha_2}$:
  - $\mathbf{x}_2 = \sqrt{\alpha_2\alpha_1}\mathbf{x}_0 + \sqrt{\alpha_2(1-\alpha_1)}\boldsymbol{\epsilon}_1 + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_2$
- Combine the noise terms (since independent Gaussians add):
  - $\mathbf{x}_2 = \sqrt{\alpha_2\alpha_1}\mathbf{x}_0 + \sqrt{1-\alpha_2\alpha_1}\boldsymbol{\epsilon}$
  - Where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ is a combined noise term

**The Pattern:** Continuing this process, we see that:
- The coefficient of $\mathbf{x}_0$ becomes a product: $\sqrt{\alpha_t \alpha_{t-1} \cdots \alpha_1}$
- The noise variance accumulates: $1 - \alpha_t \alpha_{t-1} \cdots \alpha_1$
- We can define $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ for convenience

**The Final Result:** We can collapse this entire chain into a single step:

$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$

In other words: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$.

As $t \to T$, we have $\bar{\alpha}_T \to 0$, so $\mathbf{x}_T$ becomes pure noise. The forward process is fixed—no learning happens here.

## The Reverse Process (Creation)

Now for the learned part. We want to reverse the diffusion:

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

Starting from $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$, we iteratively sample backwards to generate $\mathbf{x}_0$.

The key question: what should $\boldsymbol{\mu}_\theta$ predict?

## What to Predict: The Clever Reparameterization

Here's where the elegance shines. We *could* predict $\boldsymbol{\mu}_\theta$ directly, but there's a better way. 

From Bayes' rule, the true reverse process posterior is:

$$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t\mathbf{I})$$

where 

$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t$$

Since $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$, we can solve for $\mathbf{x}_0$:

$$\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon})$$

Substituting this back into $\tilde{\boldsymbol{\mu}}_t$, we get:

$$\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}\right)$$

**The insight**: instead of predicting $\boldsymbol{\mu}_\theta$ directly, we can train a network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ to predict the noise $\boldsymbol{\epsilon}$!

## The Training Objective

The variational lower bound (ELBO) leads to:

$$L = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

where $t \sim \text{Uniform}(1, T)$ and $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$.

That's it! We're just doing **noise prediction** at random timesteps. The loss is a simple mean squared error.

## Sampling

To generate:

1. Sample $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$
2. For $t = T, T-1, \ldots, 1$:
   - Predict noise: $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$
   - Compute mean: $\boldsymbol{\mu}_\theta = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$
   - Sample: $\mathbf{x}_{t-1} = \boldsymbol{\mu}_\theta + \sigma_t \mathbf{z}$ where $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$
3. Return $\mathbf{x}_0$

## Why This Works: The Intuition

At each timestep $t$, the network sees $\mathbf{x}_t$ which is a specific signal-to-noise mixture of the original data. The network learns to denoise by predicting what noise was added. Early timesteps ($t$ small) are easy: there's mostly signal, little noise. Later timesteps ($t$ large) are harder: mostly noise, little signal.

By training on all timesteps simultaneously, the network learns a hierarchy of denoisers. High $t$: rough structure. Low $t$: fine details.

The stochastic sampling process ensures diversity—we can generate infinitely many samples from the same pure noise starting point.

## Connection to Score-Based Models

Here's a beautiful connection: predicting the noise $\boldsymbol{\epsilon}_\theta$ is equivalent to predicting the score function $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$.

Specifically:

$$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$

This unifies diffusion models with score matching—they're learning the gradient of the log density at different noise levels. This is langevin dynamics in disguise!

## Why Diffusion Models Shine

- **Stable training**: No adversarial dynamics like GANs
- **High quality**: State-of-the-art sample quality
- **Flexible**: Easy to condition, edit, and control
- **Principled**: Solid probabilistic foundation via ELBO

The main drawback? Slow sampling due to the iterative process. But that's an active research area with methods like DDIM, DPM-Solver, and consistency models speeding things up dramatically.

---

*This is the essence of diffusion models: learn to predict noise at every stage of corruption, then reverse the process to create.*