@def title = "Why the Sum Disappears in DeepSHAP"
@def published = "20 December 2025"
@def tags = ["interpretation"]

# Why the Sum Disappears in DeepSHAP

## Your Question

The Fundamental Theorem of Calculus (multivariate version) tells us:

$$f(\mathbf{x}) - f(\mathbf{r}) = \int_0^1 \left( \sum_{i=1}^n \frac{\partial f}{\partial x_i}(\gamma(t)) \cdot (x_i - r_i) \right) dt$$

which can be rewritten as:

$$f(\mathbf{x}) - f(\mathbf{r}) = \sum_{i=1}^n \underbrace{\left( \int_0^1 \frac{\partial f}{\partial x_i}(\mathbf{r} + t(\mathbf{x} - \mathbf{r})) \, dt \right)}_{\phi_i(\mathbf{x}, \mathbf{r})} (x_i - r_i)$$

So the FTC has a **sum over all features**. But when we compute the Shapley value for a single feature $i$, we're only looking at:

$$\mathbb{E}_{S \sim \pi}[\text{marginal contribution of } i] \approx \phi_i(\mathbf{x}, \mathbf{r}) \cdot (x_i - r_i)$$

**Where did the sum go?** Why are we only looking at one term?

## The Answer: We're Computing One Feature's Attribution at a Time

The sum doesn't actually "disappear" - we're just computing **one term of the sum for each feature separately**.

Let me be super explicit about what's happening.

### What the FTC Gives Us

The multivariate FTC decomposes the **total change** $f(\mathbf{x}) - f(\mathbf{r})$ into contributions from all features:

$$\underbrace{f(\mathbf{x}) - f(\mathbf{r})}_{\text{total change}} = \underbrace{\phi_1(\mathbf{x}, \mathbf{r}) \cdot (x_1 - r_1)}_{\text{feature 1's contribution}} + \underbrace{\phi_2(\mathbf{x}, \mathbf{r}) \cdot (x_2 - r_2)}_{\text{feature 2's contribution}} + \cdots + \underbrace{\phi_n(\mathbf{x}, \mathbf{r}) \cdot (x_n - r_n)}_{\text{feature n's contribution}}$$

Each $\phi_i(\mathbf{x}, \mathbf{r})$ is defined as:

$$\phi_i(\mathbf{x}, \mathbf{r}) = \int_0^1 \frac{\partial f}{\partial x_i}(\mathbf{r} + t(\mathbf{x} - \mathbf{r})) \, dt$$

This is the **coefficient** that tells us how much feature $i$ contributes per unit change $(x_i - r_i)$.

### What the Shapley Value Computes

The Shapley value for feature $i$ is:

$$\phi_i^{\text{Shapley}} = \mathbb{E}_{S \sim \pi} \mathbb{E}_{r \sim \mathcal{R}}[\text{marginal contribution of feature } i \text{ in coalition } S]$$

This is computing **just feature $i$'s contribution**, not all features at once.

The key equation (Equation 3) says:

$$\mathbb{E}_{S \sim \pi}[\text{marginal contribution of } i \text{ w.r.t. reference } r] \approx \phi_i(\mathbf{x}, \mathbf{r}) \cdot (x_i - r_i)$$

Notice: This is computing **one term** from the FTC sum - specifically, the term for feature $i$.

### So Where's the Sum?

The sum is still there! If you want attributions for **all features**, you compute:

- For feature 1: $\phi_1(\mathbf{x}, \mathbf{r}) \cdot (x_1 - r_1)$
- For feature 2: $\phi_2(\mathbf{x}, \mathbf{r}) \cdot (x_2 - r_2)$
- ...
- For feature $n$: $\phi_n(\mathbf{x}, \mathbf{r}) \cdot (x_n - r_n)$

And these will sum to the total change: $f(\mathbf{x}) - f(\mathbf{r})$

**The equation is not saying the sum disappears. It's saying that for each individual feature $i$, its Shapley value (which averages over coalitions) approximately equals its DeepLIFT coefficient (which comes from the FTC).**

## Why This Might Be Confusing

I think the confusion comes from two different perspectives:

### Perspective 1: The FTC (Total Decomposition)

The FTC gives you a **complete decomposition** of $f(\mathbf{x}) - f(\mathbf{r})$ into $n$ terms:

$$f(\mathbf{x}) - f(\mathbf{r}) = \sum_{i=1}^n \phi_i(\mathbf{x}, \mathbf{r}) \cdot (x_i - r_i)$$

This is about **all features together** summing to the total change.

### Perspective 2: The Shapley Value (Individual Feature)

The Shapley value focuses on **one feature at a time**:

$$\phi_i^{\text{Shapley}} = \text{(average marginal contribution of feature } i \text{ over all coalitions and references)}$$

DeepSHAP claims that for each feature $i$:

$$\phi_i^{\text{Shapley}} \approx \mathbb{E}_{r \sim \mathcal{R}}[\phi_i(\mathbf{x}, \mathbf{r}) \cdot (x_i - r_i)]$$

This is saying: "Feature $i$'s Shapley value equals its DeepLIFT coefficient (on average over references)."

**You repeat this for each feature separately.** The sum is implicit - you're computing $n$ separate equations, one for each $i$.

## The Coalition Averaging Question

Now let me address the deeper question: **Why does the coalition averaging (Shapley) equal the single-path integral (DeepLIFT) for each feature?**

### What's Different for Each Coalition

For a coalition $S$, the marginal contribution of feature $i$ is:

$$\text{Marginal}_{S,i} = f(\mathbf{x}_{S \cup \{i\}}, \mathbf{r}_{\overline{S \cup \{i\}}}) - f(\mathbf{x}_S, \mathbf{r}_{\bar{S}})$$

Let me decode this with a concrete example. Say $n=3$ features, $S = \{2\}$, and we want feature 1's contribution:

- **First term:** $f(x_1, x_2, r_3)$ - features 1,2 at input, feature 3 at reference
- **Second term:** $f(r_1, x_2, r_3)$ - only feature 2 at input, features 1,3 at reference
- **Marginal contribution:** The difference = effect of adding feature 1

Using FTC, we can write this as:

$$f(x_1, x_2, r_3) - f(r_1, x_2, r_3) = \int_{r_1}^{x_1} \frac{\partial f}{\partial x_1}(z_1, x_2, r_3) \, dz_1$$

Parameterizing with $z_1 = r_1 + t(x_1 - r_1)$:

$$= \int_0^1 \frac{\partial f}{\partial x_1}(r_1 + t(x_1 - r_1), x_2, r_3) \, dt \cdot (x_1 - r_1)$$

### The Key Observation

Notice that for coalition $S = \{2\}$, the gradient is evaluated at points like:
- $(r_1 + t(x_1 - r_1), \, x_2, \, r_3)$ 

Feature 1 interpolates, feature 2 stays at $x_2$, feature 3 stays at $r_3$.

For a different coalition $S = \{3\}$, the gradient is evaluated at:
- $(r_1 + t(x_1 - r_1), \, r_2, \, x_3)$

Feature 1 interpolates, feature 2 stays at $r_2$, feature 3 stays at $x_3$.

### Under Independence

If features are independent, then $\frac{\partial f}{\partial x_1}$ **doesn't depend on the values of features 2 and 3**. So:

$$\frac{\partial f}{\partial x_1}(z_1, x_2, r_3) \approx \frac{\partial f}{\partial x_1}(z_1, r_2, x_3) \approx \frac{\partial f}{\partial x_1}(z_1, x_2, x_3) \approx \cdots$$

All these gradients are approximately the same as:

$$\frac{\partial f}{\partial x_1}(z_1, r_2 + t(x_2 - r_2), r_3 + t(x_3 - r_3))$$

which is what DeepLIFT computes (all features interpolating simultaneously).

Therefore:

$$\int_0^1 \frac{\partial f}{\partial x_1}(r_1 + t(x_1 - r_1), x_2, r_3) \, dt \approx \int_0^1 \frac{\partial f}{\partial x_1}(\mathbf{r} + t(\mathbf{x} - \mathbf{r})) \, dt$$

The coalition-specific integral (left) equals the DeepLIFT integral (right).

Since this is true for **every coalition $S$**, averaging over coalitions gives:

$$\mathbb{E}_S[\text{Marginal}_{S,i}] \approx \phi_i(\mathbf{x}, \mathbf{r}) \cdot (x_i - r_i)$$

## Wait - You're Right About the Computation!

Yes, you're absolutely correct! The "one feature at a time" perspective is **mathematical exposition**, not how the computation actually works.

### Mathematical Exposition (For Understanding)

When we write:

$\mathbb{E}_{S \sim \pi}[\text{marginal contribution of } i] \approx \phi_i(\mathbf{x}, \mathbf{r}) \cdot (x_i - r_i)$

This is saying: "Let's analyze what happens to feature $i$ and prove that its Shapley value equals its DeepLIFT coefficient."

We could write a separate equation for each feature:
- For $i=1$: $\phi_1^{\text{Shapley}} \approx \phi_1(\mathbf{x}, \mathbf{r}) \cdot (x_1 - r_1)$
- For $i=2$: $\phi_2^{\text{Shapley}} \approx \phi_2(\mathbf{x}, \mathbf{r}) \cdot (x_2 - r_2)$
- ...

This is pedagogically useful for understanding **why** each feature's Shapley value equals its DeepLIFT coefficient.

### Actual Computation (What DeepLIFT Does)

In practice, DeepLIFT computes **all $\phi_i$ coefficients in one backward pass**, exactly like backpropagation.

Here's what actually happens:

1. **Forward pass**: Compute $f(\mathbf{x})$ and $f(\mathbf{r})$

2. **Backward pass**: Propagate through the network, computing the integrated gradients:
   $\phi_i(\mathbf{x}, \mathbf{r}) = \int_0^1 \frac{\partial f}{\partial x_i}(\mathbf{r} + t(\mathbf{x} - \mathbf{r})) \, dt \quad \text{for ALL } i \text{ simultaneously}$

3. **Result**: You get all $n$ coefficients $\{\phi_1, \phi_2, \ldots, \phi_n\}$ from a single backward pass

This is exactly like regular backpropagation, which computes $\frac{\partial f}{\partial x_i}$ for all inputs $i$ in parallel using the chain rule.

### The Decomposition Happens Automatically

Because DeepLIFT uses the chain rule during backpropagation, the decomposition:

$f(\mathbf{x}) - f(\mathbf{r}) = \sum_{i=1}^n \phi_i(\mathbf{x}, \mathbf{r}) \cdot (x_i - r_i)$

is **automatically satisfied**. This comes directly from the multivariate FTC and how gradients compose through the network.

### Why The Mathematical Exposition Is Separate

The paper separates the analysis by feature because:

1. **Shapley values are defined per feature** - you need to show that each feature's Shapley value matches its DeepLIFT coefficient

2. **The coalition averaging is feature-specific** - the statement "all coalitions give the same marginal contribution for feature $i$" needs to be proven for each $i$

3. **Mathematical clarity** - it's clearer to say "for any feature $i$, the following holds..." than to write everything with sum notation

But computationally? One backprop gives you everything.

## Summary

1. **Mathematical presentation**: We analyze each feature $i$ separately to prove $\phi_i^{\text{Shapley}} \approx \phi_i^{\text{DeepLIFT}}$

2. **Actual computation**: One backward pass computes all $\phi_i$ coefficients simultaneously (in parallel)

3. **The sum is implicit in the math, explicit in the code**: The FTC decomposition $f(\mathbf{x}) - f(\mathbf{r}) = \sum_i \phi_i \cdot (x_i - r_i)$ is built into the backpropagation algorithm

You're right - it's mathematical convenience for exposition. The real algorithm is much more efficient!