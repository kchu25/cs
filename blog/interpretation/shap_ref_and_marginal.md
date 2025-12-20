@def title = "Why the DeepSHAP Double Expectation Formula Holds"
@def published = "19 December 2025"
@def tags = ["interpretation"]

# Why the DeepSHAP Double Expectation Formula Holds

## The Equation in Question

$$\phi_i = \mathbb{E}_{S \sim \pi}\mathbb{E}_{r \sim \mathcal{R}}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})]$$

This follows **mathematically** from the definition of $f_x(S)$, but you're right to question **why we need a distribution of references**.

## The Mathematical Reason (Why It Holds)

### Definition of $f_x(S)$

By definition:
$$f_x(S) = \mathbb{E}_{r \sim \mathcal{R}}[f(x_S, r_{\bar{S}})]$$

This says: "When features in $S$ are set to their values from $x$, and features outside $S$ are drawn from the reference distribution $\mathcal{R}$, what's the expected output?"

### Standard Shapley Formula

$$\phi_i = \mathbb{E}_{S \sim \pi}[f_x(S \cup \{i\}) - f_x(S)]$$

### Substitution

Simply substitute the definition:

$$\phi_i = \mathbb{E}_{S \sim \pi}\left[\mathbb{E}_{r \sim \mathcal{R}}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}})] - \mathbb{E}_{r \sim \mathcal{R}}[f(x_S, r_{\bar{S}})]\right]$$

By linearity of expectation:

$$\phi_i = \mathbb{E}_{S \sim \pi}\mathbb{E}_{r \sim \mathcal{R}}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})]$$

**That's it.** The equation holds by pure substitution and linearity of expectation.

## But Why Can't Reference Just Be Zero?

Good question! Let's think about what the reference distribution $\mathcal{R}$ actually represents.

### What Does $f_x(S)$ Mean?

$f_x(S)$ is supposed to answer: **"What's the expected model output when only features in $S$ are 'present'?"**

But what does "absent" mean? We need to provide *some* value for the absent features. The choice of reference distribution $\mathcal{R}$ determines what "absent" means.

### Could We Use a Single Reference (e.g., Zero)?

**Yes, mathematically you could!** If $\mathcal{R}$ is a point mass at zero (i.e., $r = 0$ always), then:

$$f_x(S) = f(x_S, 0_{\bar{S}})$$

The equation would still hold. But there are problems:

1. **Zero might not be meaningful**: For many features, zero isn't a "neutral" or "absent" value
   - Image pixels: zero = black, which is meaningful
   - Age: zero = newborn, not "absent"
   - One-hot encoded features: zero might be impossible

2. **Single reference = poor approximation**: Using one reference gives you only one "background" against which to measure feature importance. This can be misleading if the model behaves differently in different contexts.

### Why Use a Distribution of References?

The key insight: **different references create different "baseline" contexts**.

#### Example: Feature Interactions

Suppose you have two features $x_1$ and $x_2$, and $f(x_1, x_2) = x_1 \cdot x_2$.

**With single reference** $r = (0, 0)$:
- $f_x(\{1\}) = f(x_1, 0) = 0$ (regardless of $x_1$!)
- This suggests feature 1 has no contribution, which is wrong

**With multiple references** from $\mathcal{R}$:
- Some references have $r_2 > 0$, so $f(x_1, r_2) = x_1 \cdot r_2 \neq 0$
- Averaging over references captures the true contribution of $x_1$

#### The Distribution Matters for Fairness

Shapley values have a key property: they satisfy **efficiency** (the sum of attributions equals the prediction difference).

$$\sum_i \phi_i = f(x) - \mathbb{E}_{r \sim \mathcal{R}}[f(r)]$$

The choice of $\mathcal{R}$ determines what "baseline" we're measuring against. Using a distribution (e.g., the training data distribution) means:
- We're measuring each feature's contribution relative to "typical" values
- Not relative to one arbitrary choice like zero

### So Why Not Just One Reference?

**You could use one reference**, and the math would still work. But:

1. **DeepLIFT** (the predecessor to DeepSHAP) uses a single reference and computes contributions along the path from $r$ to $x$

2. **DeepSHAP** averages over multiple references to get a better approximation because:
   - Different references reveal different aspects of feature importance
   - Averaging reduces the bias from any single reference choice
   - It better approximates the true Shapley value (which implicitly integrates over all possible "backgrounds")

## Summary

- **Mathematically**, the equation holds by substitution + linearity of expectation, regardless of what $\mathcal{R}$ is
- **You could** use a single reference (including zero), and the equation would still be valid
- **But using multiple references** gives a better approximation because:
  - It captures how features contribute in different contexts
  - It reduces bias from arbitrary reference choices
  - It better represents the "expected" behavior when features are "absent"

The distribution isn't mathematically necessary for the equation to hold—it's necessary for the Shapley values to be **meaningful and accurate**.