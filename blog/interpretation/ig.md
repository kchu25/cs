@def title = "Integrated Gradients: A Clear Mathematical Explanation"
@def published = "4 January 2026"
@def tags = ["interpretation"]

# Integrated Gradients: A Clear Mathematical Explanation

## The Core Problem

**Attribution**: Given a neural network $F: \mathbb{R}^n \to [0,1]$ and an input $\mathbf{x}$, how much did each feature $x_i$ contribute to the prediction $F(\mathbf{x})$?

The key insight: we need a **baseline** $\mathbf{x}'$ representing "absence of signal" (e.g., black image, zero embedding). Attribution measures contribution *relative* to this baseline.

---

## Why Gradients Fail

The naive approach: use $\frac{\partial F}{\partial x_i} \cdot x_i$.

**Problem (Sensitivity violation)**: Consider $f(x) = 1 - \text{ReLU}(1-x)$.
- At baseline $x=0$: $f(0) = 0$
- At input $x=2$: $f(2) = 1$ 
- But gradient at $x=2$ is **zero** (function is flat there!)
- So naive gradient gives zero attribution despite the feature completely changing the output.

**The issue**: Gradients only capture *local* behavior at one point. They miss what happened along the path from baseline to input.

---

## The Solution: Integrated Gradients

Instead of looking at just one point, **accumulate gradients along the entire path** from baseline to input.

### Mathematical Definition

For the $i$-th feature:

$$\text{IntegratedGrad}_i(\mathbf{x}) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} d\alpha$$

**Translation**: 
- $\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}')$ is a point on the straight line from baseline to input (parameterized by $\alpha \in [0,1]$)
- We compute the gradient at *every* point along this line
- We integrate (sum up) all these gradients
- We scale by $(x_i - x'_i)$ to get the final attribution

### Practical Implementation

In practice, approximate the integral with a Riemann sum using $m$ steps:

$$\text{IntegratedGrad}_i(\mathbf{x}) \approx (x_i - x'_i) \times \sum_{k=1}^{m} \frac{\partial F(\mathbf{x}' + \frac{k}{m}(\mathbf{x} - \mathbf{x}'))}{\partial x_i} \times \frac{1}{m}$$

This is just: *compute gradients at $m$ evenly-spaced points along the path, then average them*.

In code, it's a simple loop calling your gradient function! The paper recommends $m = 20$ to $300$ steps.

---

## Why This Works: The Axioms

The paper proves Integrated Gradients is the *unique* method satisfying these desirable properties:

### 1. Completeness (Sanity Check)
$$\sum_{i=1}^{n} \text{IntegratedGrad}_i(\mathbf{x}) = F(\mathbf{x}) - F(\mathbf{x}')$$

The attributions **sum exactly** to the difference in prediction. If the baseline has near-zero score, attributions sum to the final prediction.

*Proof*: This is just the fundamental theorem of calculus for path integrals!

### 2. Sensitivity(a)

**Definition**: If baseline $\mathbf{x}'$ and input $\mathbf{x}$ differ only in feature $i$ (i.e., $x_j = x'_j$ for all $j \neq i$), and $F(\mathbf{x}) \neq F(\mathbf{x}')$, then:

$\text{Attribution}_i(\mathbf{x}, \mathbf{x}') \neq 0$

**Why Integrated Gradients satisfies this**:

From Completeness, we know:
$\sum_{j=1}^{n} \text{IntegratedGrad}_j(\mathbf{x}) = F(\mathbf{x}) - F(\mathbf{x}')$

If only feature $i$ differs, then for all $j \neq i$: $x_j = x'_j$, which means:
$(x_j - x'_j) = 0 \implies \text{IntegratedGrad}_j(\mathbf{x}) = 0$

So the completeness equation becomes:
$\text{IntegratedGrad}_i(\mathbf{x}) = F(\mathbf{x}) - F(\mathbf{x}')$

Since $F(\mathbf{x}) \neq F(\mathbf{x}')$ by assumption, we must have $\text{IntegratedGrad}_i(\mathbf{x}) \neq 0$. ∎

**Sensitivity(b)** (also called "Dummy" axiom):

If feature $x_i$ has *no effect* on the function (i.e., $F(\mathbf{x})$ is independent of $x_i$ for all $\mathbf{x}$), then:
$\text{Attribution}_i(\mathbf{x}, \mathbf{x}') = 0 \text{ for all } \mathbf{x}, \mathbf{x}'$

**Proof for Integrated Gradients**: If $F$ doesn't depend on $x_i$, then $\frac{\partial F}{\partial x_i} = 0$ everywhere. Therefore:
$\text{IntegratedGrad}_i(\mathbf{x}) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \underbrace{\frac{\partial F}{\partial x_i}}_{=0} d\alpha = 0$

### 3. Implementation Invariance

**Definition**: Suppose networks $F$ and $G$ compute the same function (i.e., $F(\mathbf{x}) = G(\mathbf{x})$ for all $\mathbf{x}$), even though they have different architectures/parameters. Then their attributions must be identical:

$\text{Attribution}_F(\mathbf{x}, \mathbf{x}') = \text{Attribution}_G(\mathbf{x}, \mathbf{x}')$

**Why Integrated Gradients satisfies this**:

If $F(\mathbf{x}) = G(\mathbf{x})$ for all $\mathbf{x}$, then by calculus:
$\frac{\partial F}{\partial x_i}(\mathbf{x}) = \frac{\partial G}{\partial x_i}(\mathbf{x}) \text{ for all } \mathbf{x}, i$

This is the fundamental principle: the gradient of a function depends only on the *function itself*, not how it's implemented.

Therefore:
$\begin{align}
\text{IntegratedGrad}^F_i(\mathbf{x}) &= (x_i - x'_i) \int_{\alpha=0}^{1} \frac{\partial F(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} d\alpha \\
&= (x_i - x'_i) \int_{\alpha=0}^{1} \frac{\partial G(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} d\alpha \\
&= \text{IntegratedGrad}^G_i(\mathbf{x})
\end{align}$

**The chain rule connection**: This property is why gradients compose nicely. If $F(\mathbf{x}) = h(g(\mathbf{x}))$, then:
$\frac{\partial F}{\partial x_i} = \frac{\partial h}{\partial g} \cdot \frac{\partial g}{\partial x_i}$

The chain rule works regardless of the "implementation detail" $h$. But discrete gradients $\frac{\Delta F}{\Delta x_i}$ don't satisfy a chain rule:
$\frac{F(x_1) - F(x_0)}{x_1 - x_0} \neq \frac{F(x_1) - F(x_0)}{h(x_1) - h(x_0)} \cdot \frac{h(x_1) - h(x_0)}{x_1 - x_0}$

This is why DeepLift and LRP, which use discrete gradients, violate Implementation Invariance.

### 4. Symmetry Preservation
If two features are interchangeable in the function (e.g., $F(x_1, x_2) = F(x_2, x_1)$), and they have equal values at input and baseline, they should get equal attributions.

The straight-line path is the **only** path that guarantees this!

---

## Why Other Methods Fail

| Method | What It Does | Why It Fails |
|--------|-------------|--------------|
| **Gradients** | $\nabla F(\mathbf{x}) \odot \mathbf{x}$ | Violates Sensitivity (misses flat regions) |
| **DeepLift / LRP** | "Discrete gradients" via backprop | Violates Implementation Invariance (chain rule doesn't hold for discrete gradients) |
| **Deconvolution / Guided Backprop** | Modified backprop (only pass positive signals) | Violates Sensitivity (ignores features where ReLU is off at input) |

---

## The Mathematical Uniqueness Result

**Theorem**: Integrated Gradients is the *unique* path-based attribution method that is symmetry-preserving.

**Path method** = any method that integrates gradients along *some* path from baseline to input.

The straight-line path $\mathbf{x}'  + \alpha(\mathbf{x} - \mathbf{x}')$ is special because:
1. It's the simplest path mathematically
2. It's the only one that preserves symmetry

*Alternative*: Shapley values average over all $n!$ coordinate-wise paths, but this is computationally prohibitive for high-dimensional inputs like images.

---

## Practical Considerations

### Choosing the Baseline
- **Images**: Black image (all pixels = 0)
- **Text**: Zero embedding vector
- **Requirement**: Baseline should have near-zero prediction score and represent "absence of signal"

### Implementation Tips
1. Use $m = 20$ to $300$ steps
2. Verify: $\sum_i \text{IntegratedGrad}_i(\mathbf{x}) \approx F(\mathbf{x}) - F(\mathbf{x}')$ (within 5%)
3. If not close, increase $m$

### Visualization
For images: scale pixel intensities by their attributions to see which pixels drove the prediction.

---

## Intuition

Think of it this way: 

- You walk from the baseline (black image) to your input (actual image)
- At each step, you ask: "If I stopped here, which features would locally matter most?"
- You sum up all these local answers to get the global attribution

This avoids the pitfalls of just looking at one local neighborhood (gradients) or using non-composable discrete steps (DeepLift/LRP).