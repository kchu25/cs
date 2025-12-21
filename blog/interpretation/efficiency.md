@def title = "Why Efficiency Matters in Explainability"
@def published = "20 December 2025"
@def tags = ["interpretation"]

# Why Efficiency Matters in Explainability

## The Efficiency Axiom

The efficiency axiom from Shapley values requires:

$$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^M \phi_i(\mathbf{x})$$

But why does this matter? The original paragraph claims it's about "correctness"—let's examine that claim.

## What Efficiency Actually Guarantees

### 1. Accounting Completeness, Not Truth

Efficiency ensures the explanation is **complete** (all contributions are accounted for), but it doesn't guarantee the explanation is **correct** in capturing how the model actually works.

- ✓ You can verify: $\sum_{i=1}^M \phi_i(\mathbf{x}) = f(\mathbf{x}) - \phi_0$
- ✗ This doesn't mean the $\phi_i$ reflect true causal mechanisms

**Key insight**: An explanation can satisfy efficiency while being completely wrong about feature interactions, causality, or model behavior.

### 2. Mathematical Consistency

Efficiency provides a consistency check. If your attribution method claims:
- Feature A contributes +0.5
- Feature B contributes +0.3  
- But total prediction difference is 1.0

Then something is mathematically inconsistent. Efficiency catches these errors.

### 3. Conservation Principle

Like conservation laws in physics, efficiency tracks where all "explanatory mass" goes:

- **Over-allocation**: Claiming more influence than exists
- **Under-allocation**: Leaving parts of the decision unexplained
- **Efficiency**: Every bit of prediction is explained exactly once

### 4. Comparability

Efficiency puts all methods on equal footing. When LIME, SHAP, and Integrated Gradients all satisfy efficiency, we can meaningfully compare their different $\phi_i$ values knowing they're all explaining the same total.

## The Correctness Confusion

The paragraph conflates **mathematical consistency** with **explanatory correctness**:

| Property | What it means |
|----------|---------------|
| Mathematically consistent | The numbers add up correctly |
| Explanatorily correct | The attribution reflects true model behavior |

You can have:
- **Efficient but misleading**: Wrong baseline, wrong interactions, wrong counterfactuals
- **Multiple valid efficient explanations**: LIME, SHAP, IG all satisfy efficiency but give different attributions

## Bottom Line

Efficiency is **necessary but not sufficient** for good explanations:

$$\text{Good Explanation} = \text{Efficiency} + \text{[Other Properties]}$$

It's about coherence and verifiability of the decomposition, not about capturing true causal structure. Think of it as a sanity check, not a truth certificate.

---

## Why Linear? A Nice Coincidence?

The additive (linear) form isn't arbitrary—it's deeply connected to efficiency, but **why linear specifically**?

### The Linearity Emerges from Completeness

Consider what we're trying to do: decompose a potentially complex function $f(\mathbf{x})$ into contributions. The efficiency requirement is:

$$f(\mathbf{x}) - \phi_0 = \sum_{i=1}^M \phi_i(\mathbf{x})$$

**Why not use a product?** 
$$f(\mathbf{x}) = \phi_0 \cdot \prod_{i=1}^M \phi_i(\mathbf{x})$$

Problems:
1. **Dimensionality**: Each $\phi_i$ would need units that make the product match $f$'s units
2. **Zero problem**: If any $\phi_i = 0$, the entire prediction becomes zero
3. **Sign issues**: Negative contributions become problematic
4. **Not additive in value**: Can't track individual contributions separately

### Game Theory Perspective

In cooperative game theory, the **value** being distributed is a scalar (e.g., money, utility). Distribution naturally follows addition:

$$\text{Total Value} = \sum_{i=1}^M \text{Player } i\text{'s share}$$

This isn't a choice—it's what "distribution" means. You're dividing a pie, not multiplying slices.

### The Deep Reason: Linearity of Expectation

Most explainability methods involve some form of expectation or averaging. The additive form naturally arises from:

$$\mathbb{E}[f(\mathbf{x})] = \mathbb{E}[\phi_0] + \sum_{i=1}^M \mathbb{E}[\phi_i(\mathbf{x})]$$

Expectation is a linear operator: $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$

So when we decompose expectations (which is what SHAP, IG, and others do), linearity falls out automatically.

### What About Interactions?

You might object: "But features interact nonlinearly!" True! But the additive form can still capture this:

$$\phi_i(\mathbf{x}) = \text{main effect of } i + \text{interaction effects involving } i$$

Each $\phi_i$ can internalize complex interactions. The linearity is in how we **sum contributions**, not in how contributions are computed.

### Is It a Coincidence?

**No.** The linear form is the unique structure that:
1. Satisfies efficiency (completeness)
2. Respects dimensionality (units match)
3. Allows both positive and negative contributions
4. Enables individual tracking of contributions
5. Aligns with how expectations and distributions work mathematically

Other functional forms (products, max, min) fail at least one of these requirements.

### The Deeper Connection

There's an equivalence:

$$\text{Additive decomposition} \iff \text{Efficiency axiom} \iff \text{Complete attribution}$$

They're three ways of saying the same thing. The linearity isn't imposed—it's the mathematical expression of "accounting for everything exactly once."

So while it might feel like a convenient coincidence that explanations are linear, it's actually the **only way** to satisfy the efficiency requirement in a mathematically coherent manner.