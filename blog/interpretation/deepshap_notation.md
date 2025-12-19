@def title = "Understanding DeepSHAP Notation and Reference Averaging"
@def published = "19 December 2025"
@def tags = ["interpretation"]

# Understanding DeepSHAP Notation and Reference Averaging

## Question 1: Why the subscript $x$ in $f_x(S)$?

### What it means

The subscript $x$ indicates **which input we're explaining**. The function $f_x(S)$ means:

> "The expected model output when features in coalition $S$ are set to their values from input $x$, and features outside $S$ are drawn from the reference distribution."

Mathematically:

$$f_x(S) = \mathbb{E}_{r \sim \mathcal{R}}[f(x_S, r_{\bar{S}})]$$

### Why we need this

We're computing Shapley values **for a specific input** $x$. Different inputs will have different Shapley values! The subscript reminds us that we're always measuring contributions relative to explaining $f(x)$.

### Concrete Example

Suppose $x = (10, 20, 30)$ (three features).

**For coalition** $S = \{1, 3\}$:

$$f_x(\{1,3\}) = \mathbb{E}_r[f(10, r_2, 30)]$$

- Features 1 and 3 are **locked** to their values in $x$: $(10, \_, 30)$
- Feature 2 varies according to the reference distribution $r_2$

**For a different input** $x' = (5, 15, 25)$:

$$f_{x'}(\{1,3\}) = \mathbb{E}_r[f(5, r_2, 25)]$$

- Now features 1 and 3 use values from $x'$: $(5, \_, 25)$
- Same coalition, but different "locked" values!

The subscript $x$ tracks which input we're explaining.

---

## Question 2: Why can marginal contribution be written as sum over references?

### The True Marginal Contribution

From the Shapley formula, the marginal contribution of feature $i$ is:

$$\phi_i = \mathbb{E}_{S \sim \pi}[f_x(S \cup \{i\}) - f_x(S)]$$

This averages over **coalitions** $S$ (all possible subsets of features).

### Expanding with References

Since $f_x(S) = \mathbb{E}_{r \sim \mathcal{R}}[f(x_S, r_{\bar{S}})]$, we can write:

$$\phi_i = \mathbb{E}_{S \sim \pi}\mathbb{E}_{r \sim \mathcal{R}}[\underbrace{f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}})}_{\text{feature } i \text{ present}} - \underbrace{f(x_S, r_{\bar{S}})}_{\text{feature } i \text{ absent}}]$$

### Why This is a "Sum" Over References

Each term in this double expectation involves:
1. A specific **coalition** $S$ (which features are active)
2. A specific **reference sample** $r$ (providing values for inactive features)

When you sample different $r$ values, you create different "background contexts" against which feature $i$'s contribution is measured.

### The Key Insight: Reference Diversity Creates Coalition Diversity

By varying $r$, you **implicitly sample different coalitions**:

- When $r_j \approx x_j$ for feature $j$: it's "almost as if" feature $j$ is in the coalition (present)
- When $r_j \ll x_j$ or $r_j \gg x_j$: feature $j$ is clearly "absent" from the coalition
- Averaging over many diverse references effectively samples the space of possible coalition configurations

### Example: How One Reference Relates to Multiple Coalitions

Suppose we have 3 features and:
- Input: $x = (10, 20, 30)$
- Reference: $r = (9, 5, 31)$

When computing $f(x) - f(r)$ with this reference:
- Feature 1: $x_1 - r_1 = 10 - 9 = 1$ (small difference → nearly "absent")
- Feature 2: $x_2 - r_2 = 20 - 5 = 15$ (large difference → clearly "present")
- Feature 3: $x_3 - r_3 = 30 - 31 = -1$ (small difference → nearly "absent")

This reference effectively captures scenarios where feature 2 is active but features 1 and 3 are not, corresponding roughly to coalition $S = \{2\}$.

### Why DeepLIFT's Single $(x, r)$ Pair Works

For one reference $r$, DeepLIFT computes:

$$C_{\Delta x_i \Delta f}(x, r) = \phi_i(x, r) \cdot (x_i - r_i)$$

This linearization captures the contribution of feature $i$ along the path from $r$ to $x$.

**By averaging over multiple references:**

$$\phi_i^{\text{DeepSHAP}} = \mathbb{E}_{r \sim \mathcal{R}}[C_{\Delta x_i \Delta f}(x, r)]$$

You're averaging these different paths, which approximates the coalition-weighted average that the Shapley formula requires.

### The Mathematical Connection

The approximation works because:

$$\mathbb{E}_{r \sim \mathcal{R}}[C_{\Delta x_i \Delta f}(x, r)] \approx \mathbb{E}_{r \sim \mathcal{R}}\mathbb{E}_{S \sim \pi}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})]$$

The left side averages DeepLIFT contributions over references. The right side is the exact Shapley value (by swapping the order of expectations).

**The bridge:** DeepLIFT's multiplier-based linearization implicitly performs the coalition sampling that Shapley requires, but does it through the network structure rather than explicit enumeration of $2^n$ coalitions.

---

## Summary

1. **The subscript $x$** reminds us we're explaining a specific input $x$—different inputs have different Shapley values.

2. **Reference averaging** works because:
   - Different references create different "background contexts"
   - This implicitly samples different feature coalitions
   - Averaging over references ≈ averaging over coalitions (the Shapley formula)
   - DeepLIFT's linearization + reference averaging = computational shortcut to Shapley values