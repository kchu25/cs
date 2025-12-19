@def title = "Making the SHAP Approximation Intuitive"
@def published = "19 December 2025"
@def tags = ["interpretation"]

# Understanding SHAP's Independence Assumption

Hey! Let me clarify the core confusion about DeepSHAP and how independence makes it work. I'll focus on making the independence assumption crystal clear.

## The Setup: What We're Trying to Compute

The true Shapley value for feature $i$ is:

$$\phi_i^{\text{Shapley}} = \mathbb{E}_{S \sim \pi} \mathbb{E}_{r \sim \mathcal{R}}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})]$$

This has two expectations:
- **Inner**: average over all coalitions $S$ (exponentially many!)
- **Outer**: average over reference samples $r$

Your question: How can one backprop (DeepLIFT) approximate this exponential average?

**Answer**: Independence makes all coalition terms equal, so the exponential average reduces to a single calculation.

---

## Understanding "Independence" - The Source of Confusion

There are TWO different things called "independence" here, which causes confusion:

### 1. Independence in the Reference Distribution (Probabilistic)

When we write $r \sim \mathcal{R}$, we're sampling from a reference distribution. The assumption is:

$p(r_1, r_2, \ldots, r_n) = p(r_1) \cdot p(r_2) \cdots p(r_n)$

**What this means**: When sampling references, we draw each feature independently from its marginal distribution.

**Example**: If training data has height and weight correlated, we're assuming we can sample heights and weights independently (which is unrealistic but makes computation easier).

**This is about how we construct $r$**.

### 2. Independence in the Model (Functional)

The model $f(x_1, \ldots, x_n)$ has "independent features" if:

$\frac{\partial^2 f}{\partial x_i \partial x_j} = 0 \text{ for all } i \neq j$

**What this means**: The model has no interaction terms. The effect of feature $i$ doesn't depend on the value of feature $j$.

**This is about the structure of $f$**.

### CRITICAL POINT: These Are NOT the Same Thing!

**They don't imply each other!** This is a key source of confusion:

- **Probabilistic independence** is an assumption about how we sample reference data
- **Functional independence** is a property of the model architecture

**The papers DON'T claim that probabilistic independence implies functional independence!** 

What actually happens:
1. The papers ASSUME functional independence (no interaction terms in the model)
2. Given that assumption, they show DeepLIFT approximates Shapley values
3. Probabilistic independence in references is a separate practical choice for sampling

**Reality check**: Most models have interactions (functional dependence), and most features are correlated (probabilistic dependence). DeepSHAP is useful despite violating both assumptions, but it's only theoretically exact when both hold.

The functional independence is the key one for understanding why one backprop works!

---

## The Key Insight: Functional Independence Makes Coalitions Irrelevant

Let me show you EXACTLY what independence means and what gets "canceled."

### Notation Breakdown

$$\frac{\partial f}{\partial x_i}(x_S, z_i, r_{\overline{S \cup \{i\}}})$$

This means:
- For features $j \in S$: use $x_j$ 
- For feature $i$: use $z_i$
- For features $j \notin S$ and $j \neq i$: use $r_j$

**Concrete example with $n=5$ features, $S = \{2, 5\}$, $i=3$:**

$$\frac{\partial f}{\partial x_3}(\underbrace{r_1}_{\text{not in } S}, \underbrace{x_2}_{\text{in } S}, \underbrace{z_3}_{i}, \underbrace{r_4}_{\text{not in } S}, \underbrace{x_5}_{\text{in } S})$$

Different coalitions give different combinations of which features are at $x$ vs $r$.

---

## Example 1: Linear Model (Perfect Independence)

$$f(x) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_4 + \beta_5 x_5$$

The gradient with respect to $x_3$ is:

$$\frac{\partial f}{\partial x_3} = \beta_3$$

**Key observation**: This is a CONSTANT. It doesn't depend on any feature values at all!

So:
- Coalition $S = \{2, 5\}$: $\frac{\partial f}{\partial x_3}(r_1, x_2, z_3, r_4, x_5) = \beta_3$
- Coalition $S = \emptyset$: $\frac{\partial f}{\partial x_3}(r_1, r_2, z_3, r_4, r_5) = \beta_3$
- Coalition $S = \{1,2,4,5\}$: $\frac{\partial f}{\partial x_3}(x_1, x_2, z_3, x_4, x_5) = \beta_3$

**All gradients equal $\beta_3$!** The coalition $S$ is completely irrelevant.

### What This Means for Shapley

For any coalition $S$ and reference $r$:

$f(x_{S \cup \{3\}}, r_{\overline{S \cup \{3\}}}) - f(x_S, r_{\bar{S}}) = \int_{r_3}^{x_3} \frac{\partial f}{\partial x_3} dz_3 = \beta_3 (x_3 - r_3)$

**Let me write out the missing steps explicitly:**

**Step 1**: What are we computing?
- Left side: $f(x_{S \cup \{3\}}, r_{\overline{S \cup \{3\}}})$ means feature 3 changes from $r_3$ to $x_3$, all others stay fixed
- More explicitly: $f(\ldots, x_2, x_3, \ldots) - f(\ldots, x_2, r_3, \ldots)$ where the $\ldots$ represents other features at either $x$ or $r$ depending on coalition $S$

**Step 2**: Apply Fundamental Theorem of Calculus (1D version)

When we vary only one variable from $r_3$ to $x_3$ while holding all others constant:

$f(\ldots, x_3, \ldots) - f(\ldots, r_3, \ldots) = \int_{r_3}^{x_3} \frac{\partial f}{\partial z_3}(\ldots, z_3, \ldots) \, dz_3$

Here $z_3$ is a dummy variable of integration representing feature 3's value as it varies from $r_3$ to $x_3$.

**Step 3**: Use the specific form of $f$ (linear model)

For our linear model: $f(x) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_4 + \beta_5 x_5$

The partial derivative is:
$\frac{\partial f}{\partial z_3} = \beta_3$

This is a **constant** - it doesn't depend on $z_3$ or any other variable!

**Step 4**: Substitute into the integral

$\int_{r_3}^{x_3} \frac{\partial f}{\partial z_3} \, dz_3 = \int_{r_3}^{x_3} \beta_3 \, dz_3$

**Step 5**: Evaluate the integral

Since $\beta_3$ is constant:
$\int_{r_3}^{x_3} \beta_3 \, dz_3 = \beta_3 \cdot z_3 \Big|_{r_3}^{x_3} = \beta_3 (x_3 - r_3)$

**Putting it all together:**

$f(x_{S \cup \{3\}}, r_{\overline{S \cup \{3\}}}) - f(x_S, r_{\bar{S}}) = \int_{r_3}^{x_3} \frac{\partial f}{\partial z_3} \, dz_3 = \int_{r_3}^{x_3} \beta_3 \, dz_3 = \beta_3 (x_3 - r_3)$

**The key point**: Because $\frac{\partial f}{\partial x_3} = \beta_3$ is constant (independent of all feature values), the integral gives the same result **regardless of which coalition $S$ we're in**. The values of features in $S$ don't appear anywhere in this calculation!

Every coalition gives the same marginal contribution: $\beta_3 (x_3 - r_3)$

Therefore:
$\mathbb{E}_S[\text{marginal}] = \beta_3 (x_3 - r_3)$

**The exponential average reduced to a single calculation!**

---

## Example 2: Model with Interactions (Independence VIOLATED)

$$f(x) = \beta_1 x_1 + \beta_2 x_2 + \beta_{12} x_1 x_2$$

The gradient with respect to $x_1$ is:

$$\frac{\partial f}{\partial x_1} = \beta_1 + \beta_{12} x_2$$

**Key observation**: This DEPENDS on $x_2$!

So:
- Coalition $S = \emptyset$: $\frac{\partial f}{\partial x_1}(z_1, r_2) = \beta_1 + \beta_{12} r_2$
- Coalition $S = \{2\}$: $\frac{\partial f}{\partial x_1}(z_1, x_2) = \beta_1 + \beta_{12} x_2$

**Different coalitions give different gradients!** If $x_2 \neq r_2$, these are not equal.

### What This Means for Shapley

The marginal contributions are:
- Empty coalition: $(\beta_1 + \beta_{12} r_2)(x_1 - r_1)$
- With feature 2: $(\beta_1 + \beta_{12} x_2)(x_1 - r_1)$

These are DIFFERENT. You must actually compute and average them. No shortcut!

---

## Example 3: Neural Network (Approximate Independence)

$$f(x) = \text{ReLU}(w_1 x_1 + w_2 x_2 + w_3 x_3 + b)$$

In regions where the ReLU is active ($w_1 x_1 + w_2 x_2 + w_3 x_3 + b > 0$):

$$\frac{\partial f}{\partial x_1} = w_1$$

**Key observation**: Within a linear region, gradient is constant (like the linear model)!

So:
- Coalition $S = \{2\}$: $\frac{\partial f}{\partial x_1}(z_1, x_2, r_3) = w_1$ (if active)
- Coalition $S = \{3\}$: $\frac{\partial f}{\partial x_1}(z_1, r_2, x_3) = w_1$ (if active)

**Approximately equal** as long as both points lie in the same linear region.

### The Catch

Different coalitions might activate different neurons, putting you in different linear regions. Then the gradients differ and independence is violated.

**DeepLIFT's approximation**: Assumes you stay in the same linear regions, so gradients are approximately equal across coalitions.

---

## The Mathematical Statement of Functional Independence

**Independence means**: The gradient with respect to $x_i$ can be written as:

$$\frac{\partial f}{\partial x_i}(x_1, \ldots, x_n) = g_i(x_i)$$

A function that ONLY depends on $x_i$, not on any other feature!

**Consequence**: 

$$\frac{\partial f}{\partial x_i}(x_S, z_i, r_{\overline{S \cup \{i\}}}) = g_i(z_i) = \frac{\partial f}{\partial x_i}(r_1, \ldots, r_{i-1}, z_i, r_{i+1}, \ldots, r_n)$$

The stuff that varies with coalition ($x_S$ vs $r_{\overline{S \cup \{i\}}}$) doesn't appear in the gradient because it only depends on $z_i$.

**This is what "cancels out"**: All the coalition-specific feature values drop out of the gradient expression.

---

## How This Makes DeepSHAP Work

### Step 1: For a Fixed Reference $r$

The marginal contribution in coalition $S$ is:

$$\Delta_i(S) = \int_0^1 \frac{\partial f}{\partial x_i}(x_S, r_i + t(x_i - r_i), r_{\overline{S \cup \{i\}}}) dt \cdot (x_i - r_i)$$

**Under independence**, gradient doesn't depend on coalition:

$$\Delta_i(S) = \int_0^1 g_i(r_i + t(x_i - r_i)) dt \cdot (x_i - r_i)$$

This is the SAME for all coalitions $S$!

### Step 2: Coalition Average Collapses

$$\mathbb{E}_{S \sim \pi}[\Delta_i(S)] = \int_0^1 g_i(r_i + t(x_i - r_i)) dt \cdot (x_i - r_i)$$

We computed an expectation over $2^{n-1}$ coalitions, but got a single value because all terms were identical.

### Step 3: DeepLIFT Approximates This

DeepLIFT integrates along the full path from $r$ to $x$:

$$\phi_i(x, r) = \int_0^1 \frac{\partial f}{\partial x_i}(r + t(x - r)) dt$$

Under independence, this approximately equals:

$$\int_0^1 g_i(r_i + t(x_i - r_i)) dt$$

Because the gradient only depends on $x_i$, not on where the other features are along the path.

### Step 4: Multiple References

Finally, we average over references:

$$\phi_i^{\text{DeepSHAP}} = \mathbb{E}_{r \sim \mathcal{R}}[\phi_i(x, r) \cdot (x_i - r_i)] \approx \frac{1}{K} \sum_{k=1}^K \phi_i(x, r^{(k)}) \cdot (x_i - r_i^{(k)})$$

---

## Summary: What Independence Does

**Functional independence** ($\frac{\partial^2 f}{\partial x_i \partial x_j} = 0$) means:

1. The gradient $\frac{\partial f}{\partial x_i}$ only depends on $x_i$, not other features
2. Therefore, all $2^{n-1}$ coalitions give the same marginal contribution
3. The coalition expectation $\mathbb{E}_S[\cdot]$ reduces from "average over exponentially many terms" to "evaluate one term"
4. DeepLIFT's single path integral gives that one value
5. Multiple references approximate the outer expectation $\mathbb{E}_r[\cdot]$

**Without independence**: Different coalitions give different gradients → must actually compute exponentially many terms → DeepSHAP is just an approximation.

**With independence**: All coalitions give same gradient → exponential computation reduces to single calculation → DeepSHAP is theoretically justified.

---

## The Bottom Line

The confusion comes from mixing probabilistic independence (how we sample $r$) with functional independence (structure of $f$).

**The key for understanding DeepSHAP**: Focus on functional independence. It makes the gradient independent of coalition membership, which collapses the exponential Shapley average to a single term that DeepLIFT can compute in one backward pass.

---

## What the Original SHAP Paper Actually Shows

Now, regarding your question about approximation quality - yes, the original Lundberg & Lee (2017) paper does show some empirical validation:

### Computational Experiments (Section 5.1)

They compared Kernel SHAP vs LIME vs Shapley sampling on decision trees, measuring:
- **Sample efficiency**: How many model evaluations needed to get accurate estimates
- **Convergence**: How estimates approach true Shapley values with more samples

**Result**: Kernel SHAP converges faster than Shapley sampling and gives more accurate estimates than LIME.

### User Studies (Section 5.2)

They compared LIME, DeepLIFT, and SHAP with human explanations on simple models:
- Simple symptom-based model
- Max pooling allocation problem

**Result**: SHAP values showed "much stronger agreement" with human explanations than other methods.

### MNIST Example (Section 5.3)

They tested on digit classification (distinguishing 8 from 3):
- Compared original DeepLIFT, modified DeepLIFT (closer to SHAP), LIME, and Kernel SHAP
- Masked pixels according to attributions and measured class change

**Result**: Methods closer to SHAP values performed better at identifying which pixels matter.

### What They DON'T Show

**Critically**, the paper does NOT:
- Rigorously quantify approximation error for DeepSHAP vs true Shapley values on neural networks
- Test how well the independence assumption holds in practice
- Compare DeepSHAP accuracy on models with strong feature interactions

The experiments show that SHAP is better than alternatives, but don't validate how close DeepSHAP's approximation is to true Shapley values when functional independence is violated.

---

## Critical Later Work on SHAP's Limitations

Since the original 2017 paper, several important critiques have emerged:

### 1. Huang & Marques-Silva (2023): "The Inadequacy of Shapley Values for Explainability"

**Key finding**: Systematic analysis showing exact Shapley values can be fundamentally misleading for explainability.

**What they showed**:
- Analyzed all Boolean functions with 4 variables
- Found that 99.67% of functions have at least one instance where an **irrelevant feature** gets non-zero Shapley value
- Proved that relevant features can get lower Shapley values than irrelevant ones

**Their conclusion**: "SHAP scores will necessarily yield misleading information about the relative importance of features." The problem isn't just approximation error—even exact Shapley values can mislead.

**Citation**: Huang, X., & Marques-Silva, J. (2023). "On the failings of Shapley values for explainability." arXiv:2302.08160

### 2. Feature Dependence Issues

**The problem**: KernelSHAP and other approximation methods assume feature independence when computing conditional expectations:

$E[f(z) | z_S] \approx E_{z_{\bar{S}}}[f(z)]$

**Reality**: Features are almost always correlated in real data.

**Consequence**: SHAP can attribute importance to features that aren't actually used in realistic scenarios.

**Practical impact**: With highly correlated features (multicollinearity), SHAP assigns high values to one feature and near-zero to correlated features, even though they carry similar information.

**Sources**: 
- Aas et al. (2021): "Explaining individual predictions when features are dependent" - proposed conditional SHAP to address this
- Multiple practitioner reports in drug development, finance, etc.

### 3. Interaction Effects Not Captured

**The additive assumption**: $f(x) - f(r) = \sum_i \phi_i (x_i - r_i)$

**Problem**: This can't capture feature interactions accurately. If features $x_i$ and $x_j$ strongly interact, the contribution of $x_i$ depends on $x_j$'s value, but SHAP averages over all possible $x_j$ values.

**Example from text**: The phrase "not good" has meaning that's not the sum of "not" + "good" individually.

**Impact**: SHAP values can be misleading for models with strong non-linear interactions.

### 4. Computational Approximation Quality Unknown

**The gap**: While methods like KernelSHAP approximate Shapley values, there's often no way to know how good the approximation is for a specific instance.

**Recent work**: Alkhatib et al. (2024) proposed using conformal prediction to measure approximation quality, showing that approximation error can be substantial and varies widely across instances.

**Citation**: "Estimating Quality of Approximated Shapley Values Using Conformal Prediction," COPA 2024.

### 5. TreeSHAP vs KernelSHAP Discrepancies

Different SHAP implementations give different values for the same model because they make different assumptions:
- **TreeSHAP**: Accounts for tree structure, but makes specific assumptions about feature dependence
- **KernelSHAP**: Model-agnostic but assumes independence
- **DeepSHAP**: Fast but only approximate for non-linear models

**Result**: The same prediction can have different explanations depending on which SHAP method you use.

### 6. Causal vs Correlational Confusion

**SHAP measures**: Correlation-based feature importance (how features correlate with predictions)

**Users often want**: Causal importance (what happens if we intervene on a feature)

**The gap**: SHAP doesn't distinguish between:
- Direct causal effects
- Spurious correlations
- Confounding variables

**Example**: A model might rely on a spurious correlation; SHAP will give that feature high importance even though it's not causally relevant.

---

## The Honest Assessment

**What later work shows**:
1. Even **exact** Shapley values can be misleading (not just approximations)
2. The independence assumption is violated in almost all real applications
3. Approximation quality varies widely and is often unknown
4. Different SHAP implementations give different answers
5. SHAP measures correlation, not causation (despite user expectations)

**Why SHAP is still popular**:
- Fast tree-based implementations (TreeSHAP)
- Well-documented Python package
- Produces seemingly interpretable outputs
- Better than previous alternatives for many use cases
- Satisfies useful theoretical properties (when assumptions hold)

**The reality**: SHAP is a useful heuristic tool but should not be treated as ground truth for feature importance. Its theoretical guarantees only hold under strong assumptions that are rarely satisfied in practice.

Does this answer your question about the critical later work?