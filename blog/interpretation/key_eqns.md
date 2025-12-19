@def title = "SHAP: Key Equations in Aas, Jullum, and Løland (2021)"
@def published = "19 December 2025"
@def tags = ["interpretation"]

### Key Equations in Aas, Jullum, and Løland (2021)

**Important correction:** The Aas et al. paper focuses on **Kernel SHAP**, not DeepSHAP specifically. However, the independence assumption they discuss applies to both methods.

The paper formalizes the distinction between conditional and marginal expectations. Look at these specific equations:

**Equation (8)** - The true Shapley value using conditional expectations:

$v(S) = \mathbb{E}[f(x_S, X_{\bar{S}}) | X_S = x_S]$

This integrates over the **conditional distribution** $p(x_{\bar{S}} | x_S)$, which respects feature dependencies.

**Section 2.3.2** - What Kernel SHAP does:

The paper states that Kernel SHAP "assumes feature independence, replacing $p(x_{\bar{S}} | x_S)$ by $p(x_{\bar{S}})$ in (8)."

This gives the **interventional** Shapley value:

$v(S) = \mathbb{E}_{X_{\bar{S}}}[f(x_S, X_{\bar{S}})]$

where the expectation is over the **marginal distribution** of absent features.

**Does this apply to DeepSHAP?**

Yes, but indirectly. Both Kernel SHAP and DeepSHAP make the same fundamental independence assumption:
- **Kernel SHAP**: Explicitly samples from the marginal distribution
- **DeepSHAP**: Averages DeepLIFT contributions over reference samples drawn from the marginal distribution

The Aas paper's critique applies to both: they're computing "interventional" Shapley values (assuming independence) rather than true conditional Shapley values.

However, Aas et al. don't provide the specific mathematical derivation of how DeepLIFT's multipliers relate to coalition sampling. Their focus is on fixing the independence problem for model-agnostic methods, not explaining the internal mechanics of DeepSHAP.

---

## CRITICAL SOURCING DISCLAIMER

**I need to be honest about my sources here:**

### What I CAN verify:
1. ✓ Lundberg's 2017 NIPS paper exists and is sparse on DeepSHAP details
2. ✓ Aas et al. (2021) discusses conditional vs marginal expectations for Kernel SHAP
3. ✓ The independence assumption is a known issue in the SHAP literature

### What I CANNOT directly verify from papers:
1. ✗ The exact "double expectation" formula $\mathbb{E}_S \mathbb{E}_r$ for DeepSHAP specifically
2. ✗ Which paper first presented this formulation
3. ✗ Whether the original document you provided is accurately representing published work

### The honest truth:
The "double expectation" framework I've been explaining appears in the document you provided, but **I cannot trace it to a specific published paper**. It may be:
- A pedagogical explanation created by the document's author
- Standard folklore in the ML interpretation community
- Derived from multiple sources without explicit citation
- An interpretation that's not formally published

**What you should do:**
1. Read Lundberg's 2017 paper Section 4.2 yourself
2. Check the Chen et al. (2022) Nature Communications paper
3. Look for the original DeepLIFT paper (Shrikumar et al. 2017)
4. Don't trust my explanation without verifying against primary sources

I apologize for presenting this as if it were well-established when I cannot actually cite specific equations from specific papers that support the "double expectation" view of DeepSHAP.# Understanding DeepSHAP Notation and Reference Averaging

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

## The Independence Assumption: DeepSHAP's Critical "Cheat"

### Yes, DeepSHAP is Absolutely Cheating

You're exactly right! The independence assumption is where DeepSHAP makes a **huge** simplification that fundamentally changes what's being computed.

### What True Shapley Values Require

The **correct** way to define $f_x(S)$ for coalition $S$ is:

$$f_x(S) = \mathbb{E}_{x_{\bar{S}} \sim P(x_{\bar{S}} | x_S)}[f(x_S, x_{\bar{S}})]$$

This means: "When features in $S$ are fixed to their values in $x$, what's the expected output over the **conditional distribution** of the remaining features?"

This respects feature dependencies! If you know that temperature = 90°F, you should sample humidity from $P(\text{humidity} | \text{temp}=90)$, not from the marginal distribution $P(\text{humidity})$.

### What DeepSHAP Actually Does

DeepSHAP instead uses:

$$f_x(S) \approx \mathbb{E}_{r \sim \mathcal{R}}[f(x_S, r_{\bar{S}})]$$

where $\mathcal{R}$ is just the marginal distribution (or a sample from training data).

**The problem:** This treats absent features as **independent** from present features. You're replacing missing features with values from $r$ that have no relationship to the features you've fixed!

### Concrete Example of Where This Breaks

**Scenario:** Predicting house price with features:
- $x_1$ = square footage = 3000 sq ft
- $x_2$ = number of bedrooms = 2

These are **highly correlated**: large houses usually have more bedrooms.

**What should happen** (true Shapley):
- For coalition $S = \{1\}$ (only square footage present): 
  - Sample bedrooms from $P(\text{bedrooms} | \text{sqft}=3000)$
  - This will give mostly 3-5 bedrooms (appropriate for 3000 sq ft)
  
**What DeepSHAP does:**
- For coalition $S = \{1\}$: 
  - Sample bedrooms from $P(\text{bedrooms})$ ignoring square footage
  - Might get 1 bedroom or 7 bedrooms paired with 3000 sq ft
  - Creates **impossible or highly unlikely combinations**!

### Why This Matters

When features are correlated, DeepSHAP:

1. **Evaluates the model on unrealistic inputs** like (3000 sq ft, 1 bedroom)
2. **Attributes importance incorrectly** because it's measuring contributions in regions of feature space that never occur in reality
3. **Violates the spirit of Shapley values**, which should measure marginal contributions given realistic scenarios

### The Mathematical Sleight of Hand

The approximation in Step 4 of the original document:

$$\mathbb{E}_{S \sim \pi}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})] \approx C_{\Delta x_i \Delta f}(x, r)$$

This is only valid when features are **conditionally independent**. For correlated features, this approximation can be arbitrarily bad!

### Why Do People Still Use It?

1. **Computational tractability:** Computing true conditional distributions is extremely hard
2. **"Good enough" in practice:** For weakly correlated features, the error might be acceptable
3. **No better alternatives:** Other fast approximations have their own problems
4. **Lack of awareness:** Many users don't realize this assumption is being made

### The Honest Truth

**DeepSHAP does NOT compute true Shapley values.**

What it computes are "Shapley values" for a **different problem**—one where:
- Features are assumed independent
- The model is evaluated on unrealistic feature combinations
- The definition of "marginal contribution" is fundamentally altered

These are not approximations of true Shapley values. They are **exact solutions to the wrong problem.**

When features are correlated, calling these "Shapley values" is misleading. You're getting precise answers to a question you didn't actually ask.

### Better Alternatives?

- **Conditional sampling methods:** Sample from $P(x_{\bar{S}} | x_S)$ using generative models (expensive!)
- **Interventional SHAP:** Use causal models to define interventions (requires domain knowledge)
- **Accept the limitations:** Use DeepSHAP but interpret results cautiously when you know features are correlated

---

## Summary

1. **The subscript $x$** reminds us we're explaining a specific input $x$—different inputs have different Shapley values.

2. **Reference averaging** works by creating different "background contexts" that implicitly sample coalitions.

3. **BUT—and this is crucial—** DeepSHAP fundamentally assumes features are independent, which is almost never true in practice. It's a computational hack that trades correctness for speed. You're right to call it "cheating"—the math only works under assumptions that are routinely violated in real applications.