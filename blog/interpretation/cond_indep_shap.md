@def title = "The Conditional Independence Assumption in DeepSHAP"
@def published = "19 December 2025"
@def tags = ["interpretation"]

# The Conditional Independence Assumption in DeepSHAP

## A Critical Notational Pitfall (READ THIS FIRST!)

Before diving into the independence assumption, there's a common confusion in the notation that needs clearing up.

### The Misleading Expression

You might see:

$\frac{f(x) - f(r)}{x_i - r_i} \cdot (x_i - r_i)$

**STOP!** If you algebraically cancel the $(x_i - r_i)$ terms, you get:

$f(x) - f(r)$

This would mean feature $i$ alone explains the *entire* output difference! That's clearly wrong—all other features would have zero contribution.

### What DeepLIFT Actually Computes

DeepLIFT computes coefficients $\phi_i(x, r)$ via a **backward pass with multipliers**, NOT by simple division. The coefficients satisfy:

$f(x) - f(r) = \sum_{i=1}^{n} \phi_i(x, r) \cdot (x_i - r_i)$

**Key point:** Each $\phi_i(x, r)$ is computed through the chain rule:

$\phi_i(x, r) = \sum_{\text{paths from } i \text{ to output}} \prod_{\text{edges in path}} m_{\text{edge}}$

where $m_{\text{edge}}$ are the multipliers computed layer-by-layer.

### Why the Confusion?

The notation $\phi_i(x, r) \cdot (x_i - r_i)$ looks like it could be written as $\frac{f(x) - f(r)}{x_i - r_i} \cdot (x_i - r_i)$, but:

- **Wrong interpretation:** $\phi_i(x, r) = \frac{f(x) - f(r)}{x_i - r_i}$ (simple ratio - INCORRECT!)
- **Correct interpretation:** $\phi_i(x, r)$ is computed via multiplier backpropagation such that $\sum_i \phi_i(x, r) \cdot (x_i - r_i) = f(x) - f(r)$

**Analogy:** It's like saying "find numbers $a, b, c$ such that $a + b + c = 10$" vs "each of $a, b, c$ equals 10." The first distributes the total; the second makes no sense.

### The Correct Picture

For a simple network: $f(x) = \text{ReLU}(w_1 x_1 + w_2 x_2)$

If $x = (2, 3)$ and $r = (0, 0)$, and both neurons activate:

- DeepLIFT gives: $\phi_1(x,r) = w_1$ and $\phi_2(x,r) = w_2$
- Contributions: $C_1 = w_1 \cdot 2$ and $C_2 = w_2 \cdot 3$
- Sum: $w_1 \cdot 2 + w_2 \cdot 3 = f(2,3) - f(0,0)$ ✓

Notice: $\phi_1 \neq \frac{f(x) - f(r)}{x_1 - r_1} = \frac{w_1 \cdot 2 + w_2 \cdot 3}{2}$

The multipliers $w_1$ and $w_2$ are determined by the network structure, not by simple division!

---

## The Central Mystery

The document claims that under conditional independence:

$$\mathbb{E}_{S \sim \pi}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})] \approx \frac{f(x) - f(r)}{x_i - r_i} \cdot (x_i - r_i)$$

This is indeed a **big assumption** and the "≈" is doing *a lot* of work. Let's unpack why this might hold and what the original papers actually say.

## What Conditional Independence Really Means

### The Formal Statement

**The assumption:** Features $X_1, \ldots, X_n$ are conditionally independent given a reference sample $r \sim \mathcal{R}$.

**Mathematically:** For any two features $i, j$:

$$P(X_i, X_j | r) = P(X_i | r) \cdot P(X_j | r)$$

### Unpacking This Carefully

This is **NOT** saying features are globally independent! It's saying:

**Given that you've fixed a particular reference sample $r$**, the features have no residual correlations.

**Concrete example - Height and Weight:**

- **Global independence (FALSE):** Height and weight are clearly correlated in the population
- **Conditional independence (THE CLAIM):** Given a baseline person with height 5'8" and weight 160lbs, if you sample other people from the population, the variation in their heights doesn't tell you about variation in their weights

**Why this is weird:** In reality, even conditional on a reference, features remain correlated! If your reference is a 160lb person, other people's heights still correlate with their weights.

### What This Means for the Reference Distribution

When we write $r \sim \mathcal{R}$, we're sampling reference points. The conditional independence assumption says:

$$p(r_i, r_j) = p(r_i) \cdot p(r_j) \text{ for all } i, j$$

**In practice:**
- If $\mathcal{R}$ = "all training samples," this assumes training features are independent (almost never true!)
- If $\mathcal{R}$ = "a single baseline" (e.g., all zeros), you're not sampling at all—you've just fixed one reference

## Why This Assumption Matters: The Coalition Distribution Problem

### The Core Issue

When computing Shapley values, we need to evaluate:

$$f_x(S) = \mathbb{E}_{r \sim \mathcal{R}}[f(x_S, r_{\bar{S}})]$$

**What this expectation means:**
- Keep features in coalition $S$ from input $x$
- Replace features NOT in $S$ with values from sampled reference $r$
- Average over many reference samples

**The problem:** The distribution of $r_{\bar{S}}$ (absent features) might depend on $S$ (which features are present)!

### Example: The Correlation Problem

Imagine predicting house prices with features: [Square Footage, Number of Bedrooms].

- **Input:** $x = (2000 \text{ sqft}, 4 \text{ beds})$
- **Coalition $S = \{1\}$:** Keep square footage from $x$, sample bedrooms from references

**Under independence:** When you sample references, you might get:
- $r^{(1)} = (1200, 2)$ → evaluate $f(2000, 2)$
- $r^{(2)} = (3500, 6)$ → evaluate $f(2000, 6)$  
- $r^{(3)} = (800, 1)$ → evaluate $f(2000, 1)$

**The problem:** A 2000 sqft house with 1 bedroom is unrealistic! But the independence assumption treats all combinations as equally valid.

**Under dependence (reality):** References should respect the correlation:
- Larger houses tend to have more bedrooms
- When we condition on 2000 sqft, we should sample bedrooms from {2, 3, 4}, not {1, 6}

### Why This Matters for the Shapley Formula

The Shapley value is:

$$\phi_i = \mathbb{E}_{S \sim \pi} \mathbb{E}_{r \sim \mathcal{R}}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})]$$

**Under independence:** The inner expectation $\mathbb{E}_{r}[...]$ gives the same result regardless of which coalition $S$ you're considering, because $r_{\bar{S}}$ is always drawn from the same marginal distribution.

**Under dependence:** Different coalitions $S$ would require different conditional distributions for $r_{\bar{S}}$, because the present features in $S$ should inform what values are realistic for the absent features.

**The consequence:** When features are dependent, the true Shapley value (with proper conditional distributions) ≠ the naive Shapley value (with independence assumption).

## The Mathematical Argument for Why They're "Equal"

### Step 1: The True Shapley Value with References

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} \mathbb{E}_r[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})]$$

This is a weighted sum over $2^{n-1}$ coalitions.

### Step 2: The DeepLIFT Linearization

For a single $(x, r)$ pair, DeepLIFT computes:

$f(x) - f(r) = \sum_{i=1}^n \phi_i(x, r) \cdot (x_i - r_i)$

This is **exact** for this pair—it's a linear decomposition of the difference.

**CRITICAL NOTATION CLARIFICATION:** The coefficient $\phi_i(x, r)$ is NOT the simple ratio $\frac{f(x) - f(r)}{x_i - r_i}$! If it were, we'd have:

$\frac{f(x) - f(r)}{x_i - r_i} \cdot (x_i - r_i) = f(x) - f(r)$

This would mean feature $i$ alone explains the entire output difference, which is nonsense!

Instead, $\phi_i(x, r)$ is computed via the **multiplier backward pass** through the network, satisfying:

$\sum_{i=1}^n \phi_i(x, r) \cdot (x_i - r_i) = f(x) - f(r)$

Each $\phi_i(x, r)$ gets a portion of the total difference, determined by the chain rule through network layers.

### Step 3: The Key Claim (Independence-Based)

**Under conditional independence**, for any fixed $r$:

$$\mathbb{E}_{S \sim \pi}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})] \approx \phi_i(x, r) \cdot (x_i - r_i)$$

**Why would this be true?** Let me break down the left-hand side:

**Left side expanded:**
- For coalition $S = \emptyset$: marginal effect is $f(x_{\{i\}}, r_{\bar{\{i\}}}) - f(r)$
- For coalition $S = \{j\}$: marginal effect is $f(x_{\{i,j\}}, r_{\overline{\{i,j\}}}) - f(x_{\{j\}}, r_{\bar{\{j\}}})$
- ... and so on for all coalitions
- Take weighted average

**Right side:** DeepLIFT finds a single coefficient $\phi_i(x,r)$ such that when multiplied by $(x_i - r_i)$, it explains the total change $f(x) - f(r)$.

### Step 4: The Detailed Derivation - Why Independence Gives the Approximation

Let me derive step-by-step why conditional independence leads to:

$\mathbb{E}_{S \sim \pi}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})] \approx \phi_i(x,r) \cdot (x_i - r_i)$

#### Part 1: Expanding the Left Side

The Shapley formula averages over all coalitions. Let's write this out explicitly for feature $i$:

$\mathbb{E}_{S \sim \pi}[\Delta_i(S)] = \sum_{S \subseteq F \setminus \{i\}} w(S) \cdot [f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})]$

where $w(S) = \frac{|S|!(n-|S|-1)!}{n!}$ are Shapley weights.

**What this means:** For each possible coalition $S$ (not containing $i$), measure how much feature $i$ contributes when added to $S$.

#### Part 2: Using the Fundamental Theorem of Calculus

For any coalition $S$, we can write:

$f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}}) = \int_0^1 \frac{\partial f}{\partial z_i}(x_S, z_i, r_{\overline{S \cup \{i\}}}) \bigg|_{z_i = r_i + t(x_i - r_i)} dt \cdot (x_i - r_i)$

**What this says:** The marginal contribution of feature $i$ in coalition $S$ can be written as an integral of the partial derivative as we interpolate $z_i$ from $r_i$ to $x_i$.

**In simpler notation:**

$\Delta_i(S) = (x_i - r_i) \int_0^1 \frac{\partial f}{\partial x_i}(x_S, r_i + t(x_i - r_i), r_{\overline{S \cup \{i\}}}) dt$

#### Part 3: The Key Independence Assumption

**Under conditional independence:** The partial derivative $\frac{\partial f}{\partial x_i}$ does NOT depend on which other features come from $x$ vs $r$ (which coalition $S$ we're in).

**Formally:** For conditionally independent features:

$\frac{\partial f}{\partial x_i}(x_S, z_i, r_{\overline{S \cup \{i\}}}) \approx \frac{\partial f}{\partial x_i}(r_1, \ldots, r_{i-1}, z_i, r_{i+1}, \ldots, r_n)$

**Why?** Because the gradient doesn't depend on the values of other features if there are no interaction terms. 

**In other words:** The effect of changing $x_i$ is the same regardless of whether other features are at their $x$ values or $r$ values.

#### Part 4: Simplifying the Coalition Average

Now substitute this into the coalition average:

$\mathbb{E}_{S \sim \pi}[\Delta_i(S)] = \mathbb{E}_{S \sim \pi}\left[(x_i - r_i) \int_0^1 \frac{\partial f}{\partial x_i}(x_S, r_i + t(x_i - r_i), r_{\overline{S \cup \{i\}}}) dt\right]$

**Under independence**, the gradient doesn't depend on $S$, so we can pull it out:

$= (x_i - r_i) \int_0^1 \frac{\partial f}{\partial x_i}(r_1, \ldots, r_{i-1}, r_i + t(x_i - r_i), r_{i+1}, \ldots, r_n) dt$

**This is just the derivative integrated from $r$ to $x$ along feature $i$ only!**

#### Part 5: Connecting to the Total Change

By the fundamental theorem of calculus, this integral is:

$\int_0^1 \frac{\partial f}{\partial x_i}(r + t(x_i - r_i)\mathbf{e}_i) dt = \frac{f(r + (x_i - r_i)\mathbf{e}_i) - f(r)}{x_i - r_i}$

where $\mathbf{e}_i$ is the unit vector in direction $i$.

**But wait!** We want to relate this to the FULL change $f(x) - f(r)$, not just the change along feature $i$.

#### Part 6: The Multi-Dimensional Taylor Expansion

Under independence, we can write:

$f(x) - f(r) = \sum_{j=1}^n \int_0^1 \frac{\partial f}{\partial x_j}(r + t(x-r)) dt \cdot (x_j - r_j)$

This decomposes the total change into contributions from each feature.

**The key insight:** Under independence (no interaction terms), each integral gives:

$\phi_j(x,r) = \int_0^1 \frac{\partial f}{\partial x_j}(r + t(x-r)) dt$

And these satisfy:

$f(x) - f(r) = \sum_{j=1}^n \phi_j(x,r) \cdot (x_j - r_j)$

#### Part 7: The Final Connection

Going back to our coalition average:

$\mathbb{E}_{S \sim \pi}[\Delta_i(S)] = (x_i - r_i) \int_0^1 \frac{\partial f}{\partial x_i}(r + t(x_i - r_i)\mathbf{e}_i) dt$

**Under independence**, this approximately equals:

$\approx (x_i - r_i) \int_0^1 \frac{\partial f}{\partial x_i}(r + t(x-r)) dt = \phi_i(x,r) \cdot (x_i - r_i)$

**Why the approximation?** In the first integral, we only vary $x_i$. In the second, we vary ALL features simultaneously. Under independence, these give similar results because the gradient w.r.t. $x_i$ doesn't depend on other features.

#### Part 8: What DeepLIFT Actually Computes

DeepLIFT's multiplier backward pass computes an approximation to:

$\phi_i(x,r) \approx \int_0^1 \frac{\partial f}{\partial x_i}(r + t(x-r)) dt$

For piecewise linear networks (ReLU), this becomes:

$\phi_i(x,r) = \sum_{\text{linear regions}} \frac{\partial f}{\partial x_i}\bigg|_{\text{region}} \cdot \text{(fraction of path in region)}$

The multipliers track which neurons activate and weight the gradients accordingly.

### Why This Is Still an Approximation

Even with the derivation above, there are gaps:

1. **Non-additive interactions:** If $\frac{\partial f}{\partial x_i}$ depends on $x_j$ (interaction terms), then $\frac{\partial f}{\partial x_i}(x_S, z_i, r_{\overline{S \cup \{i\}}})$ DOES depend on coalition $S$, and we can't pull it out of the expectation.

2. **Path dependence:** Integrating along "feature $i$ only" vs "all features together" gives different results when there are interaction terms.

3. **Discrete approximation:** DeepLIFT uses piecewise linear approximation, not true integration.

**Bottom line:** Under independence, the coalition-averaged marginal effect equals the path-integrated gradient, which DeepLIFT approximates. Without independence, these diverge.

### Step 5: The Formal Connection via Linearity

If $f$ were perfectly linear:

$$f(x) = \beta_0 + \sum_i \beta_i x_i$$

Then:
- DeepLIFT gives: $\phi_i(x,r) = \beta_i$
- Shapley gives: $\phi_i = \beta_i \cdot (x_i - r_i)$ averaged over references

They match exactly!

**For nonlinear $f$:** DeepLIFT approximates $f$ as locally linear around each $(x,r)$ pair. The multipliers capture the "effective gradients" through the network.

### Step 6: Averaging Over References

When we take $\mathbb{E}_{r \sim \mathcal{R}}[\phi_i(x,r) \cdot (x_i - r_i)]$:

**Under independence:** Different references $r^{(1)}, r^{(2)}, \ldots$ give different "baseline contexts," and averaging over them approximates averaging over coalitions because the coalition distribution doesn't depend on which features are present.

**Under dependence:** This breaks down because the distribution of absent features should depend on which features are present, but DeepLIFT treats them as independent samples.

## Why Neural Networks Make This Worse (and Better)

### The Nonlinearity Problem

Deep networks are highly nonlinear. The function $f(x_S, r_{\bar{S}})$ can be wildly different for different coalitions $S$, even with the same reference $r$.

**Example - Image Classification:**
- Coalition $S$ = "top-left pixel only" → probably outputs uniform distribution (no info)
- Coalition $S$ = "all pixels except one" → probably outputs correct class
- The marginal contribution of one pixel depends HEAVILY on context

DeepLIFT's linearization $f(x) - f(r) = \sum \phi_i (x_i - r_i)$ can't capture these extreme nonlinear interactions well.

### The Multiplier Magic (Why It Still Works)

DeepLIFT's multipliers track contribution flow through the network:

$$m_{\Delta s_j \Delta t} = \frac{\partial C_{\Delta s_j \Delta t}}{\partial \Delta s_j}$$

For ReLU networks:

$$m_{\Delta x_i \Delta y} = \prod_{\text{edges on path}} \mathbb{1}[\text{neuron activated}]$$

**Key insight:** The backward pass automatically accounts for which neurons activated, which implicitly samples "important" computational paths through the network.

**This is similar to coalition sampling:** Different neurons activating = different features being "present" in the computation.

### Why Multiple References Help

With one reference $r^{(1)}$:
- DeepLIFT gives one linearization
- Might be a poor approximation if $r^{(1)}$ is far from $x$

With many references $r^{(1)}, \ldots, r^{(K)}$:
- Each gives a different linearization
- Averaging smooths out local approximation errors
- Different references activate different parts of the network

**Analogy:** Each reference "samples" a different subset of feature interactions, and averaging approximates the full coalition average.

## The Reality Check: When Does This Fail?

### Strong Feature Dependencies

**Example 1 - Natural Language:**
- Input: "This movie is not good"
- Features: ["not", "good"]
- Independence assumption: effect("not") + effect("good") = negative + positive = ???
- Reality: "not good" has a combined meaning that's not additive

**Example 2 - Pixels in Images:**
- A single red pixel means nothing
- A cluster of red pixels forming a stop sign is meaningful
- The features (pixels) have strong spatial correlations

**What happens:** DeepSHAP treats these as independent, computing:

$$\phi_{\text{not}}(x, r) + \phi_{\text{good}}(x, r)$$

But the true Shapley value should account for their interaction:

$$\phi_{\text{not}} + \phi_{\text{good}} + \phi_{\text{not,good}}^{\text{interaction}}$$

### Unrealistic Reference Distribution

If your reference distribution $\mathcal{R}$ is poorly chosen:

**Example - All zeros:** $r = (0, 0, \ldots, 0)$
- A 0 sqft house with 0 bedrooms is nonsensical
- The baseline $f(r)$ might give garbage output
- The linearization $f(x) - f(r)$ is approximating a meaningless difference

**Better choice:** Sample references from training data
- More realistic baselines
- But still assumes features are independent within training distribution (often false)

### Interventional vs Observational

The independence assumption makes DeepSHAP compute **observational** Shapley values:
- "What if I observe feature $i = x_i$ and other features follow their natural distribution?"

But we often want **interventional** Shapley values:
- "What if I forcibly set feature $i = x_i$, holding the causal mechanism fixed?"

These differ when features are causally related!

**Example - Medical data:**
- Observational: "Patients with high blood pressure tend to take medication"
- Interventional: "If we give medication, blood pressure drops"

The independence assumption conflates these!

## What the Papers Actually Say

### DeepLIFT Paper (Shrikumar et al., 2017)

**Direct quotes:**
- "We propose a method for decomposing the output prediction of a neural network on a specific input by backpropagating the contributions of all neurons"
- Does NOT claim to compute Shapley values
- Shows empirically that it satisfies "summation-to-delta" property

**No mention of:** Conditional independence, Shapley values, or coalitions.

### SHAP Paper (Lundberg & Lee, 2017)

**Section 3.3 - DeepSHAP:**

"DeepLIFT with a reference distribution computes approximate SHAP values... The approximation is exact when:
1. The model is linear
2. The input features are independent"

**Section 5.2 - Computational Efficiency:**

"DeepSHAP is orders of magnitude faster than Kernel SHAP while providing similar accuracy"

**They explicitly acknowledge:** It's an approximation, not exact for nonlinear models with dependent features.

### Follow-up Work (Sundararajan et al., 2017 - Integrated Gradients)

Shows that path methods (like DeepLIFT) satisfy completeness (summation-to-delta) but don't necessarily give true Shapley values unless you integrate over all paths, which is equivalent to sampling all coalitions.

## The Bottom Line: It's a Practical Compromise

### What We're Really Doing

The equation:

$$\mathbb{E}_{S \sim \pi}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})] \approx \phi_i(x,r) \cdot (x_i - r_i)$$

is **not mathematically rigorous** for real-world scenarios. It's justified by:

1. **Theoretical result:** Exact under independence + linearity (rarely satisfied)
2. **Empirical observation:** Works "well enough" in practice (vague but true)
3. **Computational necessity:** True Shapley requires $O(2^n)$ evaluations (intractable)
4. **Intuition:** Multipliers approximate marginal effects through the network

### Why We Accept It Anyway

- **Speed:** DeepSHAP is 1000x faster than kernel SHAP
- **Smoothness:** Neural networks are locally smooth, so linearization isn't terrible
- **Averaging:** Multiple references average out some errors
- **Interpretability:** Even approximate attributions are useful for debugging models

### When to Worry

Use exact methods (kernel SHAP, sampling-based Shapley) when:
- Features have strong interactions (e.g., language, structured data)
- You need provably correct attributions (legal, medical applications)
- The model is extremely nonlinear
- The reference distribution is clearly wrong

Accept DeepSHAP's approximation when:
- You need speed (billions of predictions to explain)
- Features are weakly correlated
- Model is reasonably smooth
- You're okay with "directionally correct" explanations

## The Honest Technical Answer

The conditional independence assumption $P(X_i, X_j | r) = P(X_i | r) \cdot P(X_j | r)$ does three things:

1. **Makes coalition averaging = path averaging:** Under independence, averaging marginal effects over coalitions gives the same result as averaging along interpolation paths
2. **Justifies single-reference linearization:** Each DeepLIFT linearization approximates the coalition-averaged effect for that reference
3. **Enables fast computation:** Instead of $2^n$ coalition evaluations, we do $K$ forward-backward passes (typically $K \approx 10-100$)

**But in reality:**
- Features ARE correlated
- Models ARE nonlinear  
- References ARE imperfect samples
- Approximation quality varies wildly by domain

The papers are surprisingly honest about this—they prove exactness under idealized assumptions, then empirically show "good enough" performance in practice, while acknowledging limitations.

**DeepSHAP is best understood as:** A fast, heuristic approximation to Shapley values that works well when features are weakly interacting and the model is locally smooth. Not a theorem, but a useful tool.