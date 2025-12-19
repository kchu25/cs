@def title = "Making the SHAP Approximation Intuitive"
@def published = "19 December 2025"
@def tags = ["interpretation"]

# Making the SHAP Approximation Intuitive

Hey! Great question - this is exactly where most people get confused about DeepSHAP. Let me break this down conversationally.

## The Core Confusion

You're right to be puzzled! On one hand, we have:
- **Shapley values**: average over exponentially many marginal contributions (all coalitions $S$)
- **DeepLIFT/DeepSHAP**: one forward-backward pass giving us coefficients $\phi_i(x,r)$

How can one backprop approximate an exponential average? The answer is subtle and relies entirely on that independence assumption.

## What's Actually Happening: The Key Insight

Here's the crucial realization: **under independence, all those different marginal contributions collapse to the same thing**.

Let me explain what I mean.

### The Shapley Formula (What We Want)

The true Shapley value for feature $i$ is:

$\phi_i^{\text{Shapley}} = \mathbb{E}_{S \sim \pi} \mathbb{E}_{r \sim \mathcal{R}}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})]$

This has **two** expectations:
1. **Outer expectation over coalitions $S$**: average over all possible subsets
2. **Inner expectation over references $r$**: for each coalition, average over reference samples

For each coalition $S$ and reference $r$:
- Keep features in $S$ at their input values $x_S$
- Set other features to reference values $r_{\bar{S}}$
- Measure what happens when we add feature $i$

**Example with 3 features:** For feature 1, we'd compute:
- $S = \emptyset$: for each reference $r$, effect of adding $x_1$ to $(r_1, r_2, r_3)$ → $(x_1, r_2, r_3)$
- $S = \{2\}$: for each reference $r$, effect of adding $x_1$ to $(r_1, x_2, r_3)$ → $(x_1, x_2, r_3)$
- $S = \{3\}$: for each reference $r$, effect of adding $x_1$ to $(r_1, r_2, x_3)$ → $(x_1, r_2, x_3)$
- $S = \{2,3\}$: for each reference $r$, effect of adding $x_1$ to $(r_1, x_2, x_3)$ → $(x_1, x_2, x_3)$

For each coalition, we average over multiple reference samples, then average across coalitions.

### What DeepLIFT Computes (What We Get)

DeepLIFT does one backward pass along the path from $r$ to $x$ and gives us:

$$f(x) - f(r) = \sum_{i=1}^n \phi_i(x,r) \cdot (x_i - r_i)$$

This decomposes the **total change** $f(x) - f(r)$ into feature contributions. The coefficient $\phi_i(x,r)$ for feature $i$ comes from integrating the gradient along the straight-line path:

$$\phi_i(x,r) = \int_0^1 \frac{\partial f}{\partial x_i}(r + t(x-r)) \, dt$$

This is computing: "as we move from $r$ to $x$ (varying ALL features simultaneously), how much does the gradient w.r.t. feature $i$ contribute?"

## The Magic of Independence

Here's where it all comes together. The conditional independence assumption actually makes **both** expectations simplify dramatically.

### What Conditional Independence Really Means

When we say features are conditionally independent given reference distribution $\mathcal{R}$, we mean:

$p(r_i, r_j) = p(r_i) \cdot p(r_j) \text{ for all } i, j$

The reference features are sampled independently. This seems innocent, but it has profound consequences!

### Step 1: Within a Single Reference

Let's use the Fundamental Theorem of Calculus for any coalition $S$ and reference $r$.

When we add feature $i$ to coalition $S$, we can write:

$$f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}}) = \int_0^1 \frac{\partial f}{\partial x_i}(x_S, r_i + t(x_i - r_i), r_{\overline{S \cup \{i\}}}) \, dt \cdot (x_i - r_i)$$

This says: the marginal effect is the integral of the gradient as we vary $x_i$ from $r_i$ to $x_i$, **keeping the coalition $S$ fixed**.

### Step 2: The Independence Assumption Makes Coalitions Irrelevant

**The gradient $\frac{\partial f}{\partial x_i}$ doesn't depend on which coalition you're in.**

Mathematically:

$\frac{\partial f}{\partial x_i}(x_S, z_i, r_{\overline{S \cup \{i\}}}) \approx \frac{\partial f}{\partial x_i}(r_1, \ldots, r_{i-1}, z_i, r_{i+1}, \ldots, r_n)$

**In words:** The effect of changing feature $i$ is the same whether other features are at $x$ or $r$!

### Step 3: Collapsing the Coalition Average (for Fixed $r$)

If the gradient doesn't depend on $S$, then for a **fixed reference** $r$:

$\mathbb{E}_{S \sim \pi}\left[\int_0^1 \frac{\partial f}{\partial x_i}(x_S, r_i + t(x_i - r_i), r_{\overline{S \cup \{i\}}}) \, dt\right]$

becomes just:

$\int_0^1 \frac{\partial f}{\partial x_i}(r_1, \ldots, r_{i-1}, r_i + t(x_i - r_i), r_{i+1}, \ldots, r_n) \, dt$

We can pull the integral out of the coalition expectation because it no longer depends on $S$!

### Step 4: Connecting to DeepLIFT (Still for Fixed $r$)

But wait - this is **still not quite** what DeepLIFT computes. For a given reference $r$, DeepLIFT integrates along the full path from $r$ to $x$:

$\phi_i(x,r) = \int_0^1 \frac{\partial f}{\partial x_i}(r + t(x-r)) \, dt$

The difference is:
- **Coalition integral (single-feature path)**: only varies feature $i$, others stay at $r$
- **DeepLIFT integral (multi-feature path)**: varies ALL features from $r$ to $x$

### Step 5: Why These Are Approximately Equal

Under independence, the gradient $\frac{\partial f}{\partial x_i}$ doesn't depend on the values of OTHER features (no interaction terms). So:

$\frac{\partial f}{\partial x_i}(r_1, \ldots, r_{i-1}, z_i, r_{i+1}, \ldots, r_n) \approx \frac{\partial f}{\partial x_i}(r + t(x-r))$

Even though the left side has other features at $r$ and the right side has them varying along the path, **it doesn't matter** because there are no interaction terms!

So for a **fixed reference** $r$:

$\mathbb{E}_{S \sim \pi}[\text{marginal contribution}] \approx \phi_i(x,r) \cdot (x_i - r_i)$

### Step 6: Now Average Over References

The full Shapley formula has:

$\phi_i^{\text{Shapley}} = \mathbb{E}_{r \sim \mathcal{R}} \left[ \mathbb{E}_{S \sim \pi}[\text{marginal contribution w.r.t. } r] \right]$

From Step 5, the inner expectation gives $\phi_i(x,r) \cdot (x_i - r_i)$, so:

$\phi_i^{\text{Shapley}} \approx \mathbb{E}_{r \sim \mathcal{R}} [\phi_i(x,r) \cdot (x_i - r_i)]$

**This is what DeepSHAP computes!** Sample multiple references $r^{(1)}, \ldots, r^{(K)}$, compute DeepLIFT for each, and average:

$\phi_i^{\text{DeepSHAP}} = \frac{1}{K} \sum_{k=1}^K \phi_i(x, r^{(k)}) \cdot (x_i - r_i^{(k)})$

## The Concrete Intuition

Think of it this way: 

**Without independence** (features interact):
- Coalition $S = \emptyset$: Adding $x_1$ to $(r_1, r_2, r_3)$ has marginal effect $A$
- Coalition $S = \{2\}$: Adding $x_1$ to $(r_1, x_2, r_3)$ has marginal effect $B$ ≠ $A$ (different because $x_2$ changes the gradient!)
- Coalition $S = \{3\}$: Adding $x_1$ to $(r_1, r_2, x_3)$ has marginal effect $C$ ≠ $A$
- Coalition $S = \{2,3\}$: Adding $x_1$ to $(r_1, x_2, x_3)$ has marginal effect $D$ ≠ $A$
- **Shapley value** = $\mathbb{E}_{S}[\text{marginal}]$ = properly weighted average of $A, B, C, D$ over all $2^{n-1}$ coalitions
- **You MUST compute all of them and average!**

**With independence** (no interactions):
- Coalition $S = \emptyset$: Adding $x_1$ to $(r_1, r_2, r_3)$ has marginal effect $A$
- Coalition $S = \{2\}$: Adding $x_1$ to $(r_1, x_2, r_3)$ has marginal effect $A$ (same! $x_2$ doesn't change the gradient)
- Coalition $S = \{3\}$: Adding $x_1$ to $(r_1, r_2, x_3)$ has marginal effect $A$ (still same!)
- Coalition $S = \{2,3\}$: Adding $x_1$ to $(r_1, x_2, x_3)$ has marginal effect $A$ (all equal!)
- **Shapley value** = $\mathbb{E}_{S}[\text{marginal}]$ = average of $[A, A, A, A, \ldots]$ = $A$
- **The coalition expectation is still there, it just equals any single term!**

And DeepLIFT's integral along the full path also gives us $A$ because the gradient doesn't depend on where other features are.

**So to directly answer your question:** Yes, exactly! Under independence, computing the coalition expectation reduces to a **single calculation** rather than an average over many terms. 

Mathematically, you're still computing $\mathbb{E}_S[\text{marginal}]$, but since every term in that expectation is identical, you only need to evaluate one of them:

$\mathbb{E}_S[\text{marginal}] = A \text{ (when all marginal contributions equal } A)$

It's the difference between:
- **Hard problem**: "Compute the average of 1000 different numbers" (needs all 1000 evaluations)
- **Easy problem**: "Compute the average of [7, 7, 7, ..., 7] (1000 times)" (just evaluate 7 once)

You're right to think of it as "reducing to a single calculation" rather than needing to actually compute and average many terms. The expectation operator is still there in the formula, but it degenerates to just picking out any representative term (they're all the same).

That's the computational magic: independence transforms the intractable Shapley coalition average into a single path integral that DeepLIFT can compute in one backward pass.

## Why "Approximation"?

So when do we have:

$$\mathbb{E}_{S}[\text{marginal contribution of } i \text{ in coalition } S] \approx \phi_i(x,r) \cdot (x_i - r_i)$$

**Exactly true when:**
1. Model is linear: $f(x) = \sum_i \beta_i x_i$
2. Features are independent: $\frac{\partial f}{\partial x_i}$ doesn't depend on $x_j$

**Approximately true when:**
1. Model is "locally linear" (smooth)
2. Features have weak interactions
3. Reference distribution is reasonable

**Fails when:**
1. Strong feature interactions (e.g., "not good" in text)
2. Highly nonlinear regions
3. Unrealistic reference distribution

## The Multiple Backprop Confusion Resolved

You asked: "When we do multiple backprop, we are only approximating one marginal contribution term. How is that approximating the Shapley value?"

Now we can answer precisely! Each backprop with reference $r^{(k)}$ gives:

$\phi_i(x, r^{(k)}) \cdot (x_i - r_i^{(k)})$

Under independence, this approximates:

$\mathbb{E}_{S \sim \pi}[\text{marginal contribution of } i \text{ in all coalitions w.r.t. reference } r^{(k)}]$

The **multiple references** give us different "samples" from the reference distribution $\mathcal{R}$. By averaging:

$\frac{1}{K} \sum_{k=1}^K \phi_i(x, r^{(k)}) \cdot (x_i - r_i^{(k)})$

we're approximating:

$\mathbb{E}_{r \sim \mathcal{R}} \left[ \mathbb{E}_{S \sim \pi}[\text{marginal contribution}] \right]$

which is exactly the Shapley value!

**Key insight:** Each single backprop with one reference already approximates the entire coalition average (inner expectation) under independence. The multiple backprops then approximate the outer expectation over references.

## The Bottom Line

The independence assumption is doing **massive** heavy lifting:

1. It makes all $2^{n-1}$ coalition-specific marginal contributions equal
2. It makes the single-path integral (DeepLIFT) equal to the many-coalition average (Shapley)
3. It allows one backprop to capture what would require exponential computation

Without independence, DeepSHAP is just a fast heuristic approximation. With independence, it's theoretically justified - but independence rarely holds in practice!

That's why the papers say "approximate SHAP values" and why it works "well enough" empirically despite violating the theoretical assumptions.

Does this make the connection clearer?