@def title = "From DeepLIFT Contributions to Shapley Values: The Quantitative Connection"
@def published = "19 December 2025"
@def tags = ["interpretation"]

# From DeepLIFT Contributions to Shapley Values: The Quantitative Connection

## The Core Question

How does averaging DeepLIFT contribution scores over multiple reference inputs produce SHAP (Shapley) values? Let's make this connection mathematically explicit.

## Step 1: What DeepLIFT Gives You

For a single input $x$ and reference $r$, DeepLIFT computes contribution scores:

$$C_{\Delta x_i \Delta f}(x, r) = \phi_i(x, r) \cdot (x_i - r_i)$$

These satisfy the **summation-to-delta** property:

$$f(x) - f(r) = \sum_{i=1}^{n} C_{\Delta x_i \Delta f}(x, r) = \sum_{i=1}^{n} \phi_i(x, r) \cdot (x_i - r_i)$$

The coefficients $\phi_i(x, r)$ depend on both the input and reference, and are computed via the multiplier backward pass through the network.

## Step 2: The Shapley Value Definition

The true Shapley value for feature $i$ is:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f_x(S \cup \{i\}) - f_x(S)]$$

where:
- $F$ is the set of all features
- $S$ is a coalition (subset) not containing $i$
- $f_x(S)$ is the model's expected output when features in $S$ take values from $x$, and features not in $S$ are "missing" (integrated out)

This can be rewritten as an expectation:

$$\phi_i = \mathbb{E}_{S \sim \pi}[f_x(S \cup \{i\}) - f_x(S)]$$

where $\pi$ is the uniform distribution over all possible feature coalitions.

## Step 3: The Reference Distribution Approximation

**Key insight:** Define the function $f_x(S)$ using a reference distribution $\mathcal{R}$:

$f_x(S) = \mathbb{E}_{r \sim \mathcal{R}}[f(x_S, r_{\bar{S}})]$

**What this notation means:**
- $S$ is a **coalition** = a subset of features (e.g., $S = \{1, 3, 5\}$ means features 1, 3, and 5)
- $\bar{S}$ is the **complement** = all features NOT in $S$
- $x_S$ = use values from input $x$ for features in coalition $S$
- $r_{\bar{S}}$ = use values from reference $r$ for features NOT in $S$

**Concrete example:** If you have 4 features and $S = \{2, 4\}$:
- $f(x_S, r_{\bar{S}}) = f(r_1, x_2, r_3, x_4)$
- Features 2 and 4 come from $x$ (they're "present" in the coalition)
- Features 1 and 3 come from $r$ (they're "absent" from the coalition)

This hybrid input lets us measure: *"What's the model output when only features in $S$ are active?"*

With this definition, the Shapley value becomes:

$\phi_i = \mathbb{E}_{S \sim \pi}\mathbb{E}_{r \sim \mathcal{R}}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})]$

**What this measures:** The marginal contribution of adding feature $i$ to coalition $S$:
- **First term:** $f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}})$ = output when coalition $S$ AND feature $i$ are present
- **Second term:** $f(x_S, r_{\bar{S}})$ = output when only coalition $S$ is present (feature $i$ is absent)
- **Difference:** How much does feature $i$ change the output, given that $S$ is already present?

We average this marginal contribution over ALL possible coalitions $S$ (weighted by the Shapley formula).

## Step 4: The Independence Assumption

**Critical assumption:** If features are conditionally independent given the reference distribution $\mathcal{R}$, then under certain regularity conditions on the model $f$:

$\mathbb{E}_{S \sim \pi}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})] \approx \frac{f(x) - f(r)}{x_i - r_i} \cdot (x_i - r_i) = C_{\Delta x_i \Delta f}(x, r)$

**What's happening here:**

- **Left side:** We're computing the marginal effect of feature $i$ by:
  1. Taking ALL possible coalitions $S$ 
  2. For each coalition, creating a hybrid input where $S$ comes from $x$ and non-$S$ comes from $r$
  3. Measuring how much feature $i$ contributes when added to that coalition
  4. Averaging over all coalitions

- **Right side:** DeepLIFT computes a single coefficient $\frac{f(x) - f(r)}{x_i - r_i}$ that linearizes the full change from $r$ to $x$

**The approximation:** Under the independence assumption, these two approaches give similar results! DeepLIFT's linearization implicitly captures the average marginal effect across coalitions, even though it only does one forward-backward pass instead of enumerating all $2^n$ coalitions.

## Step 5: The Averaging Step (DeepSHAP)

DeepSHAP takes the expectation over the reference distribution:

$$\phi_i^{\text{DeepSHAP}}(x) = \mathbb{E}_{r \sim \mathcal{R}}[C_{\Delta x_i \Delta f}(x, r)]$$

$$= \frac{1}{|R|} \sum_{r \in R} \phi_i(x, r) \cdot (x_i - r_i)$$

where $R$ is a finite sample from $\mathcal{R}$.

## Step 6: Why This Equals the Shapley Value

Combining steps 3-5:

$\phi_i^{\text{DeepSHAP}} = \mathbb{E}_{r \sim \mathcal{R}}[C_{\Delta x_i \Delta f}(x, r)]$

$\approx \mathbb{E}_{r \sim \mathcal{R}}\mathbb{E}_{S \sim \pi}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})]$

$= \mathbb{E}_{S \sim \pi}\mathbb{E}_{r \sim \mathcal{R}}[f(x_{S \cup \{i\}}, r_{\overline{S \cup \{i\}}}) - f(x_S, r_{\bar{S}})]$

$= \phi_i^{\text{Shapley}}$

**Reading this chain of equalities:**

1. **Line 1:** DeepSHAP = average DeepLIFT contributions over references
2. **Line 2:** Each DeepLIFT contribution approximates the coalition-averaged marginal effect (from Step 4)
3. **Line 3:** Swap the order of expectations (valid since they're independent)
4. **Line 4:** This is exactly the Shapley value definition!

The key step is recognizing that DeepLIFT's linearization (via multipliers) effectively performs the coalition sampling that the Shapley formula requires, but does it through the network structure rather than explicit enumeration of $2^n$ coalitions.

## The Mathematical Machinery

### How Multipliers Enable Coalition Sampling

The multipliers $m_{\Delta s_j \Delta t}$ computed by DeepLIFT satisfy:

$$C_{\Delta x_i \Delta t} = \sum_{j} C_{\Delta x_i \Delta s_j} \cdot m_{\Delta s_j \Delta t}$$

When you vary the reference $r$, you're implicitly varying which features are "present" (from $x$) vs "absent" (from $r$) at each layer. The multiplier chain rule automatically accounts for all possible paths through the network, which corresponds to marginalizing over different feature coalitions.

### The Linearization Trick

For any $(x, r)$ pair, DeepLIFT finds coefficients such that:

$$f(x) - f(r) = \sum_{i=1}^{n} \phi_i(x, r) \cdot (x_i - r_i)$$

This is exact for this pair, but is a *linear* relationship. When we average over many references:

$$\mathbb{E}_r[\phi_i(x, r) \cdot (x_i - r_i)] \approx \text{true marginal effect of } x_i$$

The averaging smooths out the local linearizations into a global approximation of the Shapley value.

## Practical Implementation

```
For each input x:
    Initialize: Φ_i = 0 for all features i
    
    For each reference sample r in R:
        # Forward passes
        activations_x = forward(x)
        activations_r = forward(r)
        
        # Compute differences
        Δs_j = activations_x[j] - activations_r[j]
        
        # Backward pass with multipliers
        contributions = backprop_multipliers(Δs, network)
        
        # Accumulate
        Φ_i += contributions[i]
    
    # Average
    Φ_i = Φ_i / |R|
    
    Return: Shapley values Φ
```

## Why Averaging Works: Intuition

- **Single reference:** DeepLIFT gives you $\phi_i(x, r)$ that explains the difference $f(x) - f(r)$ as a linear function of input differences.

- **Multiple references:** Different references $r_1, r_2, \ldots$ represent different "baseline contexts." Averaging over them gives you the expected marginal contribution of feature $i$ across all possible contexts.

- **Shapley connection:** The Shapley formula $\sum_S \text{weight}(S) \cdot [f(S \cup \{i\}) - f(S)]$ is also an average of marginal contributions across contexts (coalitions $S$).

- **The magic:** DeepLIFT's linearization + reference averaging implicitly performs the coalition sampling that Shapley requires, but does it through the network structure rather than explicit enumeration.

## A Critical Clarification: How References and Coalitions Interact

**The key distinction:**
- The **coalition $S$** determines WHICH features come from $x$ vs $r$
- The **reference $r$** is randomly sampled and used for ALL features not in $S$

**Example walkthrough** with 3 features:

For coalition $S = \{1\}$ (only feature 1 is present):
- Sample $r^{(1)} = (5, 7, 2)$, evaluate $f(x_1, r^{(1)}_2, r^{(1)}_3) = f(x_1, 7, 2)$
- Sample $r^{(2)} = (3, 9, 1)$, evaluate $f(x_1, r^{(2)}_2, r^{(2)}_3) = f(x_1, 9, 1)$
- Average: $f_x(S=\{1\}) = \mathbb{E}_r[f(x_1, r_2, r_3)]$

Notice: Feature 1 ALWAYS uses $x_1$, but features 2 and 3 vary with each random $r$ sample.

**Why this matters:** By averaging over $r$, we're integrating out (marginalizing) the variability of absent features, which gives us the expected output when only coalition $S$ is active. This is exactly what the Shapley formula requires!

## The Approximation Quality

The approximation $\phi_i^{\text{DeepSHAP}} \approx \phi_i^{\text{Shapley}}$ is exact when:

1. The model $f$ is linear
2. Features are independent given $\mathcal{R}$
3. Sufficient references are sampled

For nonlinear networks, it's an approximation that trades accuracy for computational efficiency:
- **True Shapley:** $2^n$ model evaluations
- **DeepSHAP:** $|R|$ forward-backward passes (typically $|R| \ll 2^n$)

The approximation improves as $|R|$ increases, but there are no formal convergence guarantees—the quality depends on the reference distribution, network smoothness, and degree of feature interactions.