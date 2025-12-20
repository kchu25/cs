@def title = "Interpretation Frameworks Using Additive Decomposition"
@def published = "20 December 2025"
@def tags = ["interpretation"]

# Interpretation Frameworks Using Additive Decomposition

## Overview

Additive decomposition is a fundamental principle in many model interpretation frameworks. The general form expresses a model's prediction as:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

where $\phi_0$ is the base value and $\phi_i$ represents the contribution of feature $i$.

---

## Frameworks Using Additive Decomposition

### 1. **SHAP (SHapley Additive exPlanations)**

**Type**: Unified framework based on game theory

**Additive Form**:
$$f(x) = E[f(X)] + \sum_{i=1}^{M} \phi_i(x)$$

where $\phi_i$ are Shapley values satisfying:
- **Efficiency**: $\sum_{i=1}^{M} \phi_i = f(x) - E[f(X)]$
- **Symmetry**: Equal features get equal attributions
- **Dummy**: Zero-effect features get zero attribution
- **Additivity**: Attributions compose for ensemble models

**Variants**:
- **KernelSHAP**: Model-agnostic approximation using weighted linear regression
- **TreeSHAP**: Exact computation for tree-based models
- **DeepSHAP**: Neural networks via DeepLIFT aggregation
- **GradientSHAP**: Expected gradients as SHAP approximation

---

### 2. **LIME (Local Interpretable Model-agnostic Explanations)**

**Type**: Local surrogate models

**Additive Form**:
$$g(z) = \beta_0 + \sum_{i=1}^{M} \beta_i z_i$$

where $g$ is an interpretable linear model that locally approximates $f$ around instance $x$.

**Process**:
1. Generate perturbed samples around $x$
2. Weight samples by proximity to $x$
3. Fit linear model: $\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$
4. Extract coefficients $\beta_i$ as feature importances

**Key Difference from SHAP**: LIME approximations don't satisfy Shapley axioms (no efficiency guarantee).

---

### 3. **DeepLIFT (Deep Learning Important FeaTures)**

**Type**: Gradient-based attribution for neural networks

**Additive Form**:
$$\Delta y = y - y_{\text{ref}} = \sum_{i=1}^{M} C_{\Delta x_i \Delta y}$$

where $C_{\Delta x_i \Delta y}$ is the contribution of feature $i$ to the output difference from reference $x_{\text{ref}}$.

**Properties**:
- Uses modified chain rule for backpropagation
- Handles activation function saturation better than gradients
- Single backward pass per reference point

**Connection**: DeepSHAP aggregates DeepLIFT scores over multiple references:
$$\phi_i \approx \frac{1}{K} \sum_{k=1}^{K} C_{\Delta x_i \Delta y}^{(k)}$$

---

### 4. **Integrated Gradients**

**Type**: Path-based gradient method

**Additive Form**:
$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

where $x'$ is a baseline input.

**Satisfies**:
- **Completeness** (efficiency): $\sum_{i=1}^{M} \text{IG}_i(x) = f(x) - f(x')$
- **Sensitivity**: Non-zero gradient implies non-zero attribution
- **Implementation invariance**: Functionally equivalent networks get same attributions

**Approximation**: Typically computed via Riemann sum over path from baseline to input.

---

### 5. **Layer-wise Relevance Propagation (LRP)**

**Type**: Backward decomposition for neural networks

**Additive Form** (at each layer):
$$R_i^{(l)} = \sum_{j} R_{i \leftarrow j}^{(l,l+1)}$$

where relevance at layer $l$ is redistributed from layer $l+1$ according to propagation rules.

**Conservation Property**:
$$\sum_{i} R_i^{(l)} = \sum_{j} R_j^{(l+1)} = f(x)$$

**Different Rules**:
- **$\epsilon$-rule**: $R_{i \leftarrow j} = \frac{z_{ij}}{\sum_i z_{ij} + \epsilon \cdot \text{sign}(\sum_i z_{ij})} R_j$
- **$\alpha\beta$-rule**: Separates positive/negative contributions
- **$z^+$-rule**: For first layer, focuses on positive evidence

---

### 6. **Attention Mechanisms** (with additive interpretation)

**Type**: Built-in model component

**Additive Form** (for additive attention):
$$\text{context} = \sum_{i=1}^{T} \alpha_i h_i$$

where $\alpha_i$ are attention weights and $h_i$ are hidden states.

**Properties**:
- Attention weights can serve as feature importance when $\sum_i \alpha_i = 1$
- Not always faithful to model behavior (post-hoc analysis shows attention ≠ explanation in many cases)
- More interpretable than raw hidden states

---

### 7. **Influence Functions** (linear approximation)

**Type**: Training data attribution

**Additive Approximation**:
$$f(x) - f_{-z}(x) \approx -\nabla_\theta f(x)^\top H^{-1} \nabla_\theta L(z, \theta)$$

where $H$ is the Hessian and the influence of training point $z$ can be decomposed additively across parameters.

**Use Case**: Understanding which training examples most influenced a prediction.

---

## Comparison Table

| Framework | Computation | Theoretical Guarantees | Model Types |
|-----------|-------------|------------------------|-------------|
| **SHAP** | Exponential (exact) / Polynomial (approx) | Efficiency, Symmetry, Dummy, Additivity | All |
| **LIME** | Polynomial | Local fidelity only | All |
| **DeepLIFT** | Linear (backprop) | Reference-dependent efficiency | Neural nets |
| **Integrated Gradients** | Linear × discretization steps | Completeness, Sensitivity | Differentiable |
| **LRP** | Linear (backprop) | Conservation | Neural nets |
| **Attention** | Built-in | None (descriptive) | Attention models |
| **Influence Functions** | Cubic (Hessian) | First-order approximation | Differentiable |

---

## Key Insight: Why Additive Decomposition Makes Interpretation Tractable

The additive decomposition framework provides **accountability**: every part of the prediction is explained, and these explanations sum to the total prediction difference from a baseline. This makes debugging, auditing, and understanding model decisions much more tractable.

### What "Tractable" Means in This Context

**Tractability** here refers to both computational and cognitive manageability. Let's break down what becomes tractable:

#### 1. **Complete Accounting** (Budget-Like Analysis)

The efficiency property ensures:
$$f(x) - f(x_{\text{baseline}}) = \sum_{i=1}^{M} \phi_i$$

This is analogous to a financial budget where:
- **Total change**: $f(x) - f(x_{\text{baseline}})$ (e.g., model predicts +\$50 higher risk score)
- **Line items**: Each $\phi_i$ explains how much feature $i$ contributed

**Why this matters**:
- **No missing mass**: You can't have unexplained contributions hiding in interactions that aren't accounted for
- **Verify completeness**: Sum your attributions to check they match the prediction difference
- **Identify dominant factors**: Sort $|\phi_i|$ to see which features drive the decision

**Example**: Credit scoring model predicts score of 720 (baseline: 650, difference: +70)
```
Income level:        +35 points
Credit history:      +28 points  
Debt-to-income:      -15 points
Employment length:   +18 points
Recent inquiries:     +4 points
                     -------
Total:               +70 points ✓
```

Without additivity, you might get explanations like "income and credit history are important" but not "together they account for 63 out of 70 points."

---

#### 2. **Debugging Model Failures**

When a model makes an incorrect prediction, additive decomposition lets you:

**Identify the culprit features**:
- Which features pushed the model toward the wrong answer?
- Are any features contributing in unexpected directions?

**Example**: Medical diagnosis model incorrectly predicts high diabetes risk

```
Glucose level:        +0.45  ← Expected positive
BMI:                  +0.32  ← Expected positive
Age:                  +0.15  ← Expected positive
Exercise frequency:   +0.28  ← Should be NEGATIVE! Bug found.
Family history:       +0.12  ← Expected positive
                      ------
Total:                +1.32 (high risk prediction)
```

The additive form makes it **tractable to scan** through features and spot that "exercise frequency" has the wrong sign, revealing a potential data preprocessing bug or feature engineering error.

**Without additivity**: You might get saliency maps or attention weights that are hard to interpret quantitatively, making systematic debugging difficult.

---

#### 3. **Auditing for Bias and Fairness**

Organizations need to ensure models don't discriminate based on protected attributes. Additive decomposition makes this **legally and practically tractable**:

**Quantitative bias detection**:
$$\text{Bias} = \mathbb{E}_{x \in \text{Group A}}[\phi_{\text{protected}}] - \mathbb{E}_{x \in \text{Group B}}[\phi_{\text{protected}}]$$

**Example**: Hiring model analysis
```
For candidates with similar qualifications:

Group A (protected class):
  Gender-related features: -0.15 (on average)
  
Group B (non-protected):
  Gender-related features: +0.02 (on average)
  
Bias measure: 0.17 points penalty
```

**Why tractable**:
- **Quantifiable**: You get numbers (-0.15 vs +0.02), not just "feature is important"
- **Comparable**: Can compare bias across different models or time periods
- **Actionable**: Can set thresholds (e.g., "bias must be < 0.05") for compliance

**Contrast with non-additive methods**: Grad-CAM or attention maps show "where the model looks" but don't give you the numerical accounting needed for regulatory compliance.

---

#### 4. **Counterfactual Reasoning**

Additive decomposition makes it **tractable to answer "what if" questions**:

**Question**: "What would my credit score be if my income increased by \$10k?"

**With additive decomposition**:
- Current contribution of income: $\phi_{\text{income}} = +35$ points
- Estimated new contribution: $\phi'_{\text{income}} \approx +42$ points (via local linearity)
- Expected change: $\approx +7$ points

**Why tractable**: 
- Simple arithmetic on individual contributions
- No need to re-run the entire model
- Can explore multiple scenarios quickly

**Without additivity**: Would need to:
1. Generate new input with modified income
2. Run full model forward pass
3. Compare outputs
4. Repeat for each scenario (computationally expensive)

---

#### 5. **Model Comparison**

When choosing between models, additive decomposition makes comparison **tractable**:

**Scenario**: Comparing Model A vs Model B for loan approval

```
Feature          | Model A  | Model B  | Difference
-----------------|----------|----------|------------
Income           | +0.40    | +0.35    | -0.05
Credit score     | +0.30    | +0.40    | +0.10
Age              | +0.10    | +0.05    | -0.05
Employment       | +0.15    | +0.15    | 0.00
Debt ratio       | -0.20    | -0.18    | +0.02
```

**Tractable insights**:
- Model B weights credit score more heavily (+0.10 difference)
- Model A weights income more heavily (-0.05 difference)
- Both models treat employment similarly

**Decision**: If credit score is more reliable than income, choose Model B.

**Why tractable**: Direct numerical comparison of how models use features. Non-additive explanations (like feature importance rankings) lose this quantitative precision.

---

#### 6. **Communication to Stakeholders**

Explaining model decisions to non-technical stakeholders becomes **tractable**:

**Loan rejection letter**:
```
Your application was denied (score: 580, threshold: 620, gap: -40 points)

Contributing factors:
  • Credit history:        -25 points (largest negative factor)
  • Debt-to-income ratio:  -18 points
  • Recent inquiries:       -8 points
  • Income level:          +11 points (positive factor)
  
To improve your chances:
  1. Focus on credit history (contributes -25 points)
  2. Reduce debt-to-income ratio (contributes -18 points)
```

**Why tractable**:
- **Precise**: "Your credit history cost you 25 points"
- **Actionable**: Stakeholder knows what to fix and by how much
- **Legally defensible**: Can show exactly why decision was made

---

#### 7. **Cognitive Load Reduction**

Humans can only hold ~7±2 items in working memory. Additive decomposition makes interpretation **cognitively tractable**:

**Instead of**: "The model considers complex interactions between income, age, credit score, employment, and debt in a non-linear way..."

**You get**: 
1. Income: +35
2. Credit: +28
3. Debt: -15
4. Employment: +18
5. Inquiries: +4

**Total**: +70 ✓

This is a **closed system** you can reason about. Each number is independent (in the explanation space), so you can:
- Focus on top-K features
- Build mental model incrementally
- Verify your understanding (do the numbers add up?)

---

### Mathematical Tractability

The additivity also provides **mathematical tractability**:

**Linearity in explanation space**: Even though $f$ may be highly non-linear, the explanations are linear:
$$\phi = W \cdot \text{features} + b$$

This means:
- **Optimization is convex** when trying to achieve desired outcomes
- **Statistical analysis is straightforward** (mean, variance, correlation of contributions)
- **Interventions compose**: $\Delta\phi_i + \Delta\phi_j = \Delta\phi_{i,j}$

---

### The Trade-offs

**What we sacrifice for tractability**:
- **Computational cost**: Exact Shapley values are exponential in features
- **Choice of baseline**: Different baselines give different attributions (though sum is always preserved)
- **Theoretical guarantees**: Approximations (LIME, KernelSHAP) may not perfectly satisfy efficiency
- **Model compatibility**: Some methods work only for specific architectures

But these trade-offs are often worth it because the alternative—trying to understand a black-box model without decomposition—is cognitively and practically **intractable**.