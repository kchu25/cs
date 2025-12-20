@def title = "DeepSHAP: Additive Decomposition and DeepLIFT Connection"
@def published = "20 December 2025"
@def tags = ["interpretation"]

# DeepSHAP: Additive Decomposition and DeepLIFT Connection

## The Additive Decomposition Framework

DeepSHAP provides an **additive decomposition** of a model's output, meaning it expresses the prediction as a sum of individual feature contributions plus a base value:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

where:
- $f(x)$ is the model's output for input $x$
- $\phi_0$ is the base value (the baseline prediction)
- $\phi_i$ is the contribution of feature $i$ to the deviation from the base value
- $M$ is the total number of features

> ### Understanding $\phi_0$
> 
> **Yes, $\phi_0 = f(x_{\text{ref}})$ is the baseline prediction!**
> 
> More precisely, when using multiple reference points (as DeepSHAP does):
> 
> $$\phi_0 = \mathbb{E}_{x_{\text{ref}}}[f(x_{\text{ref}})] \approx \frac{1}{K}\sum_{k=1}^{K} f(x_{\text{ref}}^{(k)})$$
> 
> So the additive decomposition becomes:
> 
> $$f(x) = \underbrace{f(x_{\text{ref}})}_{\text{baseline}} + \underbrace{\sum_{i=1}^{M} \phi_i}_{\text{feature contributions}}$$
> 
> or equivalently:
> 
> $$\underbrace{f(x) - f(x_{\text{ref}})}_{\text{prediction difference}} = \sum_{i=1}^{M} \phi_i$$
> 
> **Interpretation**: "The prediction on input $x$ equals the baseline prediction plus the sum of all feature contributions"

**Key Property**: The contributions $\phi_i$ satisfy the **efficiency property**:

$$f(x) - f(x_{\text{ref}}) = \sum_{i=1}^{M} \phi_i$$

This means the sum of all feature contributions exactly equals the difference between the prediction and the baseline prediction.

---

## Connection to DeepLIFT Scores

DeepSHAP approximates these additive contributions $\phi_i$ by **aggregating DeepLIFT scores** across multiple reference points. Here's how:

### 1. Single Reference DeepLIFT

For a single reference input $x_{\text{ref}}$, DeepLIFT computes contribution scores $C_{\Delta x_i \Delta y}$ that decompose the output difference:

$$\Delta y = y - y_{\text{ref}} = \sum_{i=1}^{M} C_{\Delta x_i \Delta y}$$

where $C_{\Delta x_i \Delta y}$ represents how much the change in feature $i$ contributed to the change in output.

### 2. DeepSHAP Aggregation

DeepSHAP extends this by considering **multiple reference points** $\{x_{\text{ref}}^{(k)}\}_{k=1}^{K}$ sampled from a reference distribution (e.g., background dataset). The SHAP value for feature $i$ is approximated as:

$$\phi_i \approx \frac{1}{K} \sum_{k=1}^{K} C_{\Delta x_i \Delta y}^{(k)}$$

where $C_{\Delta x_i \Delta y}^{(k)}$ is the DeepLIFT contribution score computed with respect to reference $x_{\text{ref}}^{(k)}$.

---

## Why This Connection Matters

### Additive Decomposition Properties

The additive form is crucial because it provides:

1. **Local Accuracy**: $\sum_{i} \phi_i = f(x) - \mathbb{E}[f(X)]$ (approximately)
2. **Missingness**: Features not present have zero contribution
3. **Consistency**: Similar feature changes yield similar attribution changes

### DeepLIFT as the Computational Engine

DeepLIFT provides the **efficient backpropagation-based computation** of these contributions:

- **Chain Rule Decomposition**: DeepLIFT propagates contributions backwards through the network using modified chain rules
- **Non-linear Handling**: It handles non-linearities (ReLU, sigmoid, etc.) by assigning contributions based on the difference from reference
- **Efficiency**: Single backward pass per reference point (similar cost to gradient computation)

---

## Mathematical Flow

The complete process can be summarized as:

$$\boxed{\phi_i = \mathbb{E}_{x_{\text{ref}} \sim \mathcal{D}}[C_{\Delta x_i \Delta y}(x, x_{\text{ref}})]}$$

This equation encapsulates:
- **Left side**: The additive SHAP contribution for feature $i$
- **Right side**: The expectation of DeepLIFT scores over a reference distribution

In practice:

$$\phi_i \approx \frac{1}{K} \sum_{k=1}^{K} \text{DeepLIFT}(x, x_{\text{ref}}^{(k)})_i$$

---

## Practical Implications

1. **Additive decomposition** ensures we can understand $f(x)$ as: baseline + contribution₁ + contribution₂ + ... + contributionₘ

2. **DeepLIFT aggregation** provides computationally feasible approximation of true SHAP values for deep networks

3. **Multiple references** help capture the model's behavior across different contexts, making attributions more robust

The beauty of DeepSHAP is that it maintains the theoretical guarantees of additive decomposition while leveraging the computational efficiency of DeepLIFT's score propagation through neural networks.