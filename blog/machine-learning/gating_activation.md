@def title = "Self-gating activation functions"
@def published = "6 October 2025"
@def tags = ["machine-learning"]

# Self-gating activation functions

## Overview

This table presents common activation functions used in neural networks, expressed in a unified form: $f(x) = g(x) \odot x$, where $g(x)$ is applied elementwise and $\odot$ represents elementwise multiplication.

**Terminology note**: While "gating" is standard in recurrent architectures (LSTMs, GRUs) where separate signals control information flow, this specific formulation for activation functions is sometimes called **"self-gating"** (as coined in the Swish paper) because the input gates itself: $x \cdot g(x)$. However, this isn't universally standard terminology—the key insight is simply that these functions can be viewed as **multiplicative modulation** of the input.

## Activation Functions

| **Activation** | **Gating Function $g(x_i)$** |
|----------------|------------------------------|
| **ReLU** | $\mathbb{1}(x_i > 0)$ |
| **Leaky ReLU** | $\mathbb{1}(x_i > 0) + \alpha \cdot \mathbb{1}(x_i \leq 0)$ |
| **PReLU** | $\mathbb{1}(x_i > 0) + \alpha_i \cdot \mathbb{1}(x_i \leq 0)$ |
| **ELU** | $\mathbb{1}(x_i > 0) + \alpha(e^{x_i} - 1)/x_i \cdot \mathbb{1}(x_i \leq 0)$ |
| **Swish/SiLU** | $\sigma(x_i) = \frac{1}{1 + e^{-x_i}}$ |
| **GELU** | $\Phi(x_i)$ |
| **Mish** | $\tanh(\text{softplus}(x_i)) = \tanh(\ln(1 + e^{x_i}))$ |
| **Hard Sigmoid** | $\max(0, \min(1, \frac{x_i + 1}{2}))$ |
| **Hard Swish** | $\max(0, \min(1, \frac{x_i + 3}{6}))$ |

## Key Concepts

**Indicator Function**: $\mathbb{1}(condition)$ equals 1 when the condition is true, 0 otherwise.

**Sigmoid Function**: $\sigma(x) = \frac{1}{1 + e^{-x}}$ produces values between 0 and 1.

**Gaussian CDF**: $\Phi(x)$ is the cumulative distribution function of the standard normal distribution.

**Softplus**: $\text{softplus}(x) = \ln(1 + e^x)$ is a smooth approximation of ReLU.

## Historical Context: Why the Gating Form Matters

### Classical Smooth Activations (Pre-2010s)
Traditional activations like **sigmoid** ($\sigma(x) = \frac{1}{1+e^{-x}}$) and **tanh** are **transformation functions** that replace the input:
- Sigmoid: $f(x) = \sigma(x)$ outputs [0,1]
- Tanh: $f(x) = \tanh(x)$ outputs [-1,1]

**These do NOT fit the gating form $f(x) = g(x) \odot x$** - they transform rather than modulate.

**Problems with classical activations:**
- **Vanishing gradients**: For large $|x|$, sigmoid/tanh saturate (gradient → 0)
- **Bounded outputs**: Cause gradient scaling issues in deep networks
- **Information loss**: Input magnitude information is compressed

### The ReLU Revolution (2010s)
ReLU ($f(x) = \max(0, x) = \mathbb{1}(x>0) \cdot x$) can be viewed in this multiplicative form:
- **Hard multiplicative gating**: Binary gate that's either 0 or 1
- **No saturation** for positive values (gradient = 1)
- **Sparse activations** (exactly 0 for negative inputs)
- **Computationally cheap** (simple thresholding)
- But introduced "dying ReLU" problem (neurons stuck at 0)

### Modern Self-Gated Activations (2017+)
The term **"self-gated"** comes from the Swish paper, describing activations where the input modulates itself through a smooth function:
- **Swish/SiLU**: $f(x) = x \cdot \sigma(x)$ - called "self-gated" because $x$ gates itself via sigmoid
- **GELU**: $f(x) = x \cdot \Phi(x)$ - probabilistic gating, used in BERT/GPT
- **Mish**: $f(x) = x \cdot \tanh(\ln(1+e^x))$ - smoother than Swish

**Key insight:** The multiplicative form $f(x) = g(x) \cdot x$ preserves input magnitude information while smoothly controlling signal flow. Unlike sigmoid which replaces the input with a bounded value, these functions modulate the input, allowing unbounded growth for large positive values (no vanishing gradients) while smoothly suppressing negative values.

## How Multiplication Creates the Characteristic Shapes

Understanding why $f(x) = x \cdot g(x)$ produces functions like GELU requires seeing how the gate $g(x)$ modulates the linear term $x$.

### Example: GELU Decomposition

**GELU**: $f(x) = x \cdot \Phi(x)$ where $\Phi(x)$ is the standard normal CDF

**How the multiplication works:**

1. **Linear component** $x$: Just a straight line through the origin
2. **Gate component** $\Phi(x)$: S-shaped curve from 0 to 1
   - $\Phi(-\infty) = 0$ (gate fully closed)
   - $\Phi(0) \approx 0.5$ (gate half-open)
   - $\Phi(+\infty) = 1$ (gate fully open)

3. **Product** $x \cdot \Phi(x)$:
   - When $x \ll 0$: $\Phi(x) \approx 0$, so $f(x) \approx x \cdot 0 = 0$ (negative inputs suppressed)
   - When $x \approx 0$: $\Phi(x) \approx 0.5$, so $f(x) \approx 0.5x$ (transition region)
   - When $x \gg 0$: $\Phi(x) \approx 1$, so $f(x) \approx x \cdot 1 = x$ (positive inputs pass through)

**The resulting shape:**
- Smooth curve near origin (not sharp like ReLU)
- Small negative bump for slightly negative $x$ (gate opens gradually, not abruptly)
- Asymptotically linear for large positive $x$ (gate fully open → acts like identity)
- Bounded below at approximately $-0.17$ (the gate never fully closes for finite $x$)

### Why This Works Better Than Classical Activations

**Sigmoid alone**: $\sigma(x)$ squashes everything to [0,1] → loses magnitude information

**GELU/Swish**: $x \cdot \sigma(x)$ or $x \cdot \Phi(x)$ → magnitude grows linearly when $x$ is large and positive (gate ≈ 1), but smoothly suppresses when $x$ is negative (gate ≈ 0). This preserves gradient flow while adding smooth nonlinearity.

## What's Happening?

This formulation reveals that many activation functions can be viewed as **multiplicative modulation mechanisms** that control how much of the input signal should pass through at each position. The term **"self-gating"** (from the Swish paper) specifically refers to activations where the input gates itself via a smooth function: $f(x) = x \cdot g(x)$.

Rather than seeing activations as arbitrary nonlinearities, this perspective shows they determine "how much" of the input should pass through. Modern smooth activations (Swish, GELU, Mish) use data-dependent, continuous modulation that allows for richer gradient flow compared to hard binary gates like ReLU.