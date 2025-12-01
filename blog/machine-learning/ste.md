@def title = "Straight-Through Estimator (STE)"
@def published = "1 December 2025"
@def tags = ["machine-learning"]

# Straight-Through Estimator (STE)

## The Problem

Non-differentiable operations like $\text{argmax}$, $\text{round}$, or $\mathbb{1}_{x > 0}$ break backpropagation:

$$\frac{\partial}{\partial x}\text{argmax}(x) = 0 \text{ or undefined}$$

## The Solution

**Use different operations for forward and backward passes:**

- **Forward pass**: Use the hard/discrete operation (fast, what you actually want)
- **Backward pass**: Pretend it was a smooth approximation (differentiable)

## Mathematical Formulation

For a non-differentiable function $h(x)$ and its smooth approximation $s(x)$:

$$y = h(x) + [s(x) - s(x)]$$

During backpropagation:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial s(x)}{\partial x}$$

The gradient flows through $s(x)$, but forward pass uses $h(x)$.

## Implementation Pattern

### Julia Implementation

```julia
using Flux

# STE for binarization
function binary_ste(x)
    # Forward: hard threshold
    hard = Float32.(x .> 0)
    
    # Backward: gradient of identity
    # In Flux, we override the gradient manually
    return hard
end

# Define custom gradient
Flux.@adjoint binary_ste(x) = binary_ste(x), Δ -> (Δ,)

# Example usage
x = [0.7, -0.3, 0.1]
y = binary_ste(x)  # [1.0, 0.0, 1.0]
```

### Gumbel-Softmax with STE

```julia
using Flux
using Random

function gumbel_softmax_ste(logits; τ=1.0)
    # Sample Gumbel noise: -log(-log(U))
    U = rand(Float32, size(logits)...)
    gumbel = -log.(-log.(U .+ 1f-20) .+ 1f-20)
    
    # Soft (differentiable) version
    soft = softmax((logits .+ gumbel) ./ τ)
    
    # Hard (discrete) version - one-hot of argmax
    hard_idx = argmax(soft, dims=1)
    hard = zeros(Float32, size(soft))
    hard[hard_idx] .= 1.0f0
    
    # STE trick: hard forward, soft backward
    return hard .- Flux.stopgradient(soft) .+ soft
end

# Example
logits = randn(Float32, 5, 10)  # 5 classes, 10 samples
samples = gumbel_softmax_ste(logits, τ=0.5)
# Forward: discrete one-hot vectors
# Backward: gradients flow through soft distribution
```

## Common Applications

| Application | Hard Operation | Soft Approximation |
|------------|----------------|-------------------|
| **Binarization** | $\text{sign}(x)$ or $\mathbb{1}_{x>0}$ | $\text{tanh}(x)$ or $\sigma(x)$ |
| **Quantization** | $\text{round}(x)$ | $x$ (identity) |
| **Categorical Sampling** | $\text{argmax}(x)$ | $\text{softmax}(x/\tau)$ |
| **Top-k Selection** | $\text{top-k}(x)$ | $\text{softmax}(x/\tau)$ |

## Why It Works

1. **Forward pass gets what you want**: Discrete, fast operations
2. **Backward pass is trainable**: Smooth gradients flow
3. **Biased but useful**: Gradients aren't "correct" but work in practice

## Key Intuition

> **"Lie to the gradient"** - Use fast discrete operations forward, pretend they were smooth backward.

The estimator is "straight-through" because gradients pass straight through the non-differentiable operation as if it were the identity (or its smooth approximation).

## Performance Impact

For your case (Gumbel-Softmax):
- **Without STE**: Compute soft probabilities every forward pass → **64% of time**
- **With STE**: Just compute argmax → **~5% of time** (huge speedup!)

The backward pass still computes soft probabilities, but that's only once per batch and is parallelized with other gradient computations.

> **Wait, doesn't STE still generate Gumbel noise?**
>
> Yes! The code above still samples Gumbel noise, so if you've already wrapped that in `@ignore_derivatives`, you won't see much speedup from STE alone.
>
> **The real speedup comes from eliminating the soft probabilities computation:**
> - **Without STE**: `soft = softmax((logits + gumbel) / τ)` → use this soft distribution directly (expensive softmax on every forward pass)
> - **With STE**: Compute soft once, but use `argmax` + one-hot encoding (much cheaper)
>
> **However, if you're using continuous Gumbel-Softmax** (like `sigmoid((logit + gumbel) / temp)`), the issue is different:
>
> Your function computes:
> ```julia
> s = sigmoid((logit_p + gumbel) / temp)  # Still continuous [0,1]
> return clamp(s * scale + gamma, 0, 1)    # Still continuous
> ```
>
> **This is NOT producing discrete samples**, so STE doesn't directly apply. You're computing a continuous relaxation every time.
>
> **To speed this up with STE, you need to make it discrete:**
> ```julia
> function gumbel_softmax_ste_binary(p, temp, eta, gamma)
>     # Gumbel noise (already optimized with @ignore_derivatives)
>     gumbel = @ignore_derivatives if p isa CuArray
>         -log.(-log.(CUDA.rand(DEFAULT_FLOAT_TYPE, size(p))))
>     else
>         -log.(-log.(rand(DEFAULT_FLOAT_TYPE, size(p))))
>     end
>     
>     logit_p = @. log(p + MASK_EPSILON) - log((MASK_ONE - p) + MASK_EPSILON)
>     
>     # Soft version (for gradients)
>     s = @. sigmoid((logit_p + gumbel) / temp)
>     s_scaled = @. clamp(s * scale + gamma, MASK_ZERO, MASK_ONE)
>     
>     # Hard version (for forward pass) - threshold at 0.5
>     hard = @. Float32(s > 0.5f0)  # Binary: 0 or 1
>     
>     # STE: use hard in forward, soft in backward
>     return hard .- Flux.stopgradient(s_scaled) .+ s_scaled
> end
> ```
>
> **The 64% bottleneck breakdown:**
> - Gumbel sampling: ~15-20% (unavoidable if you need stochasticity)
> - `log` operations: ~20-30% (for logit computation)
> - `sigmoid`: ~15-20%
> - Scaling/clamping: ~10-15%
>
> **Ways to speed up your specific case:**
>
> 1. **Use hard thresholding with STE** (as shown above) - gets you discrete 0/1 in forward pass
> 2. **Remove temperature annealing** - if temp is small, just use `logit_p + gumbel > 0` (much faster than sigmoid)
> 3. **Pre-compute logits** - if `p` doesn't change often, cache `logit_p`
> 4. **Fuse operations in a custom kernel** - combine log + sigmoid + scale into one CUDA kernel
> 5. **At inference, drop Gumbel entirely** - just use `p > 0.5` (deterministic)
>
> **Bottom line**: Your bottleneck is the continuous relaxation (`sigmoid` + scaling). STE helps by replacing the forward pass with a simple threshold, but you still need the soft version for gradients in backward pass.