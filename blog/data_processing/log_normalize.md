@def title = "Log Min-Max Normalization"
@def published = "3 November 2025"
@def tags = ["data-processing"]

# Log Min-Max Normalization

A specialized normalization that combines logarithmic transformation with min-max scaling, useful for data spanning multiple orders of magnitude.

## Range: [0, 1]

### Forward Transform
$$y_{\text{norm}} = \frac{\log(y) - \log(y_{\min})}{\log(y_{\max}) - \log(y_{\min})}$$

### Inverse Transform
$$y = \exp\left(y_{\text{norm}} \cdot (\log(y_{\max}) - \log(y_{\min})) + \log(y_{\min})\right)$$

### Neural Network Settings
- **Output Activation:** Sigmoid
- **Output Range:** [0, 1]
- **Loss Function:** MSE or MAE on normalized values

---

## Range: [-1, 1]

### Forward Transform
$$y_{\text{norm}} = 2 \cdot \frac{\log(y) - \log(y_{\min})}{\log(y_{\max}) - \log(y_{\min})} - 1$$

### Inverse Transform
$$y = \exp\left(\frac{y_{\text{norm}} + 1}{2} \cdot (\log(y_{\max}) - \log(y_{\min})) + \log(y_{\min})\right)$$

### Neural Network Settings
- **Output Activation:** Tanh
- **Output Range:** [-1, 1]
- **Loss Function:** MSE or MAE on normalized values

---

## When to Use

✅ **Good for:**
- Data spanning many orders of magnitude (prices, populations, counts)
- Multiplicative relationships that need bounded outputs
- Need guaranteed output bounds for production safety
- Requires strictly positive values

⚠️ **Note:** Plain log transform is usually sufficient! Add min-max only when you need guaranteed bounds. The added complexity is rarely worth it in most cases.

---

## Caveats & Limitations

🚨 **Critical Issues:**
- **Requires strictly positive values:** Cannot handle y ≤ 0 (log undefined)
- **Sensitive to min/max from training data:** If test data has values outside [y_min, y_max], the normalization breaks
- **Extrapolation problems:** Neural network can't predict beyond training bounds even if real data extends further
- **Zero-crossing issues:** If your data can be zero or negative, this method fails entirely

⚠️ **Practical Concerns:**
- **Outliers define the scale:** A single extreme value in training sets y_max, compressing all other values
- **Non-robust:** Unlike quantile/IQR methods, sensitive to extreme values
- **Compression at extremes:** Values near y_min or y_max get compressed into small regions of [0,1] or [-1,1]
- **Sigmoid/tanh saturation:** Network outputs at boundaries have vanishing gradients, making learning harder

🔧 **Implementation Pitfalls:**
- Must store y_min and y_max from training for inference
- Numerical instability: log of values close to zero can be very negative
- If y_min ≈ y_max, division by near-zero causes numerical issues

---

## Log Transforms: Sensitivity Near Zero vs Large Values

**Problem:** Standard log is extremely sensitive near zero (log(0.001) = -6.9, log(0.0001) = -9.2) but less sensitive for large values.

### Solution 1: Arcsinh Transform

$$y_{\text{norm}} = \sinh^{-1}(y) = \log(y + \sqrt{y^2 + 1})$$

**Inverse:**
$$y = \sinh(y_{\text{norm}}) = \frac{e^{y_{\text{norm}}} - e^{-y_{\text{norm}}}}{2}$$

**Behavior:**
- **Near zero:** Linear with derivative = 1 (standard sensitivity, like identity)
- **Large values:** Approximately logarithmic (≈ log(2y))
- **Works with:** Zero, positive, and negative values

**Use when:** You want a smooth transition from linear to logarithmic, without arbitrary constants

### Solution 2: Log(1 + y) Transform

$$y_{\text{norm}} = \log(1 + y)$$

**Inverse:**
$$y = e^{y_{\text{norm}}} - 1$$

**Behavior:**
- **Near zero:** More linear than log(y), derivative ≈ 1 at y=0
- **Large values:** Logarithmic (offset by 1)
- **Requires:** y ≥ 0 (but handles zero!)

### Solution 3: Log(y + c) with Large Constant

$$y_{\text{norm}} = \log(y + c)$$

**Inverse:**
$$y = e^{y_{\text{norm}}} - c$$

**Behavior:**
- **At y=0:** Derivative = 1/c (less sensitive with larger c)
- **At y=50:** Derivative = 1/(50+c)
- **Sensitivity ratio:** (50+c)/c = 1 + 50/c

| Constant c | Derivative at y=0 | Sensitivity Ratio |
|------------|-------------------|-------------------|
| 3 | 0.333 | 17.7x |
| 10 | 0.100 | 6.0x |
| 50 | 0.020 | 2.0x |
| 100 | 0.010 | 1.5x |

**Use when:** You want to reduce sensitivity near zero by making the function flatter. Larger c = less sensitive near zero, but also reduces compression of large values.

### Solution 4: Shifted Log with Threshold

$$y_{\text{norm}} = \begin{cases}
y & \text{if } y < \theta \\
\theta + \log(1 + y - \theta) & \text{if } y \geq \theta
\end{cases}$$

**Behavior:**
- **Below threshold θ:** Identity (standard sensitivity)
- **Above threshold θ:** Logarithmic compression
- **Sharp transition** at θ

### Solution 5: Log-Modulus Transform

$$y_{\text{norm}} = \text{sign}(y) \cdot \log(1 + |y|)$$

**Inverse:**
$$y = \text{sign}(y_{\text{norm}}) \cdot (e^{|y_{\text{norm}}|} - 1)$$

**Behavior:**
- Symmetric for positive and negative
- Linear near zero, logarithmic for large magnitudes
- Handles negatives naturally

### Comparison Table

| Transform | Derivative at y=0 | Derivative at y=50 | Sensitivity Ratio | Good for |
|-----------|------------------|-------------------|-------------------|----------|
| log(y) | ∞ | 0.020 | ∞ | Never use for y≈0 |
| arcsinh(y) | 1.000 | 0.020 | **50x** | Smooth linear→log transition |
| log(1+y) | 1.000 | 0.020 | **50x** | Similar to arcsinh, simpler |
| log(y+3) | 0.333 | 0.019 | **17x** | Medium clustering |
| log(y+10) | 0.100 | 0.017 | **6x** | More clustering |
| log(y+50) | 0.020 | 0.010 | **2x** | Heavy clustering |
| log(y+100) | 0.010 | 0.007 | **1.5x** | Extreme clustering |

**Key insight:** Higher sensitivity ratio means more sensitive at zero relative to large values.

---

## Real Example: Heavily Skewed Data (240k values, mostly near zero)

**Scenario:** 240k values, ~237k near zero (fractional), only ~3k above 50

### Quantifying Different Transforms

Let's compare transforms for small values:

| y value | log(y+3) | log(y+10) | log(y+50) | arcsinh(y) |
|---------|----------|-----------|-----------|------------|
| 0.001 | 1.099 | 2.303 | 3.912 | 0.001 |
| 0.01 | 1.101 | 2.304 | 3.913 | 0.010 |
| 0.1 | 1.131 | 2.332 | 3.932 | 0.100 |
| 1.0 | 1.386 | 2.398 | 3.932 | 0.881 |
| 10.0 | 2.565 | 2.996 | 4.094 | 2.998 |
| 50.0 | 3.970 | 4.094 | 4.605 | 4.605 |
| 100.0 | 4.635 | 4.700 | 5.011 | 5.298 |

**Key insight:** 
- log(y+50) and log(y+100): Fractional values (0.001 to 1.0) are tightly clustered
- arcsinh(y): Fractional values spread out more, maintains distinctions

### Recommended Strategy for Your Data

**Option 1: Log(y + 50) or Log(y + 100)** ⭐ **Best for your case**
```julia
# Forward
y_norm = log.(y .+ 50)

# Inverse
y = exp.(y_norm) .- 50
```

**Why this works for you:**
- Makes 237k near-zero values cluster tightly together
- Still compresses 3k large values logarithmically
- Sensitivity ratio only 2x (very flat near zero)
- Simple and interpretable

**Option 2: Two-Regime Transform (Custom threshold)**
```julia
threshold = 1.0  # tune based on your data

# Forward
y_norm = ifelse.(y .< threshold, 
                 y ./ threshold,  # linear for small values
                 1 .+ log.(y ./ threshold))  # log for large values

# Inverse  
y = ifelse.(y_norm .< 1,
            y_norm .* threshold,  # linear region
            threshold .* exp.(y_norm .- 1))  # log region
```

**Why this works:**
- Explicitly treats small values as "approximately zero"
- Only applies log compression above threshold
- You control the transition point

**Option 3: Quantile Transform (Nuclear option)**
```julia
using Statistics

# Fit on training data - store quantiles
function fit_quantile_transform(y)
    sorted_y = sort(y)
    n = length(y)
    quantiles = [(i-0.5)/n for i in 1:n]
    return sorted_y, quantiles
end

function transform_quantile(y, sorted_y, quantiles)
    # Map each value to its quantile rank
    ranks = [searchsortedlast(sorted_y, val) for val in y]
    return [quantiles[max(1, min(r, length(quantiles)))] for r in ranks]
end
```

**Why this works:**
- Completely distribution-agnostic
- 237k near-zero values spread evenly across output range
- Most robust to your extreme skew

### What NOT to Use for Your Data

❌ **arcsinh(y)** or **log(1+y)**: These have derivative = 1 at y=0, making them **more sensitive** to fractional differences than log(y+3). Your 237k near-zero values won't cluster together enough.

❌ **Plain log(y)**: Explodes at zero, unusable.

### Recommendation Summary

Given 98.75% of data is near zero:

1. **Best choice:** `log.(y .+ 50)` or `log.(y .+ 100)` → tight clustering of near-zero values
2. **Custom control:** Two-regime transform with threshold → explicit transition point
3. **Maximum robustness:** Quantile transform → if only ordinal relationships matter

---

## Why log(y + c) Sensitivity Depends on c

**Common approach:** Add constant to handle zeros: `log(y + c)`

### Numerical Example with log(y + 3)

| Original y | log(y + 3) | Δ from previous |
|------------|------------|-----------------|
| 0.0 | 1.099 | - |
| 0.1 | 1.131 | **0.032** |
| 0.5 | 1.253 | **0.122** |
| 1.0 | 1.386 | **0.133** |
| 5.0 | 2.079 | **0.693** |
| 10.0 | 2.565 | **0.486** |
| 50.0 | 3.970 | **1.405** |
| 100.0 | 4.635 | **0.665** |

**Key issue:** The derivative of log(y + c) is:
$$\frac{d}{dy}\log(y + c) = \frac{1}{y + c}$$

At y = 0: derivative = 1/3 = 0.333  
At y = 50: derivative = 1/53 = 0.019

**Sensitivity ratio: 17.5x more sensitive at zero than at 50!** 🚨

> **What does "17x more sensitive" mean?**
> 
> Sensitivity = how much the transformed value changes for a small input change.
> 
> - At y=0: increasing by 0.1 → log changes by ~0.033
> - At y=50: increasing by 0.1 → log changes by ~0.0019
> - Ratio: 0.033 / 0.0019 = **17.5x**
> 
> So the **same +0.1 change** creates **17x bigger change** in log-space at y=0 vs y=50.
> 
> **Why this matters:** Your 237k fractional values have noise/differences that create large changes in log-space, dominating the loss function. Meanwhile, meaningful differences in large values (50→60) create smaller changes. You might want the opposite!

### How to Reduce Sensitivity at Zero

**Option: Use larger constant**
```julia
y_norm = log.(y .+ 100)  # Much flatter near zero
```
- At y=0: derivative = 1/100 = 0.01
- At y=50: derivative = 1/150 = 0.0067
- Now only 1.5x more sensitive (much better!)
- Trade-off: Also reduces compression of large values

### Comparison: Sensitivity at y=0

| Transform | Derivative at y=0 | Sensitivity Ratio | Use Case |
|-----------|------------------|-------------------|----------|
| log(y) | **undefined** | ∞ | Never near zero |
| log(y + 3) | **0.333** | 17x | Medium sensitivity |
| log(y + 10) | **0.100** | 6x | Lower sensitivity |
| log(y + 50) | **0.020** | 2x | Very flat near zero |
| log(y + 100) | **0.010** | 1.5x | Extremely flat |
| arcsinh(y) | **1.000** | 50x | More sensitive at zero! |

---

## Power Transforms: Box-Cox and Yeo-Johnson

### Box-Cox Transform (y > 0 only)

$y_{\text{norm}} = \begin{cases}
\frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(y) & \text{if } \lambda = 0
\end{cases}$

**Inverse:**
$y = \begin{cases}
(\lambda \cdot y_{\text{norm}} + 1)^{1/\lambda} & \text{if } \lambda \neq 0 \\
e^{y_{\text{norm}}} & \text{if } \lambda = 0
\end{cases}$

**Derivative at y (for λ ≠ 0):**
$\frac{d}{dy}\text{BoxCox}(y) = y^{\lambda - 1}$

**Sensitivity Analysis:**

| λ value | Name | Deriv at y=0.1 | Deriv at y=50 | Ratio (0.1:50) | Effect |
|---------|------|----------------|---------------|----------------|--------|
| **2.0** | Square | 0.2 | 100 | **0.002x** | Very insensitive at zero |
| **1.0** | Identity | 1.0 | 1.0 | **1x** | Uniform sensitivity |
| **0.5** | Square root | 3.16 | 0.141 | **22.4x** | Sensitive at zero |
| **0.0** | Log | ∞ | 0.02 | **∞** | Explodes at zero |
| **-0.5** | Inverse sqrt | 31.6 | 0.0028 | **11,300x** | Extremely sensitive at zero |
| **-1.0** | Inverse | 100 | 0.0004 | **250,000x** | Wildly sensitive at zero |

**Key insights:**
- **λ > 1**: Makes function MORE sensitive to large values, LESS sensitive to small values (compresses near zero)
- **λ = 1**: Identity transform, uniform sensitivity
- **0 < λ < 1**: Makes function MORE sensitive to small values (like square root)
- **λ = 0**: Log transform (special case)
- **λ < 0**: Extremely sensitive to small values, unstable near zero

**For your data (237k near zero):** Use **λ = 2 or λ = 3** to compress small values!

```julia
# Forward (λ = 2)
lambda = 2.0
y_norm = (y.^lambda .- 1) ./ lambda

# Inverse
y = (lambda .* y_norm .+ 1).^(1/lambda)
```

### Yeo-Johnson Transform (handles negatives and zero)

$y_{\text{norm}} = \begin{cases}
\frac{(y+1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, y \geq 0 \\
\log(y+1) & \text{if } \lambda = 0, y \geq 0 \\
-\frac{(-y+1)^{2-\lambda} - 1}{2-\lambda} & \text{if } \lambda \neq 2, y < 0 \\
-\log(-y+1) & \text{if } \lambda = 2, y < 0
\end{cases}$

**Derivative at y=0.1 (for positive values, λ ≠ 0):**
$\frac{d}{dy}\text{YeoJohnson}(y) = (y+1)^{\lambda - 1}$

**Sensitivity Analysis (positive values only):**

| λ value | Deriv at y=0.1 | Deriv at y=50 | Ratio (0.1:50) | Effect |
|---------|----------------|---------------|----------------|--------|
| **2.0** | 0.289 | 2601 | **0.0001x** | Very insensitive at zero |
| **1.0** | 1.0 | 51 | **0.02x** | Like log(y+1) |
| **0.5** | 3.02 | 7.14 | **0.42x** | Moderate |
| **0.0** | ∞ (≈9.1) | 0.0196 | **≈500x** | Like log(y+1) |
| **-0.5** | 27.5 | 0.0014 | **19,600x** | Very sensitive at zero |

**Key difference from Box-Cox:** Yeo-Johnson uses `(y+1)` instead of `y`, so it handles zero naturally!

**For your data:** Use **λ = 1.5 or λ = 2.0** to reduce sensitivity near zero while handling zeros.

```julia
# Forward (λ = 2, for y ≥ 0)
lambda = 2.0
y_norm = ((y .+ 1).^lambda .- 1) ./ lambda

# Inverse
y = (lambda .* y_norm .+ 1).^(1/lambda) .- 1
```

### Comprehensive Sensitivity Comparison

For your dataset (0.001 to 100, mostly near zero):

| Transform | Deriv at 0.1 | Deriv at 50 | Ratio | Clusters near-zero? |
|-----------|--------------|-------------|-------|---------------------|
| **Box-Cox λ=3** | 0.01 | 2500 | 0.000004x | ✅✅✅ Best clustering |
| **Box-Cox λ=2** | 0.1 | 100 | 0.001x | ✅✅ Excellent |
| **Yeo-Johnson λ=2** | 0.29 | 2601 | 0.0001x | ✅✅ Excellent |
| **log(y+100)** | 0.010 | 0.0067 | 1.5x | ✅ Good |
| **log(y+50)** | 0.020 | 0.010 | 2x | ✅ Good |
| **log(y+10)** | 0.100 | 0.017 | 6x | ⚠️ Moderate |
| **log(y+3)** | 0.333 | 0.019 | 17x | ⚠️ Some sensitivity |
| **Box-Cox λ=1** | 1.0 | 1.0 | 1x | ❌ Identity |
| **arcsinh(y)** | 1.0 | 0.02 | 50x | ❌ Too sensitive |
| **Box-Cox λ=0.5** | 3.16 | 0.141 | 22x | ❌ Too sensitive |
| **log(y)** | ∞ | 0.02 | ∞ | ❌ Explodes |

### Recommendation for Your Data (237k near zero)

**Best options ranked:**

1. **Box-Cox with λ = 2 or λ = 3** ⭐⭐⭐ 
   - Extreme compression of small values
   - Sensitivity ratio < 0.001x means near-zero values cluster extremely tightly
   - Large values still get expanded/emphasized
   
2. **Yeo-Johnson with λ = 2** ⭐⭐
   - Similar to Box-Cox but handles exact zeros
   - Good if your data might have zeros
   
3. **log(y + 100)** ⭐
   - Simpler, no λ to tune
   - Good clustering but not as extreme as Box-Cox

**How to find optimal λ:**
```julia
using Optim

# Find λ that maximizes log-likelihood (makes data most Gaussian)
function box_cox_loglik(lambda, y)
    if lambda == 0
        return sum(log.(y))
    else
        y_trans = (y.^lambda .- 1) ./ lambda
        return -(length(y)/2) * log(var(y_trans)) + (lambda - 1) * sum(log.(y))
    end
end

# Optimize
result = optimize(lambda -> -box_cox_loglik(lambda[1], y), [0.5])
optimal_lambda = result.minimizer[1]
```

Or simply try λ = 2 as a starting point for heavy compression of small values!

---

## How is λ Typically Determined?

### Method 1: Maximum Likelihood Estimation (MLE) ⭐ **Standard approach**

**Goal:** Find λ that makes the transformed data most "Gaussian-like"

**Box-Cox Log-Likelihood:**
$\ell(\lambda) = -\frac{n}{2}\log(\sigma^2_\lambda) + (\lambda - 1)\sum_{i=1}^n \log(y_i)$

where $\sigma^2_\lambda$ is the variance of the transformed data.

**Implementation:**
```julia
using Optim, Statistics

function box_cox_mle(y; lambda_range=(-2.0, 3.0))
    # Objective: negative log-likelihood
    function neg_loglik(lambda)
        if abs(lambda) < 1e-10
            y_trans = log.(y)
        else
            y_trans = (y.^lambda .- 1) ./ lambda
        end
        
        n = length(y)
        sigma2 = var(y_trans)
        
        # Log-likelihood
        ll = -n/2 * log(sigma2) + (lambda - 1) * sum(log.(y))
        
        return -ll  # Return negative for minimization
    end
    
    # Optimize
    result = optimize(neg_loglik, lambda_range[1], lambda_range[2])
    return result.minimizer
end

# Use it
optimal_lambda = box_cox_mle(y)
println("Optimal λ = $optimal_lambda")
```

**Pros:** 
- Mathematically principled
- Widely used in statistics
- Works well when goal is normality

**Cons:**
- Assumes you want Gaussian output (may not be your goal!)
- Can give unexpected λ values
- For your data (237k near zero), might give λ < 1 which increases sensitivity at zero

### Method 2: Cross-Validation on Downstream Task ⭐⭐ **Best for ML**

**Goal:** Find λ that gives best performance on your actual prediction task

```julia
using Statistics

function cross_validate_lambda(X, y, lambdas; n_folds=5)
    n = length(y)
    fold_size = div(n, n_folds)
    
    best_lambda = nothing
    best_score = Inf
    
    for lambda in lambdas
        scores = []
        
        for fold in 1:n_folds
            # Split data
            val_idx = (fold-1)*fold_size+1 : min(fold*fold_size, n)
            train_idx = setdiff(1:n, val_idx)
            
            # Transform with current lambda
            if abs(lambda) < 1e-10
                y_train_trans = log.(y[train_idx])
                y_val_trans = log.(y[val_idx])
            else
                y_train_trans = (y[train_idx].^lambda .- 1) ./ lambda
                y_val_trans = (y[val_idx].^lambda .- 1) ./ lambda
            end
            
            # Train model (placeholder - use your actual model)
            # model = train_your_model(X[train_idx], y_train_trans)
            # predictions = predict(model, X[val_idx])
            # score = mean((predictions - y_val_trans).^2)
            
            # For now, just measure variance (lower = more compressed)
            score = var(y_val_trans)
            push!(scores, score)
        end
        
        avg_score = mean(scores)
        if avg_score < best_score
            best_score = avg_score
            best_lambda = lambda
        end
        
        println("λ = $lambda, avg score = $avg_score")
    end
    
    return best_lambda
end

# Try a grid of lambdas
lambdas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
best_lambda = cross_validate_lambda(X, y, lambdas)
```

**Pros:**
- Optimizes for your actual task
- Directly measures what you care about
- Works even if you don't want Gaussian output

**Cons:**
- Computationally expensive
- Need enough data for meaningful CV
- Can overfit to validation set

### Method 3: Grid Search with Visual Inspection ⭐ **Practical**

**Goal:** Try common values and look at histograms

```julia
using Plots

function visualize_transforms(y, lambdas)
    n_lambdas = length(lambdas)
    plots = []
    
    for lambda in lambdas
        if abs(lambda) < 1e-10
            y_trans = log.(y)
            title_str = "λ = 0 (log)"
        else
            y_trans = (y.^lambda .- 1) ./ lambda
            title_str = "λ = $lambda"
        end
        
        p = histogram(y_trans, bins=50, title=title_str, 
                     xlabel="Transformed value", ylabel="Frequency",
                     legend=false)
        push!(plots, p)
    end
    
    plot(plots..., layout=(2, div(n_lambdas+1, 2)), size=(1200, 800))
end

# Visualize
lambdas = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
visualize_transforms(y, lambdas)
```

**Common λ values to try:**
- **λ = -1**: Reciprocal (1/y) - extreme sensitivity at zero
- **λ = -0.5**: Inverse sqrt - very sensitive at zero
- **λ = 0**: Log transform
- **λ = 0.5**: Square root - common for count data
- **λ = 1**: Identity (no transform)
- **λ = 2**: Square - compresses small values
- **λ = 3**: Cube - extreme compression of small values

**Pros:**
- Simple and intuitive
- Can see effect immediately
- Easy to understand what each λ does

**Cons:**
- Subjective
- Not automated
- May miss optimal value between grid points

### Method 4: Domain Knowledge 🎯 **Often overlooked**

**Common guidelines:**

| Data Type | Typical λ | Reasoning |
|-----------|-----------|-----------|
| Count data (e.g., number of events) | 0.5 (sqrt) | Variance often proportional to mean |
| Right-skewed continuous | 0 to 0.5 | Compress large values |
| Percentage/proportion data | Use logit instead | Box-Cox not ideal |
| Income, prices (multiplicative) | 0 (log) | Multiplicative relationships |
| Already normalized | 1 (identity) | No transform needed |
| Heavy tail on small values | 2 to 3 | Compress small values |

**For your data (237k near zero, 3k large):**
- λ = 2 or λ = 3 makes sense if you want to cluster near-zero values
- λ = 0 (log) if you want traditional log transform of large values
- **Don't use λ < 1** - will make sensitivity at zero worse!

### Method 5: Profile Likelihood (Statistical Rigor) ⭐⭐⭐ **Most rigorous**

**Goal:** Find λ and confidence interval using likelihood ratio

```julia
using Optim, Distributions, Statistics

function profile_likelihood(y; alpha=0.05)
    # Find MLE
    lambda_mle = box_cox_mle(y)
    
    # Compute log-likelihood at MLE
    function loglik(lambda)
        if abs(lambda) < 1e-10
            y_trans = log.(y)
        else
            y_trans = (y.^lambda .- 1) ./ lambda
        end
        n = length(y)
        sigma2 = var(y_trans)
        return -n/2 * log(sigma2) + (lambda - 1) * sum(log.(y))
    end
    
    ll_mle = loglik(lambda_mle)
    
    # Find confidence interval
    # Using likelihood ratio test: 2(ll_mle - ll(λ)) ~ χ²(1)
    chi2_critical = quantile(Chisq(1), 1 - alpha)
    
    function ll_diff(lambda)
        return abs(2 * (ll_mle - loglik(lambda)) - chi2_critical)
    end
    
    # Find lower bound
    lambda_lower = optimize(ll_diff, -2.0, lambda_mle).minimizer
    # Find upper bound
    lambda_upper = optimize(ll_diff, lambda_mle, 3.0).minimizer
    
    return (mle=lambda_mle, ci=(lambda_lower, lambda_upper))
end

result = profile_likelihood(y)
println("MLE: λ = $(result.mle)")
println("95% CI: [$(result.ci[1]), $(result.ci[2])]")
```

**Pros:**
- Gives confidence intervals
- Statistically principled
- Can test if specific λ values (like 0, 0.5, 1) are plausible

**Cons:**
- Assumes Gaussian goal
- More complex
- May be overkill for ML applications

---

### Practical Recommendation for Your Data

**Step 1:** Start with domain knowledge
- You have 237k near-zero, want them clustered → try λ = 2 or λ = 3

**Step 2:** Quick visual check
```julia
lambdas = [0.0, 1.0, 2.0, 3.0]
visualize_transforms(y, lambdas)
```

**Step 3:** If building ML model, use cross-validation
- Try λ ∈ [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
- Pick λ that gives best validation performance

**Step 4:** (Optional) Compute MLE for comparison
- See if MLE agrees with your choice
- If MLE gives λ < 1, be skeptical (it's optimizing for Gaussianity, not your goal)

**Don't overthink it:** For your specific case, λ = 2 is a very reasonable starting point based on the sensitivity analysis!

---

## Proof of Correctness ([-1, 1] version)

Starting from forward transform:
$$y_{\text{norm}} = 2 \cdot \frac{\log(y) - \log(y_{\min})}{\log(y_{\max}) - \log(y_{\min})} - 1$$

1. Add 1: $y_{\text{norm}} + 1 = 2 \cdot \frac{\log(y) - \log(y_{\min})}{\log(y_{\max}) - \log(y_{\min})}$

2. Divide by 2: $\frac{y_{\text{norm}} + 1}{2} = \frac{\log(y) - \log(y_{\min})}{\log(y_{\max}) - \log(y_{\min})}$

3. Multiply by $(\log(y_{\max}) - \log(y_{\min}))$:
   $$\frac{y_{\text{norm}} + 1}{2} \cdot (\log(y_{\max}) - \log(y_{\min})) = \log(y) - \log(y_{\min})$$

4. Add $\log(y_{\min})$:
   $$\frac{y_{\text{norm}} + 1}{2} \cdot (\log(y_{\max}) - \log(y_{\min})) + \log(y_{\min}) = \log(y)$$

5. Exponentiate:
   $$y = \exp\left(\frac{y_{\text{norm}} + 1}{2} \cdot (\log(y_{\max}) - \log(y_{\min})) + \log(y_{\min})\right)$$

∎ **Verified:** The inverse correctly recovers the original value.