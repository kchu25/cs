@def title = "Loss Functions for Log-Scale Regression"
@def published = "4 November 2025"
@def tags = ["machine-learning", "data-processing", "loss-function"]

# Loss Functions for Log-Scale Regression

When dealing with data that spans multiple orders of magnitude, choosing the right loss function is crucial. Let's break down your options and when to use each.

## The Standard Approach: Transform Then MSE

**What most people do:**
```julia
y_log = log.(y)
# Fit your model to y_log
log_pred = predict(model, X)
y_pred = exp.(log_pred)  # Transform back
```

**What you're actually optimizing:**
$$L = \mathbb{E}[(\log(\hat{y}) - \log(y))^2] = \text{MSLE (Mean Squared Log Error)}$$

### Pros:
✅ Simple - just transform targets before training
✅ **Multiplicative errors** - treats relative errors equally across scales (explained below)
✅ Works with any model (linear regression, neural nets, trees, etc.)
✅ Automatically handles heteroscedasticity

## Understanding Multiplicative vs Additive Errors

This is crucial to understanding why log transforms work the way they do!

### Additive (Absolute) Errors

**Plain MSE** optimizes additive errors:
$L = (\hat{y} - y)^2$

This cares about the **absolute difference**:

| True Value | Prediction | Absolute Error | Loss |
|------------|-----------|----------------|------|
| 10 | 11 | 1 | 1 |
| 10 | 15 | 5 | 25 |
| 1000 | 1001 | 1 | 1 |
| 1000 | 1005 | 5 | 25 |

Notice: Being off by 5 has the same loss (25) whether you're at scale 10 or scale 1000.

**Problem:** Being off by 5 when predicting 10 is a **50% error** (terrible!), but being off by 5 when predicting 1000 is only a **0.5% error** (great!). Plain MSE treats them the same!

### Multiplicative (Relative) Errors  

**Log transform** optimizes multiplicative errors. Here's why:

$L = (\log(\hat{y}) - \log(y))^2$

Using the logarithm property: $\log(a) - \log(b) = \log(a/b)$

$L = \left[\log\left(\frac{\hat{y}}{y}\right)\right]^2$

This is a **ratio**! The loss depends on $\frac{\hat{y}}{y}$, not $\hat{y} - y$.

**Examples:**

| True Value | Prediction | Ratio $\frac{\hat{y}}{y}$ | Log Error | Loss |
|------------|-----------|----------|-----------|------|
| 10 | 11 | 1.1 | log(1.1) ≈ 0.095 | 0.0091 |
| 10 | 15 | 1.5 | log(1.5) ≈ 0.405 | 0.164 |
| 1000 | 1100 | 1.1 | log(1.1) ≈ 0.095 | 0.0091 |
| 1000 | 1500 | 1.5 | log(1.5) ≈ 0.405 | 0.164 |

**Key insight:** Being off by a **factor of 1.1** (10% too high) has the same loss whether you're at scale 10 or scale 1000!

### Why "Multiplicative"?

Because the model learns to predict **ratios/multipliers** rather than differences:

**Additive thinking:** 
- "I need to add 500 to my base prediction"
- Absolute differences matter

**Multiplicative thinking:**
- "I need to multiply my base prediction by 1.5"  
- Relative ratios matter

### Concrete Example: House Prices

Imagine predicting house prices in two neighborhoods:

**Neighborhood A:** Houses around \$100,000
**Neighborhood B:** Houses around \$1,000,000

**Scenario 1: Additive errors (plain MSE)**

Your model makes these predictions:
- House A: True=\$100k, Pred=\$110k, Error=\$10k, Loss=100M
- House B: True=\$1M, Pred=\$1.01M, Error=\$10k, Loss=100M

Same loss! But House A is **10% off** (bad!) while House B is **1% off** (great!).

The model treats these equally, so it might sacrifice accuracy on cheap houses to get expensive houses within \$10k.

**Scenario 2: Multiplicative errors (log transform)**

Same predictions:
- House A: True=\$100k, Pred=\$110k, Ratio=1.1, Log error=0.095, Loss=0.009
- House B: True=\$1M, Pred=\$1.01M, Ratio=1.01, Log error=0.01, Loss=0.0001

Now House A has **90x more loss** because the percentage error is much worse!

The model learns: "Being 10% off is equally bad whether the house costs \$100k or \$1M."

### Mathematical Relationship

For small errors, we can approximate:

$\log\left(\frac{\hat{y}}{y}\right) \approx \frac{\hat{y} - y}{y}$

So log error ≈ **percentage error**!

That's why:
- MSLE ≈ Mean Squared Percentage Error (for small errors)
- The model optimizes relative accuracy, not absolute accuracy

### When to Use Each

| Domain | Example | Type | Reasoning |
|--------|---------|------|-----------|
| **Economics & Finance** |
| Stock prices | \$50 vs \$5000/share | **Multiplicative** | 10% gain/loss matters equally; doubling is doubling |
| Revenue/Sales | \$1M vs \$100M | **Multiplicative** | Growth rates (%) are what matter, not absolute \$ |
| Salary | \$50k vs \$500k | **Multiplicative** | 20% raise has similar impact at any level |
| GDP | \$1T vs \$20T economy | **Multiplicative** | Growth measured in %, not absolute billions |
| Market cap | \$1B vs \$1T company | **Multiplicative** | Valuation multiples, not absolute differences |
| Interest/Returns | 5% vs 10% return | **Multiplicative** | Compounding makes ratios matter |
| Exchange rates | 1 USD = 100 JPY | **Multiplicative** | 10% currency move significant at any scale |
| Wealth/Assets | \$100k vs \$10M | **Multiplicative** | Lifestyle changes by orders of magnitude |
| **Demographics** |
| Population | 1000 vs 1M people | **Multiplicative** | Growth rates (%), doubling time |
| Population density | 10 vs 10,000 per km² | **Multiplicative** | Urban vs rural by orders of magnitude |
| Life expectancy | 45 vs 85 years | **Additive** | Each year of life equally valuable |
| Age | 5 vs 50 years old | **Additive** | 1 year is 1 year regardless |
| **Science & Nature** |
| Earthquake magnitude | 5.0 vs 8.0 Richter | **Multiplicative** | Logarithmic scale, each +1 is 10x energy |
| pH levels | 3 vs 7 (acidity) | **Multiplicative** | Log scale, each unit is 10x H⁺ concentration |
| Decibels (sound) | 60 vs 90 dB | **Multiplicative** | Log scale, each +10 is 10x intensity |
| Star brightness | Magnitude 1 vs 6 | **Multiplicative** | Log scale, each magnitude is 2.5x |
| Species count | 10 vs 10,000 species | **Multiplicative** | Biodiversity measured in orders of magnitude |
| Viral load | 1000 vs 1M copies/mL | **Multiplicative** | Exponential growth, log-fold changes |
| Gene expression | 10x vs 1000x baseline | **Multiplicative** | Fold-changes (2x, 10x) standard in biology |
| Bacterial growth | 1k vs 1M CFU | **Multiplicative** | Exponential growth dynamics |
| **Web & Tech** |
| Website traffic | 100 vs 100k visits/day | **Multiplicative** | Growth measured in %, orders of magnitude matter |
| User base | 1k vs 1B users | **Multiplicative** | Network effects scale multiplicatively |
| Data storage | 1GB vs 1PB | **Multiplicative** | KB→MB→GB→TB progression |
| Response time | 10ms vs 1000ms | **Multiplicative** | 2x slower feels similar at any scale |
| API rate limits | 100 vs 10k req/sec | **Multiplicative** | Order of magnitude determines capacity tier |
| Follower count | 100 vs 100k followers | **Multiplicative** | Influence grows non-linearly |
| **Business Metrics** |
| Customer count | 10 vs 10,000 | **Multiplicative** | Scaling challenges at each order of magnitude |
| Conversion rate | 1% vs 10% | **Multiplicative** | Doubling conversion = doubling revenue |
| Customer lifetime value | \$100 vs \$10k | **Multiplicative** | Different customer segments by magnitude |
| Churn rate | 1% vs 10% monthly | **Multiplicative** | Compound effects over time |
| **Measurements - Additive** |
| Temperature (Celsius) | 10°C vs 30°C | **Additive** | 5° difference feels similar anywhere (within range) |
| Distance | 10m vs 1000m | **Additive** | 1m is 1m, though sometimes multiplicative for navigation |
| Height/Length | 150cm vs 200cm | **Additive** | 1cm is 1cm regardless of total height |
| Weight (small range) | 50kg vs 80kg | **Additive** | 1kg is 1kg in normal ranges |
| Time duration | 10 sec vs 60 sec | **Additive** | Each second equally valuable |
| Angles/Degrees | 10° vs 90° | **Additive** | Each degree is same angular unit |
| Percentage points | 10% vs 30% | **Additive** | +5 percentage points is +5 points |
| Test scores | 70 vs 90 out of 100 | **Additive** | Each point equally valuable |
| **Sports & Games** |
| Points scored | 10 vs 100 points | **Additive** | Each point counts the same |
| Game score | 3-2 vs 103-102 | **Additive** | Win by 1 is win by 1 |
| Marathon time | 2:30 vs 4:00 hours | **Additive** | Each minute is same effort (roughly) |
| Golf score | 72 vs 85 strokes | **Additive** | Each stroke matters equally |
| **Medicine & Health** |
| Tumor size | 1cm vs 10cm | **Multiplicative** | Doubling time, growth dynamics |
| Blood cell count | 1k vs 10k per μL | **Multiplicative** | Orders of magnitude indicate different conditions |
| Drug dosage | 10mg vs 100mg | **Multiplicative** | Half-life, therapeutic windows scale |
| Heart rate | 60 vs 120 bpm | **Additive** | Each beat per minute similar impact |
| Blood pressure | 120/80 vs 160/100 | **Additive** | Each mmHg similar risk increment |
| Body temperature | 36°C vs 40°C | **Additive** | Each degree equally concerning |
| BMI | 20 vs 30 | **Additive** | Linear scale for health categories |
| **Real Estate & Geography** |
| House price | \$100k vs \$1M | **Multiplicative** | Market dynamics by price tier |
| Square footage | 500 vs 5000 sq ft | **Multiplicative** | Cost per sq ft changes with scale |
| Land area | 0.1 vs 100 acres | **Multiplicative** | Use cases differ by orders of magnitude |
| Altitude/Elevation | 100m vs 5000m | **Additive** | Each meter is same vertical distance |
| Latitude/Longitude | 10° vs 80° | **Additive** | Each degree is same angular distance |
| **Energy & Physics** |
| Energy/Power | 1W vs 1MW | **Multiplicative** | Orders of magnitude: device→building→city |
| Frequency | 10Hz vs 10MHz | **Multiplicative** | Orders of magnitude matter (radio spectrum) |
| Wavelength | 1nm vs 1m | **Multiplicative** | Different physical phenomena at each scale |
| Particle count | 10⁶ vs 10²³ (mole) | **Multiplicative** | Chemistry deals with log scales |
| Half-life | 1 sec vs 1000 years | **Multiplicative** | Decay rates span huge ranges |
| Speed (relative) | 1 m/s vs 1000 m/s | **Additive** | Each m/s is same increment (non-relativistic) |
| Electric charge | 1μC vs 1C | **Additive** | Linear superposition |
| **Computation** |
| Algorithm complexity | O(n) vs O(n²) | **Multiplicative** | Growth rates by factors |
| Memory usage | 1KB vs 1GB | **Multiplicative** | KB→MB→GB→TB tiers |
| CPU cycles | 1k vs 1B | **Multiplicative** | Performance scales |
| Pixels | 100px vs 1000px | **Additive** | Each pixel is same unit |
| **Ambiguous Cases** |
| Income inequality (Gini) | 0.3 vs 0.5 | **Additive** | Scale from 0-1, differences matter linearly |
| Probability | 0.01 vs 0.5 | **Both** | Small probabilities multiplicative (odds ratios), medium/high additive |
| Speed (wide range) | 1 km/h vs 300,000 km/s | **Multiplicative** | When spanning orders of magnitude |
| Weight (wide range) | 1g vs 1000kg | **Multiplicative** | When spanning orders of magnitude |

### Key Patterns to Recognize

**Use Multiplicative (Log Transform) when:**
- ✅ Data spans **multiple orders of magnitude** (10x, 100x, 1000x differences)
- ✅ **Growth rates/percentages** are what matter (10% growth is 10% growth)
- ✅ Physical processes are **exponential** (compound growth, decay)
- ✅ Measurements use **logarithmic scales** (Richter, decibels, pH)
- ✅ **Ratios/multiples** are natural way to think (2x bigger, 10x more users)
- ✅ Phenomena have **threshold effects** at different scales (1k vs 1M users = different product)

**Use Additive (No Transform) when:**
- ✅ Data in **narrow range** or roughly same order of magnitude
- ✅ **Absolute differences** are what matter (each unit equally important)
- ✅ Scale is **human-defined/arbitrary** (test scores, angles, points)
- ✅ **Physical measurements** with consistent precision
- ✅ **Linear relationships** expected
- ✅ **Zero is meaningful** (can't take log of zero)

### Real-World Tip

Ask yourself: **"If I double all the values, does the relationship change?"**

- Stock portfolio: $100k → $200k feels like same % gain = **Multiplicative**
- Temperature: 10°C → 20°C is NOT double the heat = **Additive** (unless Kelvin!)
- Website views: 1k → 2k visitors same as 100k → 200k = **Multiplicative**
- Test score: 50 → 100 is NOT same as 25 → 50 points = **Additive**

### Visual Intuition

**Additive errors:** Think of a number line where each unit of distance is equally important
```
0----10----20----30----40----50
     └─5─┘       └─5─┘
   Same error, same loss
```

**Multiplicative errors:** Think of a logarithmic scale where each doubling/halving is equally important
```
1----2----4----8----16----32
     └×2┘      └×2┘
   Same ratio, same loss
```

On a log scale:
- 1 → 2 is the same "distance" as 10 → 20 or 100 → 200 (all are 2x multipliers)
- Being off by 2x at any scale is equally bad

### Back to Your Original Observation

You said: "log transform makes the model more capable at predicting large values"

Now you can see why! 

**Without log transform:**
- Error of 100 at y=1000 → loss = 10,000
- Error of 10 at y=100 → loss = 100
- Large value errors dominate, but the model might still underfit them to avoid huge losses

**With log transform:**
- Both are 10% errors → same log error ≈ 0.095
- Model learns to be 10% accurate everywhere
- When you transform back, you get good **relative** accuracy at all scales
- Large values benefit because the model isn't scared of their magnitude

But (as you noticed) this also makes it hypersensitive near zero, because:
- Error from 0.01 to 0.02 = 2x multiplier = log(2) ≈ 0.69
- Error from 100 to 200 = 2x multiplier = log(2) ≈ 0.69  
- Same loss, but 0.01 → 0.02 is probably noise while 100 → 200 is important!

That's the whole sensitivity problem we discussed earlier.

### Cons:
❌ **Biased predictions** when transformed back (systematically underestimates)
❌ Requires strictly positive values
❌ Hypersensitive to near-zero values
❌ Model learns to predict **median** not mean

### The Bias Problem

Due to Jensen's inequality, $\mathbb{E}[e^X] > e^{\mathbb{E}[X]}$.

If residuals in log-space have variance $\sigma^2$, you need to correct:
$$\hat{y}_{\text{unbiased}} = \exp(\hat{y}_{\log} + \frac{\sigma^2}{2})$$

**Example:**
```julia
using Statistics

# Calculate residual variance on validation set
residuals = y_log_true .- y_log_pred
sigma_squared = var(residuals)

# Correct bias when transforming back
y_pred = exp.(log_pred .+ sigma_squared / 2)
```

Most people skip this and their predictions are too low!

## Alternative 1: Direct MSLE (No Transform)

**Keep targets in original scale, use custom loss:**
```julia
# Custom loss function
function msle_loss(y_pred, y_true)
    return mean((log.(y_pred) .- log.(y_true)).^2)
end

# For Flux.jl
using Flux
loss(x, y) = Flux.Losses.msle(model(x), y)

# Or manual implementation
function msle_loss(ŷ, y)
    log_diff = log.(ŷ) .- log.(y)
    return mean(log_diff.^2)
end
```

**What you're optimizing:**
$$L = \mathbb{E}[(\log(\hat{y}) - \log(y))^2]$$

Same as before, but no manual transformation needed!

### Pros:
✅ No need to transform targets manually
✅ No inverse transform needed
✅ Can apply bias correction during training
✅ Predictions are directly in original scale

### Cons:
❌ Still has bias issue
❌ Requires custom loss implementation
❌ Gradients can be unstable if $\hat{y}$ gets close to zero
❌ Need to clip predictions: $\hat{y} > \epsilon$ to avoid log(0)

### Gradient Behavior:
$$\frac{\partial L}{\partial \hat{y}} = \frac{2(\log(\hat{y}) - \log(y))}{\hat{y}}$$

Notice that $\frac{1}{\hat{y}}$ term - if prediction is small, gradient explodes! Need to be careful.

## Alternative 2: RMSLE (Root Mean Squared Log Error)

Just the square root of MSLE:
$L = \sqrt{\mathbb{E}[(\log(\hat{y}) - \log(y))^2]}$

```julia
function rmsle_loss(y_pred, y_true)
    return sqrt(mean((log.(y_pred .+ 1) .- log.(y_true .+ 1)).^2))
end
```

Note the `+1` to handle zeros! This makes it technically:
$$L = \sqrt{\mathbb{E}[(\log(\hat{y}+1) - \log(y+1))^2]}$$

### Pros:
✅ Same scale as log(y), easier to interpret
✅ The +1 offset handles zeros
✅ Popular in Kaggle competitions

### Cons:
❌ Still has all the MSLE issues
❌ The +1 is arbitrary
❌ Square root in loss can slow convergence

## Alternative 3: Mean Absolute Log Error (MALE)

Use L1 instead of L2 in log-space:
$L = \mathbb{E}[|\log(\hat{y}) - \log(y)|]$

```julia
function male_loss(y_pred, y_true)
    return mean(abs.(log.(y_pred) .- log.(y_true)))
end
```

### Pros:
✅ More robust to outliers than MSLE
✅ Predicts **median** explicitly (no pretense of predicting mean)
✅ No bias correction needed (for median estimation)
✅ Simpler gradients

### Cons:
❌ If you want mean predictions, this is wrong objective
❌ L1 gradients don't go to zero (can be jumpy)
❌ Still sensitive near zero

### Gradient:
$$\frac{\partial L}{\partial \hat{y}} = \frac{\text{sign}(\log(\hat{y}) - \log(y))}{\hat{y}}$$

Constant magnitude (±1/ŷ), just changes sign. Can be more stable than L2.

## Alternative 4: Weighted MSE in Original Space

**Don't transform at all, just weight the loss based on what you care about:**

### Option A: Optimize Relative/Percentage Errors (weights = 1/y²)
```julia
# Makes relative errors equal across all scales
function relative_mse(y_pred, y_true; epsilon=1e-6)
    weights = 1.0 ./ (y_true .+ epsilon).^2
    return mean(weights .* (y_pred .- y_true).^2)
end

# Or more directly:
function relative_mse(y_pred, y_true; epsilon=1e-6)
    return mean(((y_pred .- y_true) ./ (y_true .+ epsilon)).^2)
end
```

This optimizes **percentage error squared**:
$L = \mathbb{E}\left[\frac{(\hat{y} - y)^2}{y^2}\right] = \mathbb{E}\left[\left(\frac{\hat{y} - y}{y}\right)^2\right]$

**Effect:** 10% error at y=10 has same loss as 10% error at y=1000
- Error of 1 at y=10 → loss = (1/10)² = 0.01
- Error of 100 at y=1000 → loss = (100/1000)² = 0.01

⚠️ **This makes small values MORE important in absolute terms!** A weight of 1/y² means smaller y → larger weight.

### Option B: Prioritize Large Values (weights = yᵖ)
```julia
# Makes large values MORE important
function large_value_mse(y_pred, y_true; power=1, epsilon=1e-6)
    weights = (y_true .+ epsilon).^power
    return mean(weights .* (y_pred .- y_true).^2)
end
```

With `power=1`:
$L = \mathbb{E}[y \cdot (\hat{y} - y)^2]$

**Effect:** Errors on large values dominate the loss
- Error of 1 at y=10 → loss = 10 × 1² = 10
- Error of 1 at y=1000 → loss = 1000 × 1² = 1000

The large value contributes **100x more** to the loss!

With `power=2`, it's even more extreme (10,000x difference).

### Which Weighting Should You Use?

| Your Goal | Use This Weighting | Formula |
|-----------|-------------------|---------|
| **Relative errors matter** (10% at any scale is equally bad) | `weights = 1/y²` | Percentage errors |
| **Large absolute values matter more** | `weights = y` or `y²` | Large value focus |
| **Uniform treatment** | `weights = 1` | Plain MSE |

### Pros:
✅ No transformation needed at all
✅ **No bias issues** - predictions are naturally unbiased
✅ Interpretable: can optimize for relative errors OR large value focus
✅ Works with any model
✅ Direct control over what matters

### Cons:
❌ Can blow up if $y \approx 0$ with 1/y² weighting (need epsilon)
❌ Requires custom loss function
❌ Need to decide on weighting scheme (relative vs absolute focus)

### Relationship to MSLE:
For **relative errors** (weights = 1/y²), small errors satisfy:
$\left(\frac{\hat{y} - y}{y}\right)^2 \approx \left(\frac{\hat{y}}{y} - 1\right)^2 \approx (\log(\hat{y}) - \log(y))^2$

So for small relative errors, relative MSE ≈ MSLE! But:
- Relative MSE has **no bias issues**
- MSLE needs bias correction when transforming back
- For large errors, they diverge

## Alternative 5: Huber Loss in Log Space

Combines L2 (MSLE) for small errors with L1 (MALE) for large errors:
$L = \begin{cases}
\frac{1}{2}(\log(\hat{y}) - \log(y))^2 & \text{if } |\log(\hat{y}) - \log(y)| \leq \delta \\
\delta \cdot (|\log(\hat{y}) - \log(y)| - \frac{\delta}{2}) & \text{otherwise}
\end{cases}$

```julia
function log_huber_loss(y_pred, y_true; delta=1.0)
    log_diff = log.(y_pred) .- log.(y_true)
    abs_diff = abs.(log_diff)
    quadratic = 0.5 .* log_diff.^2
    linear = delta .* (abs_diff .- 0.5 * delta)
    return mean(ifelse.(abs_diff .<= delta, quadratic, linear))
end
```

### Pros:
✅ Robust to outliers (doesn't penalize huge errors as much)
✅ Still smooth around zero (better optimization than L1)
✅ Tunable threshold δ

### Cons:
❌ Another hyperparameter to tune (δ)
❌ Still has bias issues from log transform

## Alternative 6: Quantile Regression

Instead of predicting the mean or median, predict a specific quantile:
$L = \mathbb{E}[\rho_\tau(\log(y) - \log(\hat{y}))]$

where $\rho_\tau(u) = u(\tau - \mathbb{1}_{u < 0})$ is the quantile loss.

```julia
function quantile_log_loss(y_pred, y_true; tau=0.5)
    log_diff = log.(y_true) .- log.(y_pred)
    return mean(ifelse.(log_diff .> 0, 
                        tau .* log_diff,
                        (tau - 1) .* log_diff))
end

# For training multiple quantiles:
function train_quantile_models(X, y, quantiles=[0.25, 0.5, 0.75])
    models = Dict()
    for tau in quantiles
        loss(ŷ, y) = quantile_log_loss(ŷ, y; tau=tau)
        # Train your model with this loss
        models[tau] = trained_model
    end
    return models
end
```

Set $\tau = 0.5$ for median, $\tau = 0.75$ for upper quartile, etc.

### Pros:
✅ Gives you prediction intervals, not just point estimates
✅ Can focus on underestimation risk (τ > 0.5) or overestimation (τ < 0.5)
✅ Very robust

### Cons:
❌ More complex
❌ Need to train separate models for different quantiles
❌ Harder to interpret than mean predictions

## Alternative 7: Weighted Loss in Log Space

You can combine log transforms with custom weighting to control priorities in log-space:

```julia
# Weight errors in log-space by original scale
function weighted_log_mse(y_pred, y_true; weight_power=1, epsilon=1e-6)
    log_pred = log.(y_pred .+ epsilon)
    log_true = log.(y_true .+ epsilon)
    weights = y_true.^weight_power  # Weight by original scale
    return mean(weights .* (log_pred .- log_true).^2)
end
```

### What does this do?

With `weight_power = 1`:
$L = \mathbb{E}[y \cdot (\log(\hat{y}) - \log(y))^2]$

**Effect:** Errors in log-space on large values matter more
- Log error of 0.1 at y=10 → loss = 10 × 0.1² = 0.1
- Log error of 0.1 at y=1000 → loss = 1000 × 0.1² = 100

Same percentage error (since log difference is the same), but the large value contributes 1000x more to the loss!

### Does this make sense?

It depends on what you want:

**Standard MSLE (no weighting):**
- Treats multiplicative errors equally at all scales
- Log error of 0.1 means ~10% relative error whether at y=10 or y=1000
- Model focuses equally on all scales in percentage terms

**Weighted log MSE (weight = y):**
- Combines multiplicative scaling with absolute value importance
- Model cares about percentage errors BUT weighted by magnitude
- A 10% error on 1000 is treated as 100x more important than 10% error on 10

**When to use weighted log loss:**

✅ **Use it when:** You want relative/multiplicative errors, but large values are more important to get right
- Example: Revenue prediction where 10% error on \$1M revenue hurts more than 10% error on \$1k revenue

❌ **Don't use it when:** You truly care about percentage errors equally (then use standard MSLE)

### Comparison Table

| Approach | What it optimizes | Example (error of 1) |
|----------|------------------|---------------------|
| **Plain MSE** | Absolute errors uniformly | y=10: loss=1², y=1000: loss=1² (same) |
| **MSLE (no weight)** | Relative errors uniformly | y=10: log error ≈ 0.095², y=1000: log error ≈ 0.001² |
| **Weighted MSE (w=1/y²)** | Percentage errors | y=10: loss=(1/10)²=0.01, y=1000: loss=(1/1000)²=0.000001 |
| **Weighted MSE (w=y)** | Large values (absolute focus) | y=10: loss=10×1²=10, y=1000: loss=1000×1²=1000 |
| **Weighted log MSE (w=y)** | Large values (relative focus) | y=10: log error × 10, y=1000: log error × 1000 |

### Practical Implementation

```julia
# For Flux.jl or other gradient-based training
function weighted_log_loss(ŷ, y; weight_power=1, epsilon=1e-6)
    log_error = log.(ŷ .+ epsilon) .- log.(y .+ epsilon)
    weights = y.^weight_power
    return mean(weights .* log_error.^2)
end

# Example: prioritize large values moderately
loss(ŷ, y) = weighted_log_loss(ŷ, y; weight_power=0.5)
# weight_power = 0: standard MSLE (no prioritization)
# weight_power = 0.5: moderate prioritization (10x value → 3.16x weight)
# weight_power = 1.0: linear prioritization (10x value → 10x weight)
# weight_power = 2.0: strong prioritization (10x value → 100x weight)
```

### My take:

**Weighted log loss can make sense, but it's getting complicated.** You're mixing two different scaling philosophies:
1. Log transform = multiplicative/relative scaling
2. Weighting by y = absolute value importance

If you care about large values in absolute terms, it's cleaner to just use:
```julia
# Direct: weight by value in original space
large_value_mse(y_pred, y_true; power=1)
```

If you care about relative errors equally, use:
```julia
# Standard log transform (with bias correction)
```

**Weighted log loss is for the specific case where:** "I want multiplicative scaling, but I also want large values to contribute more to the loss than small values in proportion to their magnitude."

## Alternative 8: Poisson Deviance Loss

If your data is count-like (non-negative integers or positive reals), consider:
$L = \mathbb{E}[y \log(y/\hat{y}) + (\hat{y} - y)]$

This is the negative log-likelihood for Poisson distribution.

```julia
function poisson_deviance(y_pred, y_true)
    return mean(y_true .* log.(y_true ./ y_pred) .+ (y_pred .- y_true))
end

# More stable version that handles zeros:
function poisson_deviance_stable(y_pred, y_true; epsilon=1e-10)
    y_pred = max.(y_pred, epsilon)  # Clip predictions
    # When y_true = 0, the y*log(y/ŷ) term is 0
    log_term = ifelse.(y_true .> 0, 
                       y_true .* log.(y_true ./ y_pred),
                       0.0)
    return mean(log_term .+ (y_pred .- y_true))
end
```

### Pros:
✅ Natural for count data
✅ Handles zeros properly
✅ Well-studied statistical properties
✅ Predictions are naturally unbiased

### Cons:
❌ Only appropriate for count/rate data
❌ Assumes Poisson variance structure (variance = mean)

## Comparison Table

| Loss Function | Bias? | Handles Zeros? | Sensitivity Near Zero | Prioritizes Large Values? | Good For |
|--------------|-------|----------------|----------------------|---------------------------|----------|
| **MSLE (transform)** | ⚠️ Yes, needs correction | ❌ No | 🔥 Very high | ❌ No, relative errors | Quick & dirty, multiplicative data |
| **Direct MSLE** | ⚠️ Yes | ❌ No | 🔥 Very high | ❌ No, relative errors | Custom training loop |
| **MALE** | ✅ No (for median) | ❌ No | 🔥 High | ❌ No | Robust outliers, median prediction |
| **Relative MSE (1/y²)** | ✅ No | ⚠️ Need epsilon | 🔥 High (small values matter more!) | ❌ No, percentage errors | **Relative/percentage error optimization** |
| **Large Value MSE (yᵖ)** | ✅ No | ✅ Yes | Low (large values matter more!) | ✅ Yes! | **When large values are what matters** |
| **Huber (log)** | ⚠️ Yes | ❌ No | 🔥 High | ❌ No | Outlier-robust MSLE |
| **Quantile** | ✅ No | ❌ No | 🔥 High | ❌ No | Uncertainty estimates |
| **Poisson** | ✅ No | ✅ Yes | Medium | ⚠️ Somewhat | Count data specifically |

## My Recommendations

### For optimizing relative/percentage errors: Relative MSE
```julia
function relative_mse_loss(y_pred, y_true; epsilon=1e-6)
    weights = 1.0 ./ (y_true .+ epsilon).^2
    return mean(weights .* (y_pred .- y_true).^2)
end

# Or more directly:
function relative_mse_loss(y_pred, y_true; epsilon=1e-6)
    return mean(((y_pred .- y_true) ./ (y_true .+ epsilon)).^2)
end
```

**Why:**
- No transformation headaches
- No bias issues  
- Optimizes relative errors (10% error equally bad at any scale)
- Clean gradients

**⚠️ Important:** This makes small values MORE important in absolute terms! It's optimizing percentage errors, not prioritizing large values.

### For prioritizing large values: Large Value MSE
```julia
function large_value_mse(y_pred, y_true; power=1, epsilon=1e-6)
    weights = (y_true .+ epsilon).^power
    return mean(weights .* (y_pred .- y_true).^2)
end
```

**Why:**
- No transformations
- No bias issues
- Errors on large values contribute much more to loss
- Power parameter controls how much you prioritize large values

**Use when:** You genuinely care more about getting 1000 → 1001 right than 1 → 2.

### If you must use log transform: Apply bias correction
```julia
using Statistics

# During training
y_log = log.(y)
# ... train your model on y_log ...

# At inference
log_pred = predict(model, X)
residuals = y_log_val .- predict(model, X_val)
sigma_squared = var(residuals)
y_pred = exp.(log_pred .+ sigma_squared / 2)  # ← Don't forget this!
```

**Why:** Without correction, you're predicting the geometric mean (median in log space), which systematically underestimates large values.

### For production systems: Quantile regression
```julia
# Train 3 models: 25th, 50th, 75th percentile
quantiles = [0.25, 0.5, 0.75]
models = Dict()

for tau in quantiles
    loss(ŷ, y) = quantile_log_loss(ŷ, y; tau=tau)
    models[tau] = train_model(X, y, loss)
end

# Now you have uncertainty bounds!
y_lower = predict(models[0.25], X_test)
y_median = predict(models[0.5], X_test)
y_upper = predict(models[0.75], X_test)
```

Especially valuable when large predictions have high stakes.

## The Real Question: What Are You Optimizing For?

The "right" loss depends on your actual business/scientific goal:

| Your Goal | Use This Loss | Why |
|-----------|---------------|-----|
| Minimize **relative/percentage errors** equally across all scales | Relative MSE (weights = 1/y²) or MSLE | 10% error at 10 = 10% error at 1000 |
| Minimize **absolute errors** uniformly | Plain MSE (no transform!) | All errors weighted equally |
| **Large values matter more** in absolute terms | Large Value MSE (weights = yᵖ) | Error of 1 at y=1000 matters more than error of 1 at y=10 |
| Predict **median** (robust to outliers) | MALE or quantile (τ=0.5) | Median is more robust |
| Predict **mean** unbiased | Large Value MSE or MSLE with correction | Avoid systematic bias |
| **Underestimation is worse** than overestimation | Quantile (τ > 0.5) or asymmetric loss | Penalize low predictions more |
| **Overestimation is worse** | Quantile (τ < 0.5) or asymmetric loss | Penalize high predictions more |
| Need **prediction intervals** | Quantile regression (multiple τ) | Get uncertainty bounds |
| Data is **counts/rates** | Poisson deviance | Natural for count data |

**Key Insight:** 
- **Relative MSE (1/y²)**: Treats percentage errors equally → small values get MORE weight in absolute terms
- **Large Value MSE (yᵖ)**: Large values get MORE weight → absolute errors on large values dominate
- **Plain MSE**: Uniform weighting → absolute errors treated equally
- **Log transform**: Like relative MSE but with bias issues

**Bottom line:** If you care about predicting large quantities accurately in absolute terms, use **Large Value MSE with power=1 or 2**, NOT relative MSE or log transforms!