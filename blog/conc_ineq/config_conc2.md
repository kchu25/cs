@def title = "Uniform Sampling with Concentration Bounds for Configuration Enumeration - II"
@def published = "7 December 2025"
@def tags = ["machine-learning", "concentration-inequalities"]

# Uniform Sampling with Concentration Bounds for Configuration Enumeration - II 
## Clarified Version

## Executive Summary

A principled approach to scaling motif discovery when α (code image sparsity) is large:

* **No heuristics**: Pure uniform random sampling
* **Theoretical guarantees**: Concentration inequalities bound error
* **Unbiased**: Every configuration has equal probability
* **Scalable**: O(log N) sample complexity instead of O(N) enumeration
* **Simple**: Just set desired error ε and confidence δ

---

## The Problem

### Current Bottleneck

Given code image $Z$ with $m \leq \alpha$ non-zero entries:
* Need to evaluate all $q$-component configurations
* Total configurations: $N = \binom{m}{q}$
* **Exhaustive enumeration becomes intractable** when $\alpha$ is large

**Examples:**
* $\alpha = 32$: $\binom{32}{3} = 4{,}960$ ✓ (feasible)
* $\alpha = 100$: $\binom{100}{3} = 161{,}700$ (expensive)
* $\alpha = 200$: $\binom{200}{3} = 1{,}313{,}400$ (prohibitive)
* $\alpha = 500$: $\binom{500}{3} = 20{,}708{,}500$ (impossible)

---

## Key Clarification: What Are We Sampling?

**Important conceptual point**: We're not sampling configurations that appear with different frequencies. Instead:

1. There are $N = \binom{m}{q}$ total **possible** configurations
2. Each configuration is equally likely when sampled uniformly
3. Each configuration gets **tested** (e.g., with Fisher exact test) to see if it's biologically significant
4. We want to find **all significant configurations** without testing all $N$ of them

Think of it this way:
* **Space to explore**: $N$ possible configurations (like $N$ lottery tickets)
* **Goal**: Find all "winning tickets" (biologically significant patterns)
* **Unknown**: Which configurations will pass our significance test
* **Challenge**: We can't test all $N$ tickets when $N$ is huge

---

## The Solution: Uniform Random Sampling

### Core Principle

**Sample configurations uniformly at random**, then test each sampled configuration. Use **concentration inequalities** to guarantee we find important patterns.

### What We're Actually Guaranteeing

Let's say a configuration is **"significant"** if it passes our Fisher exact test (or other biological validation).

**Key insight**: We don't know which configurations are significant until we test them. But if there are at least K significant configurations among the N total (where K/N ≥ ε), we can guarantee finding them with high probability.

---

## Theoretical Guarantees (Corrected)

### The Actual Question: Will We Find Rare Needles in a Haystack?

Suppose there are $K$ "significant" configurations hidden among $N$ total configurations, where:
* $K/N \geq \varepsilon$ (at least $\varepsilon$ fraction are significant)
* We sample $n$ configurations uniformly at random
* We want to find **at least one** significant configuration with probability $\geq 1-\delta$

**Theorem (Coupon Collector Style):**

If there are at least $K \geq \varepsilon N$ significant configurations, the probability we find **at least one** after $n$ samples is:

$\mathbb{P}(\text{find} \geq 1 \text{ significant}) = 1 - \left(1 - \frac{K}{N}\right)^n \geq 1 - (1 - \varepsilon)^n$

Using the approximation $(1-\varepsilon)^n \approx e^{-n\varepsilon}$:

$\mathbb{P}(\text{miss all significant configs}) \leq e^{-n\varepsilon}$

Setting $e^{-n\varepsilon} = \delta$:

$n \geq \frac{\ln(1/\delta)}{\varepsilon}$

**More conservatively (using Chernoff bound):**

$n \geq \frac{2\ln(2/\delta)}{\varepsilon^2}$

**Interpretation**: If at least $\varepsilon$ fraction of configurations are significant, we need only $O(1/\varepsilon^2)$ samples to find one, **regardless of $N$**.

### Example

* Suppose 1% of all configurations are biologically significant ($\varepsilon = 0.01$)
* Want 99% confidence ($\delta = 0.01$)
* Need $n \geq 2\ln(200) / 0.01^2 \approx 106{,}000$ samples
* **This is the same whether $N = 1$ million or $N = 1$ billion!**

---

## Why This Works: The Birthday Paradox in Reverse

This is related to the **coupon collector problem**:

* **Classic coupon collector**: "How many samples to collect ALL coupons?" → $O(N \log N)$
* **Our version**: "How many samples to collect AT LEAST ONE coupon from the 'good' pile?" → $O(1/\varepsilon)$

If $\varepsilon$ fraction of configurations are significant, each sample has probability $\varepsilon$ of being significant. After $n$ samples, probability of finding none is $(1-\varepsilon)^n$, which decays exponentially.

---

## Multiple Significant Configurations

### Finding ALL Significant Configurations

If we want to find **all** significant configurations (not just one), we need more samples.

**Theorem**: To find all $K$ significant configurations with probability $\geq 1-\delta$, we need:

$n \geq \frac{N}{K} \cdot \left[\ln(K) + \ln(1/\delta)\right]$

Or in terms of $\varepsilon = K/N$:

$n \geq \frac{1}{\varepsilon} \cdot \left[\ln(\varepsilon N) + \ln(1/\delta)\right]$

**Now $N$ appears!** Because finding ALL needles requires searching through more of the haystack as the haystack grows.

### The Tradeoff

**Finding ONE significant pattern**: $O(1/\varepsilon^2)$ samples (independent of $N$)

**Finding ALL significant patterns**: $O((1/\varepsilon)\cdot\log(N))$ samples (logarithmic in $N$)

Both are vastly better than $O(N)$ exhaustive enumeration!

---

## Practical Algorithm

```python
def sample_configurations(Z, q, n_samples):
    """
    Uniformly sample configurations and test for significance.
    
    Parameters:
    -----------
    Z : Matrix
        Code image with non-zero sparse code entries
    q : int
        Configuration size (typically 3)
    n_samples : int
        Number of configurations to sample
    
    Returns:
    --------
    significant_configs : list
        Configurations that pass significance testing
    """
    # Get non-zero positions
    nonzero_positions = get_nonzero(Z)
    m = len(nonzero_positions)
    N = comb(m, q)
    
    print(f"Total possible configurations: {N:,}")
    print(f"Sampling {n_samples:,} configurations ({100*n_samples/N:.2f}%)")
    
    tested_configs = set()
    significant_configs = []
    
    for _ in range(n_samples):
        # Sample q positions uniformly at random (without replacement)
        indices = random.sample(range(m), q)
        config = tuple(sorted(nonzero_positions[i] for i in indices))
        
        # Avoid testing the same configuration twice
        if config in tested_configs:
            continue
        tested_configs.add(config)
        
        # Test for biological significance
        p_value = fisher_exact_test(config, test_set)
        if p_value < 1e-6:  # Significance threshold
            significant_configs.append(config)
    
    return significant_configs
```

---

## Sample Size Recommendations

### Conservative Strategy (Find ONE significant pattern)

If you expect at least $\varepsilon$ fraction of configurations to be significant:

$n = \frac{2\ln(2/\delta)}{\varepsilon^2}$

Example values:

| $\varepsilon$ (expected fraction significant) | $\delta$ (failure prob) | $n$ (samples needed) |
|-----------------------------------|------------------|-------------------|
| 0.01 (1%)                        | 0.01            | ~106,000          |
| 0.05 (5%)                        | 0.01            | ~4,300            |
| 0.10 (10%)                       | 0.01            | ~1,100            |

### Aggressive Strategy (Find MOST significant patterns)

If you want high coverage of all significant patterns:

$n = \frac{1}{\varepsilon} \cdot \left[\ln(N) + \ln(1/\delta)\right]$

| $\alpha$ (sparsity) | $N = \binom{\alpha}{3}$ | $\varepsilon = 0.01$ | $n$ (samples) | Speedup |
|--------------|------------|----------|-------------|---------|
| 100          | 161,700    | 0.01     | ~1,500      | 108×    |
| 200          | 1,313,400  | 0.01     | ~1,600      | 821×    |
| 500          | 20,708,500 | 0.01     | ~1,900      | 10,900× |

---

## Are These Bounds Too Pessimistic?

**Short answer: Yes!** The theoretical bounds are quite conservative. Here's why and what to do about it.

### Why the Bounds Are Pessimistic

#### 1. Worst-Case vs. Average-Case

The formula $n \geq \frac{2\ln(2/\delta)}{\varepsilon^2}$ is a **worst-case guarantee** that works even in unlucky scenarios.

**Expected number of samples** to find one significant configuration:
$\mathbb{E}[n] = \frac{1}{\varepsilon}$

**Guaranteed with high confidence:**
$n = \frac{2\ln(2/\delta)}{\varepsilon^2}$

**Example** ($\varepsilon = 0.01$, $\delta = 0.01$):
- **Expected (average case)**: $1/0.01 = 100$ samples
- **Guaranteed (worst case)**: $106,000$ samples

That's a **1,000× difference**!

#### 2. Actual Success Probabilities

Let's compute the **actual** probability of success for $\varepsilon = 0.01$:

$\mathbb{P}(\text{find} \geq 1) = 1 - (1-\varepsilon)^n$

| Samples $n$ | Actual P(success) | Bound guarantees? | Status |
|-------------|-------------------|-------------------|--------|
| 100 | 63.4% | ❌ No | Expected value |
| 300 | 95.0% | ❌ No | Pretty good! |
| 500 | 99.3% | ❌ No | Excellent! |
| 1,000 | 99.996% | ❌ No | Nearly certain |
| 106,000 | ~100% | ✅ Yes | Theoretical bound |

**In practice**: 500-1,000 samples would work great, but the theoretical bound says you need 106,000!

#### 3. The Concentration Bound Is Loose

The Chernoff bound gives an **upper bound** on failure probability:
$\mathbb{P}(\text{miss all}) \leq e^{-n\varepsilon}$

The **actual** probability is:
$\mathbb{P}(\text{miss all}) = (1-\varepsilon)^n$

For small $\varepsilon$, we use the approximation $(1-\varepsilon)^n \approx e^{-n\varepsilon}$, which adds conservatism.

### When Are Pessimistic Bounds Useful?

The conservative theoretical bounds are valuable when:

1. **You absolutely cannot afford to fail** (high-stakes applications)
2. **You're doing many independent searches** (need union bound over experiments)
3. **You don't know $\varepsilon$ in advance** and want to be safe
4. **You're proving theoretical guarantees** in a paper

### Practical Recommendations

For actual implementation, consider these strategies:

#### Strategy 1: Use Expected Value with Safety Factor

$n_{\text{practical}} = \frac{k}{\varepsilon}$

where $k = 3$ to $5$ (gives ~95-99% success rate for finding at least one pattern)

**Example**: For $\varepsilon = 0.01$:
- **Theoretical bound**: $n = 106,000$
- **Practical approach**: $n = 300$ to $500$
- **Speedup**: 200-350× fewer samples!

#### Strategy 2: Sequential Sampling with Early Stopping

```python
def sample_until_found(Z, q, epsilon, max_samples=None):
    """
    Sample until we find a significant configuration.
    Much more efficient than using the theoretical bound!
    """
    n_expected = int(1 / epsilon)
    max_samples = max_samples or int(10 * n_expected)  # 10× expected
    
    significant_configs = []
    
    for i in range(max_samples):
        config = sample_one_configuration(Z, q)
        
        if is_significant(config):
            significant_configs.append(config)
            print(f"Found significant pattern after {i+1} samples")
            
            # Optional: stop after finding k patterns
            if len(significant_configs) >= desired_count:
                return significant_configs
    
    if not significant_configs:
        print(f"No pattern found after {max_samples} samples")
        print(f"This suggests epsilon < {1/max_samples:.6f}")
    
    return significant_configs
```

#### Strategy 3: Adaptive Doubling

```python
def adaptive_sampling(Z, q, epsilon_guess, confidence_threshold=0.95):
    """
    Start small and gradually increase sample size if needed.
    """
    # Start with 3× expected value
    n_initial = int(3 / epsilon_guess)
    
    # Theoretical bound as upper limit
    n_theoretical = int(2 * np.log(2/0.01) / epsilon_guess**2)
    
    n_samples = n_initial
    all_samples = []
    
    while n_samples <= n_theoretical:
        # Sample this batch
        batch = sample_configurations(Z, q, n_samples - len(all_samples))
        all_samples.extend(batch)
        
        # Check for significant patterns
        significant = [c for c in all_samples if is_significant(c)]
        
        if significant:
            actual_rate = len(significant) / len(all_samples)
            print(f"Found {len(significant)} patterns in {len(all_samples)} samples")
            print(f"Empirical rate: {actual_rate:.4f}")
            return significant
        
        # Double sample size for next iteration
        n_samples = min(n_samples * 2, n_theoretical)
    
    print(f"Exhausted theoretical bound ({n_theoretical} samples)")
    return []
```

### Comparison: Theory vs. Practice

| Scenario | Theoretical $n$ | Practical $n$ | Ratio |
|----------|----------------|---------------|-------|
| $\varepsilon=0.10$ | 1,100 | 30-50 | 22-37× |
| $\varepsilon=0.05$ | 4,300 | 60-100 | 43-72× |
| $\varepsilon=0.01$ | 106,000 | 300-500 | 212-353× |
| $\varepsilon=0.001$ | 10.6M | 3,000-5,000 | 2,120-3,533× |

**Key insight**: The gap between theory and practice **increases** as $\varepsilon$ gets smaller, because the theoretical bound scales as $O(1/\varepsilon^2)$ while practical needs scale as $O(1/\varepsilon)$.

### Real-World Scenario

In the motif discovery context:

- You're typically running this on **many sequences**
- You'll likely find patterns in the **first few hundred samples** per sequence
- The theoretical bound is there for **worst-case protection**
- Most practitioners would use $n \approx 5/\varepsilon$ instead of $n \approx 1/\varepsilon^2$

### Bottom Line

**Use the theoretical bound when:**
- Writing a paper and need provable guarantees
- Cannot afford any failures
- Need to bound worst-case behavior

**Use practical heuristics when:**
- Implementing for real applications
- Can afford to occasionally need more samples
- Want computational efficiency

The theoretical bounds are designed to work in **all cases**, even the unlucky $1\%$ of scenarios. For the other $99\%$ of cases, you can get away with far fewer samples!

---

## When Does Sampling Beat Exhaustive Enumeration?

### Crossover Analysis

Exhaustive enumeration costs: $N = \binom{m}{q}$

Sampling costs: $n \approx (1/\varepsilon)\cdot\ln(N)$ (for finding most patterns)

**Crossover point**: When $(1/\varepsilon)\cdot\ln(N) < N$

For $\varepsilon = 0.01$, $q = 3$:
* $\alpha \approx 60$: Sampling and exhaustive are comparable
* $\alpha > 60$: Sampling is better
* $\alpha < 60$: Exhaustive is better

```python
def choose_method(alpha, q, epsilon=0.01):
    """Decide whether to use sampling or exhaustive enumeration."""
    N = comb(alpha, q)
    n_sample = int((1/epsilon) * (np.log(N) + 5))  # +5 for safety
    
    if n_sample < 0.5 * N:
        return "sampling", n_sample
    else:
        return "exhaustive", N
```

---

## Multiple Sequences: Finding Cross-Dataset Patterns

When working with multiple sequences, the question changes:

**Goal**: Find configuration patterns that appear as significant in **many** sequences (not just one).

```python
def find_cross_sequence_patterns(all_Z, q, epsilon=0.01, delta=0.01):
    """
    Find patterns that are significant across multiple sequences.
    """
    N_sequences = len(all_Z)
    pattern_to_sequences = defaultdict(list)
    
    # For each sequence, sample and test configurations
    for seq_idx, Z in enumerate(all_Z):
        m = np.sum(Z != 0)
        N = comb(m, q)
        
        # Sample size: find at least one significant pattern per sequence
        n_samples = int(2 * np.log(2*N_sequences/delta) / epsilon**2)
        
        configs = sample_configurations(Z, q, n_samples)
        
        for config in configs:
            pattern_to_sequences[config].append(seq_idx)
    
    # Filter: keep patterns appearing in many sequences
    min_sequences = int(0.05 * N_sequences)  # Appear in ≥5% of sequences
    frequent_patterns = {
        pattern: seqs 
        for pattern, seqs in pattern_to_sequences.items()
        if len(seqs) >= min_sequences
    }
    
    return frequent_patterns
```

---

## Key Takeaways

1. **We're sampling from a uniform distribution** over $N$ possible configurations
2. **Each configuration is equally likely** in our sample
3. **We test each sampled configuration** for biological significance
4. **The guarantee**: If $\geq\varepsilon$ fraction are significant, we'll find at least one with high probability using $O(1/\varepsilon^2)$ samples
5. **The speedup**: $O(\log N)$ samples instead of $O(N)$ exhaustive search when seeking good coverage
6. **The beauty**: Sample complexity depends on the **density of significant patterns** ($\varepsilon$), not the **size of the search space** ($N$)

---

## Why the Original Article's "False Positive" Language Was Confusing

The original article talked about "false positives" in a way that suggested configurations have different underlying frequencies. But actually:

* All N configurations are **equally probable** when sampling uniformly
* The "frequency" language makes more sense when thinking about patterns **across multiple sequences**
* Or when thinking about configurations that **pass significance testing** vs those that don't

The correct framing is:
* We have $N$ lottery tickets (configurations)
* Some fraction $\varepsilon$ are "winners" (biologically significant)
* We want to find winners without checking all tickets
* Concentration bounds tell us how many tickets to sample

---

## References

* **Chernoff bound**: Chernoff, H. (1952)
* **Coupon collector problem**: Classic probability problem
* **Union bound**: Basic probability inequality
* **PAC Learning**: Valiant, L. G. (1984)