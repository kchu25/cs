@def title = "Understanding "Frequency" in ε² and Logarithmic Sample Complexity"
@def published = "8 December 2025"
@def tags = ["machine-learning", "concentration-inequalities"]

# Understanding "Frequency" in ε² and Logarithmic Sample Complexity

You're essentially correct, but let me clarify the nuances.

## The Two Types of Frequency

In this context, there are **two related but distinct concepts**:

### 1. **True Frequency** ($\varepsilon$)
This is the **proportion in the underlying population/distribution** that you're sampling from.

- For example: "1% of all configurations are good" means $\varepsilon = 0.01$
- This is a **fixed, unknown parameter** of the population
- It's what we're trying to learn about

### 2. **Empirical Frequency** ($\hat{\varepsilon}$)
This is the **proportion in your sample** that you actually observed.

- Defined as $\hat{\varepsilon} = \frac{X}{n}$ where $X$ is the count of good items in your $n$ samples
- This is a **random variable** that varies from sample to sample
- It's our **estimate** of the true frequency $\varepsilon$

## The Key Distinction in the Article

The article discusses two different problems:

### Problem 1: Finding at least one good item
- **Question**: "Did I find anything?"
- **Sample complexity**: $n = O(1/\varepsilon)$
- **What matters**: Only the true frequency $\varepsilon$ (how dense good items are)
- **Why**: We just need $P(X \geq 1)$ to be high, which requires $n\varepsilon \gtrsim \ln(1/\delta)$

### Problem 2: Estimating the frequency accurately
- **Question**: "How close is $\hat{\varepsilon}$ to $\varepsilon$?"
- **Sample complexity**: $n = O(1/\varepsilon^2)$
- **What matters**: The deviation $|\hat{\varepsilon} - \varepsilon|$
- **Why**: We need concentration around the mean, which requires controlling variance

## The Nuance You're Asking About

When the article says "frequency" without qualification, it typically means:

- **The true frequency $\varepsilon$** of the underlying population

However, the **whole point** of sampling is to estimate this true frequency using the empirical frequency $\hat{\varepsilon}$ from our sample.

### Example to Illustrate

Suppose:
- Population: 1 million configurations, 10,000 are "good" → true frequency $\varepsilon = 0.01$
- You sample $n = 500$ configurations randomly
- You observe $X = 7$ good ones → empirical frequency $\hat{\varepsilon} = 7/500 = 0.014$

The concentration inequalities tell you:
- **With high probability**, $\hat{\varepsilon}$ will be close to $\varepsilon$
- The sample size needed depends on **both** your desired accuracy and the true $\varepsilon$

## The $\varepsilon^2$ Factor Explained

The $\varepsilon^2$ appears because:

$$\text{Var}(\hat{\varepsilon}) = \text{Var}(X/n) = \frac{1}{n^2} \cdot \text{Var}(X) = \frac{1}{n^2} \cdot n\varepsilon(1-\varepsilon) = \frac{\varepsilon(1-\varepsilon)}{n}$$

The standard deviation is $\sqrt{\varepsilon/n}$ (roughly). To make the standard deviation smaller than your tolerance $\varepsilon$ itself:

$$\sqrt{\frac{\varepsilon}{n}} \lesssim \varepsilon \implies n \gtrsim \frac{1}{\varepsilon}$$

But for **exponential concentration** (high confidence via Hoeffding), you get:

$$n \geq \frac{\ln(2/\delta)}{2\varepsilon^2}$$

## Summary

- **"Frequency" usually refers to the true population parameter $\varepsilon$**
- But we estimate it using the **empirical frequency** $\hat{\varepsilon}$ from our sample
- The relationship between them is governed by concentration inequalities
- The sample complexity depends on **how accurately** you need $\hat{\varepsilon}$ to approximate $\varepsilon$

So yes, you're right that it refers to frequency in the population, but the entire framework is about relating the sampled (empirical) frequency to the true (population) frequency!