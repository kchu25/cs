@def title = "Understanding ε² and Logarithmic Sample Complexity"
@def published = "9 December 2025"
@def tags = ["machine-learning", "concentration-inequalities"]

# Understanding ε² and Logarithmic Sample Complexity

Hey! So you're looking at equations 8, 9, and 10 from that concentration inequalities article and wondering "why is ε² showing up?" and "why does sample complexity only need logarithmic samples instead of checking all N items?" Great questions! Let me break this down conversationally.

## The Big Picture First

The key insight is that we're dealing with **two different types of questions**:

1. **Finding at least one good item** (simple question) → needs $O(1/\varepsilon)$ samples
2. **Getting accurate frequency estimates** (harder question) → needs $O(1/\varepsilon^2)$ samples

The ε² shows up in the second case, and it's deeply connected to variance. Let me explain why.

---

## Why Does ε² Appear? The Variance Story

### The Setup
Imagine you're flipping a biased coin with probability $\varepsilon$ of landing heads (a "good" item). You flip it $n$ times and count how many heads you get. Call this count $X$.

**What we know:**
- Expected value: $E[X] = n\varepsilon$
- Variance: $\text{Var}(X) = n\varepsilon(1-\varepsilon)$

### The Two Types of Questions

**Question 1: Did we get at least one head?**

This is asking $P(X = 0) \leq \delta$. Using the simple approximation:

$$P(X = 0) = (1-\varepsilon)^n \approx e^{-n\varepsilon}$$

Setting this equal to $\delta$ and solving:

$$e^{-n\varepsilon} = \delta$$
$$n\varepsilon = \ln(1/\delta)$$
$$n = \frac{\ln(1/\delta)}{\varepsilon}$$

**Notice:** Only $\varepsilon$ appears, not $\varepsilon^2$! This is because we're just asking "yes or no" — did we find anything at all?

**Question 2: Is our sample frequency close to the true frequency?**

Now we want the **empirical frequency** $\hat{\varepsilon} = X/n$ to be close to the true frequency $\varepsilon$. Specifically, we want:

$$P\left(|\hat{\varepsilon} - \varepsilon| > \text{tolerance}\right) \leq \delta$$

This is where Chernoff/Hoeffding bounds come in. Hoeffding's inequality says:

$$P\left(|\hat{\varepsilon} - \varepsilon| > t\right) \leq 2e^{-2nt^2}$$

**The key thing:** That $t^2$ in the exponent! If we want $|\hat{\varepsilon} - \varepsilon| \leq t$ with probability $\geq 1-\delta$, we need:

$$2e^{-2nt^2} \leq \delta$$
$$n \geq \frac{\ln(2/\delta)}{2t^2}$$

If we set our tolerance $t = \varepsilon$ (we want to detect deviations on the order of $\varepsilon$), we get:

$$n \geq \frac{\ln(2/\delta)}{2\varepsilon^2}$$

**BAM!** There's your $\varepsilon^2$ in the denominator.

### Why the Difference?

The fundamental reason is **concentration around the mean**.

- **Finding one**: We only care if $X \geq 1$. The mean is $n\varepsilon$, so if $n\varepsilon$ is reasonably large (say, $> 5$), we're very likely to get at least one. This only requires $n = O(1/\varepsilon)$.

- **Estimating frequency**: We need $X/n$ to be close to $\varepsilon$. The **standard deviation** of $X/n$ is roughly $\sqrt{\varepsilon(1-\varepsilon)/n} \approx \sqrt{\varepsilon/n}$. To make this smaller than $\varepsilon$ (so our estimate is accurate), we need:

$$\sqrt{\frac{\varepsilon}{n}} \lesssim \varepsilon$$
$$\frac{\varepsilon}{n} \lesssim \varepsilon^2$$
$$n \gtrsim \frac{1}{\varepsilon}$$

Wait, that gives $1/\varepsilon$, not $1/\varepsilon^2$! What happened?

The subtlety: To **guarantee** (with high probability) that our deviation is at most $\varepsilon$, we need to go several standard deviations out. Hoeffding tells us the exponential tail decay, and when you work through the math with $\delta$ confidence, you get the $1/\varepsilon^2$ dependence.

Think of it this way:
- Standard deviation of $\hat{\varepsilon}$ is $\Theta(\sqrt{\varepsilon/n})$
- We want deviation $\leq \varepsilon$ with high probability
- Setting $k\sqrt{\varepsilon/n} = \varepsilon$ for some constant $k$ (related to $\ln(1/\delta)$)
- This gives $n = k^2/\varepsilon$, which is $O(1/\varepsilon)$ if $k$ is constant
- But to get *exponentially good* concentration (failure probability $\delta$), the constant becomes $\sqrt{\ln(1/\delta)}$, and squaring it brings in the extra $\ln(1/\delta)$ factor we see in the bound

The cleaner way to see it: Hoeffding bounds **squared deviations**, which naturally brings in $\varepsilon^2$.

---

## Why Logarithmic in N? The Magic of Independence

Okay, now the big question: why don't we need to check all $N$ configurations? Why is $O(\log N)$ enough?

### The Intuition

Imagine you have $N = 1{,}000{,}000$ configurations, and 1% of them are "good" ($\varepsilon = 0.01$). 

**Naive thought:** "I need to check all 1 million to find the good ones."

**Actual truth:** "If I sample randomly, each sample has a 1% chance of hitting a good one. After just a few hundred samples, I'm basically guaranteed to have found at least one."

The reason is **independence**. Each sample is like rolling a 100-sided die and hoping for a specific number. The probability you *miss* after $n$ rolls is:

$$(0.99)^n$$

This decays **exponentially fast** in $n$:
- After 100 samples: $(0.99)^{100} \approx 0.366$ (still 37% chance of missing!)
- After 300 samples: $(0.99)^{300} \approx 0.049$ (5% chance of missing)
- After 500 samples: $(0.99)^{500} \approx 0.0066$ (< 1% chance of missing)

So with just 300-500 samples, we're very confident of finding at least one, **regardless of the fact that $N = 1{,}000{,}000$**.

### The Math

For "finding at least one good item":

$$n \geq \frac{\ln(1/\delta)}{\varepsilon}$$

**Notice:** $N$ doesn't appear! This bound works for *any* $N$, as long as the fraction of good items is $\varepsilon$.

For "finding all good items" (coupon collector problem), we get:

$$n \geq \frac{1}{\varepsilon}\left[\ln(N) + \ln(1/\delta)\right]$$

**Now $N$ appears**, but only inside a $\log$! This is the magic.

### Why Logarithmic?

The reason $\log N$ appears (when finding all items) is because of the coupon collector problem. To collect all $K$ coupons when sampling uniformly from $K$ items, you need roughly:

$$K \cdot H_K \approx K \ln K$$

samples, where $H_K$ is the $K$-th harmonic number.

In our case, there are $K = \varepsilon N$ good items out of $N$ total. Adjusting for the density:

$$n \approx \frac{N}{K} \cdot K \ln K = \frac{N}{\varepsilon N} \cdot \varepsilon N \ln(\varepsilon N) = \frac{1}{\varepsilon} \ln(\varepsilon N)$$

Simplifying:

$$n \approx \frac{1}{\varepsilon}\left[\ln(\varepsilon) + \ln(N)\right] \approx \frac{\ln(N)}{\varepsilon}$$

(The $\ln(\varepsilon)$ term is typically absorbed into the constant.)

### The Crossover Point

Sampling beats exhaustive search when:

$$\frac{\ln N}{\varepsilon} < N$$

This is true when:

$$\ln N < \varepsilon N$$

For small $\varepsilon$ (say 0.01), this is satisfied for pretty much any reasonable $N$. For example:
- $N = 10^6$: $\ln(10^6) \approx 14$, but $\varepsilon N = 10{,}000$. Sampling wins by 700×!
- $N = 10^9$: $\ln(10^9) \approx 21$, but $\varepsilon N = 10{,}000{,}000$. Sampling wins by 500,000×!

The intuition: **Logarithms grow incredibly slowly**. Even if $N$ is astronomical (billions, trillions), $\ln N$ is only 20-40. This is why sampling is so powerful.

---

## Connecting to Pattern 3: False Positives

Now, in **Pattern 3** from the article (avoiding false positives), we get:

$$n \geq \frac{\ln(N/\delta)}{2\varepsilon^2}$$

Here, **both** $N$ and $\varepsilon^2$ appear! Why?

The reason is **union bound**. We're not just asking "did I find a good item?" We're asking "did *any* of the $N$ items incorrectly appear good?"

- For a single item, Hoeffding gives us: $P[\text{false positive}] \leq 2e^{-2n\varepsilon^2}$
- Union bound over all $N$ items: $P[\text{any false positive}] \leq N \cdot 2e^{-2n\varepsilon^2}$
- Setting this $\leq \delta$ and solving for $n$:

$$N \cdot 2e^{-2n\varepsilon^2} \leq \delta$$
$$e^{-2n\varepsilon^2} \leq \frac{\delta}{2N}$$
$$2n\varepsilon^2 \geq \ln\left(\frac{2N}{\delta}\right)$$
$$n \geq \frac{\ln(2N/\delta)}{2\varepsilon^2}$$

So now:
- The $\varepsilon^2$ comes from Hoeffding (we're estimating frequencies)
- The $\ln N$ comes from the union bound (we're testing $N$ hypotheses)

This is the **most conservative** bound because it's controlling false positives across *all* $N$ items. In practice, if you only care about finding true positives (and can tolerate some false positives), you can use the simpler $O(1/\varepsilon)$ bound.

---

## Summary: The Two Key Insights

### 1. Why $\varepsilon^2$?

- $\varepsilon$ in denominator: For "find at least one" questions (just need $X \geq 1$)
- $\varepsilon^2$ in denominator: For "estimate frequencies accurately" questions (need $|\hat{\varepsilon} - \varepsilon|$ small)
- The squared term comes from concentration inequalities (Hoeffding) that bound deviations from the mean. These inherently deal with **variance**, which scales like $\varepsilon/n$, so to make variance-scaled deviations small, you need $n \propto 1/\varepsilon^2$.

### 2. Why $\log N$ instead of $N$?

- **Independence is magic**: Each sample is an independent chance to find something good
- Probability of missing after $n$ samples: $(1-\varepsilon)^n \approx e^{-n\varepsilon}$
- This decays **exponentially** in $n$, so you only need $n = O(\ln(1/\delta)/\varepsilon)$ to drive failure probability below $\delta$
- For finding *all* items or controlling false positives, you pay an extra $\log N$ factor (from coupon collector or union bound), but this is still **way** better than linear $O(N)$

The fundamental reason sampling works: **Exponential concentration beats polynomial growth**. Even though the space is huge ($N$ can be millions or billions), the exponential decay of $(1-\varepsilon)^n$ means you only need logarithmically many samples to be confident.

---

Hope this clarifies things! The key is understanding that different questions require different sample complexities, and the $\varepsilon^2$ appears when you need strong concentration (accurate estimates), while $\log N$ appears because exponential probabilities grow incredibly fast.