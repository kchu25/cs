@def title = "Understanding Landau Notation in Concentration Inequalities"
@def published = "8 December 2025"
@def tags = ["machine-learning", "concentration-inequalities"]

# Understanding Landau Notation in Concentration Inequalities

Hey! So you're reading about concentration inequalities and seeing all these Big-O, little-o notations everywhere. Let me break down what's happening in the context of this article about sample complexity.

## The Big Picture: What's Landau Notation Doing Here?

In this article, Landau notation (Big-O, little-o, Big-Ω, etc.) is used to capture **how things scale** as parameters get large or small. When we say $n = O(1/\varepsilon)$, we're saying "the sample size grows roughly like $1/\varepsilon$" without worrying about exact constants.

Think of it as a **zoom lens for mathematics**—we're stepping back to see the shape of the relationship rather than the exact numerical details.

## The Cast of Characters

### Big-O: $O(\cdot)$ — "Upper Bound" or "At Most This Fast"

**Definition:** $f(n) = O(g(n))$ means there exist constants $C > 0$ and $n_0$ such that:
$$|f(n)| \leq C \cdot g(n) \text{ for all } n \geq n_0$$

**Translation:** "$f$ grows **no faster than** $g$ (up to a constant factor)"

**In the article:**
- "$n \sim O(1/\varepsilon)$" means the sample size grows **at most** proportionally to $1/\varepsilon$
- "$O(1/\varepsilon^2)$" means sample size might need up to (constant times) $1/\varepsilon^2$ samples
- "$O((1/\varepsilon)\log N)$" means sample complexity scales **at most** like $\frac{\log N}{\varepsilon}$

**Why use it?** We don't care about the exact constant (is it $2/\varepsilon^2$ or $5/\varepsilon^2$?), we care that it's **quadratic** in $1/\varepsilon$.

### Little-o: $o(\cdot)$ — "Strictly Smaller"

**Definition:** $f(n) = o(g(n))$ means:
$$\lim_{n \to \infty} \frac{f(n)}{g(n)} = 0$$

**Translation:** "$f$ is **negligible** compared to $g$"

**Example in context:**
- If error is $o(1/n)$, it means the error **vanishes faster** than $1/n$
- If we say $\ln(1-\varepsilon) = -\varepsilon + o(\varepsilon)$, the remaining terms (like $-\varepsilon^2/2$) are **negligible** compared to $-\varepsilon$ when $\varepsilon$ is small

**Not in the article explicitly, but helpful to know:** The difference between $O$ and $o$:
- $2n = O(n)$ ✓ and $2n = o(n)$ ✗
- $\sqrt{n} = O(n)$ ✓ and $\sqrt{n} = o(n)$ ✓

### Big-Ω: $\Omega(\cdot)$ — "Lower Bound" or "At Least This Fast"

**Definition:** $f(n) = \Omega(g(n))$ means there exist constants $C > 0$ and $n_0$ such that:
$$f(n) \geq C \cdot g(n) \text{ for all } n \geq n_0$$

**Translation:** "$f$ grows **at least as fast as** $g$"

**Why it matters:** If you prove a lower bound of $\Omega(1/\varepsilon^2)$, you're saying "you **cannot** do better than this—any algorithm needs at least this many samples"

### Big-Θ: $\Theta(\cdot)$ — "Tight Bound" or "Exactly This Rate"

**Definition:** $f(n) = \Theta(g(n))$ means:
$$f(n) = O(g(n)) \text{ AND } f(n) = \Omega(g(n))$$

**Translation:** "$f$ grows **at exactly the same rate** as $g$"

**In the article:** When we say finding one needle is $\Theta(1/\varepsilon)$, we mean:
- Upper bound: you **can** do it with $O(1/\varepsilon)$ samples
- Lower bound: you **must** use $\Omega(1/\varepsilon)$ samples
- Together: it's **exactly** $\Theta(1/\varepsilon)$

## How to Read These in the Article's Context

Let's look at some specific examples from the article:

### Example 1: "Result: $n \sim O(1/\varepsilon)$"

This appears in Pattern 1. Here's how to think about it:

```
n ≥ ln(1/δ) / ε
```

The **dominant term** is $1/\varepsilon$. The $\ln(1/\delta)$ is just a constant factor that depends on your confidence level. So we write:

$$n = O(1/\varepsilon)$$

**What this tells you:**
- If $\varepsilon$ is halved, you need roughly **twice** as many samples
- The **logarithmic** dependence on $\delta$ is **negligible** compared to the **linear** dependence on $1/\varepsilon$

### Example 2: Chernoff gives "$O(1/\varepsilon^2)$"

```
n ≥ 2ln(2/δ) / ε²
```

Now the dominant term is $1/\varepsilon^2$:

$$n = O(1/\varepsilon^2)$$

**What this tells you:**
- If $\varepsilon$ is halved, you need roughly **four times** as many samples (quadratic!)
- This is **worse** scaling than $O(1/\varepsilon)$ when $\varepsilon$ is small

**Comparison:**
- $\varepsilon = 0.1$: $O(1/\varepsilon) \sim 10$ vs $O(1/\varepsilon^2) \sim 100$
- $\varepsilon = 0.01$: $O(1/\varepsilon) \sim 100$ vs $O(1/\varepsilon^2) \sim 10,000$

The gap **explodes** as $\varepsilon$ shrinks!

### Example 3: "Finding all: $O((1/\varepsilon)\log N)$"

```
n ≥ (1/ε) · [ln(N) + ln(1/δ)]
```

This is saying:

$$n = O\left(\frac{\log N}{\varepsilon}\right)$$

**What this tells you:**
- Linear in $1/\varepsilon$ (like finding one)
- But now also **logarithmic** in $N$ (the size of the space)
- Doubling $N$ only adds a **tiny** amount (one more bit of information)

**The magic:** Even though $N$ might be **massive** (millions or billions), $\log N$ is tiny:
- $N = 10^6$ → $\log N \approx 14$
- $N = 10^9$ → $\log N \approx 21$

So $\log N$ barely matters compared to $1/\varepsilon$ when $\varepsilon$ is small!

## The Patterns You See in Proofs

When you're reading proofs with Landau notation, here's the **mental workflow**:

### Step 1: Identify the exact bound

You'll see something like:
$$\mathbb{P}[\text{fail}] \leq 2e^{-2n\varepsilon^2}$$

### Step 2: Set equal to your failure tolerance

$$2e^{-2n\varepsilon^2} \leq \delta$$

### Step 3: Solve for $n$

Take logs:
$$\ln(2) - 2n\varepsilon^2 \leq \ln(\delta)$$
$$-2n\varepsilon^2 \leq \ln(\delta) - \ln(2) = \ln(\delta/2)$$
$$n \geq -\frac{\ln(\delta/2)}{2\varepsilon^2} = \frac{\ln(2/\delta)}{2\varepsilon^2}$$

### Step 4: Extract the Landau notation

Look at the **dominant terms**:
- Main term: $1/\varepsilon^2$ (this is what **explodes** when $\varepsilon$ shrinks)
- Logarithmic term: $\ln(2/\delta)$ (grows **slowly** with $1/\delta$)

So we write:
$$n = O(1/\varepsilon^2)$$

or more precisely:
$$n = O\left(\frac{\log(1/\delta)}{\varepsilon^2}\right)$$

**Key insight:** We're **hiding** the constant factors and the logarithmic terms because:
1. Constants like 2 don't affect the **scaling behavior**
2. Logarithmic terms grow so slowly they're often negligible
3. We care about **how things scale** as $\varepsilon \to 0$ or $N \to \infty$

## When Can You "Write Big-O on the Next Step"?

Here's the pattern recognition game:

### Pattern 1: Dropping constant factors

**If you have:**
$$n \geq \frac{5 \ln(3/\delta)}{\varepsilon^2}$$

**You can write:**
$$n = O(1/\varepsilon^2)$$

**Because:** The 5 and the 3 are **constants** that don't affect the **rate of growth**

### Pattern 2: Dropping lower-order terms

**If you have:**
$$n = \frac{1}{\varepsilon^2} + \frac{10}{\varepsilon} + 7$$

**You can write:**
$$n = O(1/\varepsilon^2)$$

**Because:** When $\varepsilon$ is small:
- $1/\varepsilon^2$ **dominates** (e.g., if $\varepsilon = 0.01$, this is $10,000$)
- $10/\varepsilon$ is **negligible** (this is only $1,000$)
- $7$ is **tiny** (just $7$!)

**The rule:** Keep only the **fastest-growing** term

### Pattern 3: Logarithms are often "free"

**If you have:**
$$n = \frac{\log N}{\varepsilon}$$

**You might write:**
$$n = O(1/\varepsilon)$$

**or more carefully:**
$$n = O((1/\varepsilon) \log N)$$

**Why the ambiguity?** It depends on context:
- If $N$ is **fixed** (not growing), $\log N$ is just a constant
- If $N$ is **variable**, we include it

### Pattern 4: Products and sums

**Products:** $O(f) \cdot O(g) = O(f \cdot g)$

Example:
$$n = O(1/\varepsilon) \cdot O(\log N) = O\left(\frac{\log N}{\varepsilon}\right)$$

**Sums:** $O(f) + O(g) = O(\max\{f, g\})$

Example:
$$n = O(1/\varepsilon) + O(1/\varepsilon^2) = O(1/\varepsilon^2)$$

(The $1/\varepsilon^2$ term **dominates**)

## The Approximation Dance

One of the key moves in the article is using **approximations** to get cleaner bounds. Here's how to think about it:

### The $\ln(1-\varepsilon) \approx -\varepsilon$ approximation

**Exact:**
$$\ln(1-\varepsilon) = -\varepsilon - \frac{\varepsilon^2}{2} - \frac{\varepsilon^3}{3} - \cdots$$

**Approximation:**
$$\ln(1-\varepsilon) \approx -\varepsilon$$

**When is this good?**
- When $\varepsilon$ is **small** (say $\varepsilon < 0.1$)
- The error is $O(\varepsilon^2)$, which is **negligible** compared to the main term

**Why use it?**
$$(1-\varepsilon)^n = e^{n\ln(1-\varepsilon)} \approx e^{-n\varepsilon}$$

This is **much easier** to work with than $(1-\varepsilon)^n$!

**The Landau version:**
$$\ln(1-\varepsilon) = -\varepsilon + O(\varepsilon^2)$$

or even more precisely:
$$\ln(1-\varepsilon) = -\varepsilon + o(\varepsilon)$$

(The remaining terms are **strictly smaller** than $\varepsilon$)

### The inequality version: $(1-\varepsilon)^n \leq e^{-n\varepsilon}$

The article mentions this comes from **convexity**. Here's the intuition:

Since $\ln(1-\varepsilon) \leq -\varepsilon$ for all $\varepsilon \in [0,1)$, we have:
$$(1-\varepsilon)^n = e^{n\ln(1-\varepsilon)} \leq e^{-n\varepsilon}$$

This is an **inequality**, not just an approximation! So it gives us a **rigorous upper bound**.

**Landau interpretation:**
$$(1-\varepsilon)^n = e^{-n\varepsilon}(1 + O(\varepsilon))$$

for small $\varepsilon$. The $O(\varepsilon)$ term captures the "error" in the approximation.

## Practical Examples from the Article

### Example 1: Pattern 1 transition

**Start with:**
$$\mathbb{P}[\text{miss all}] = (1-\varepsilon)^n$$

**Approximate:**
$$(1-\varepsilon)^n \approx e^{-n\varepsilon}$$

**Write the constraint:**
$$e^{-n\varepsilon} \leq \delta$$

**Solve:**
$$n \geq \frac{\ln(1/\delta)}{\varepsilon}$$

**Extract Landau notation:**
$$n = O(1/\varepsilon)$$

**What we dropped:**
- Constant factors (the "1" in $\ln(1/\delta)$ becomes $O(1)$)
- Logarithmic dependence on $\delta$ (if $\delta$ is fixed, this is $O(1)$)

### Example 2: Comparing Pattern 1 and Chernoff

**Pattern 1:**
$$n = O(1/\varepsilon)$$

**Chernoff:**
$$n = O(1/\varepsilon^2)$$

**What's going on?**
- Pattern 1: "Did we find **anything at all**?" → Linear scaling
- Chernoff: "Is our **sample frequency close** to the true frequency?" → Quadratic scaling

**The intuition:**
- Finding **one** thing is "easy" → $O(1/\varepsilon)$
- **Estimating** the frequency accurately is "harder" → $O(1/\varepsilon^2)$

The **variance** of a Bernoulli is proportional to $\varepsilon(1-\varepsilon)$, so to get error $\varepsilon$ in our estimate, we need $O(1/\varepsilon^2)$ samples.

### Example 3: When $N$ appears

**Pattern 3 (False positives):**
$$n \geq \frac{\ln(N/\delta)}{2\varepsilon^2}$$

**Extract:**
$$n = O\left(\frac{\log N}{\varepsilon^2}\right)$$

**Why $\log N$?**
We're doing a **union bound** over $N$ items:
$$\mathbb{P}[\text{any false positive}] \leq N \cdot \mathbb{P}[\text{single false positive}]$$

That $N$ multiplier gives us an extra $\log N$ in the sample complexity!

**But notice:** Even if $N = 10^9$ (a billion), $\log N \approx 30$, so this is **way better** than the naive $O(N)$ exhaustive search.

## The Big Takeaway

Landau notation is your **abstraction layer** for understanding:

1. **Scaling behavior**: How do requirements grow as parameters change?
2. **Bottlenecks**: Which terms **dominate** and which are **negligible**?
3. **Comparisons**: Is algorithm A **fundamentally better** than algorithm B?

**The workflow:**
1. Derive the **exact** bound (with all constants)
2. Identify the **dominant term(s)**
3. Write the Landau notation to **abstract away** the details
4. Use this to **compare** different approaches

**In proofs:**
- You **don't** write $O(\cdot)$ until you've **solved** for the quantity
- Once you have the solution, you **extract** the dominant terms
- The Landau notation is the **conclusion**, not the intermediate steps

**Reading tip:**
When you see $n = O(1/\varepsilon^2)$ in the article, mentally translate it as:

> "The sample size needs to be **roughly** $C/\varepsilon^2$ for some constant $C$. The exact value of $C$ depends on our confidence level $\delta$ and might be 2 or 5 or 10, but who cares? The key point is it scales **quadratically** with $1/\varepsilon$."

That's the **spirit** of Landau notation—focusing on the **shape** of the relationship rather than the exact numbers.