@def title = "Design Patterns for Concentration Inequalities in Sample Complexity"
@def published = "8 December 2025"
@def tags = ["machine-learning", "concentration-inequalities"]

# Design Patterns for Concentration Inequalities in Sample Complexity

Hey! So you're wondering about the key "design patterns" that show up when we use concentration inequalities to figure out sample complexity? Let me walk you through the main ideas in a conversational way.

## The Big Picture

Think of concentration inequalities as your toolbox for answering: **"How many samples do I need to be confident about something?"** The cool thing is that certain patterns show up over and over again, kind of like design patterns in software engineering.

## Pattern 1: The "Finding at Least One Needle" Pattern

**The Setup:** You have a huge haystack with $N$ possible items. Some fraction $\varepsilon$ of them are "good" (the needles you want). You sample uniformly at random.

**The Question:** How many samples $n$ do I need to find at least one good item with probability $\geq 1-\delta$?

**The Pattern:**
$$n \geq \frac{\ln(1/\delta)}{\varepsilon}$$

Or more conservatively (using Chernoff):
$$n \geq \frac{2\ln(2/\delta)}{\varepsilon^2}$$

**Why It Works:** Each sample is like a coin flip with success probability $\varepsilon$. The probability of missing all good items after $n$ samples is:
$$(1-\varepsilon)^n \approx e^{-n\varepsilon}$$

Set this equal to your acceptable failure rate $\delta$, solve for $n$, and boom—you have your sample size.

> **Derivation of $n \geq \frac{\ln(1/\delta)}{\varepsilon}$**
> 
> **Step 1:** For any single sample, probability of NOT picking a good item = $(1 - \varepsilon)$
> 
> **Step 2:** Probability of missing ALL good items in $n$ independent samples:
> $$P(\text{miss all}) = (1 - \varepsilon)^n$$
> 
> **Step 3:** Use the approximation $(1-\varepsilon)^n \approx e^{-n\varepsilon}$ (valid for small $\varepsilon$)
> 
> This comes from $\ln(1-\varepsilon) \approx -\varepsilon$ for small $\varepsilon$.
>
> **Yes, this is related to convexity!** More precisely, it's the **first-order Taylor approximation** of $\ln(1-\varepsilon)$ around $\varepsilon = 0$:
> $\ln(1-\varepsilon) = -\varepsilon - \frac{\varepsilon^2}{2} - \frac{\varepsilon^3}{3} - \cdots \approx -\varepsilon$
>
> But there's also a **convexity inequality** at play here! Since $\ln(x)$ is **concave**, we have:
> $\ln(1-\varepsilon) \leq -\varepsilon \quad \text{for all } \varepsilon \in [0,1)$
>
> So the approximation $(1-\varepsilon)^n \approx e^{-n\varepsilon}$ is actually an **upper bound** (we're approximating from above):
> $(1-\varepsilon)^n = e^{n\ln(1-\varepsilon)} \leq e^{-n\varepsilon}$
>
> This is **crucial** for the concentration bound! We're using:
> - **Upper bound** on failure probability: $(1-\varepsilon)^n \leq e^{-n\varepsilon}$
> - Which gives us a **lower bound** on sample size when we solve for $n$
>
> So yes, the convexity of $\ln$ (or equivalently, the inequality $1-x \leq e^{-x}$) is what makes this work!
> 
> **Step 4:** We want failure probability $\leq \delta$:
> $$e^{-n\varepsilon} \leq \delta$$
> 
> **Step 5:** Take natural log of both sides:
> $$-n\varepsilon \leq \ln(\delta) = -\ln(1/\delta)$$
> 
> **Step 6:** Divide by $-\varepsilon$ (flips inequality):
> $$n \geq \frac{\ln(1/\delta)}{\varepsilon}$$
> 
> Done! This tells us how many samples we need so that the probability of missing all good items is at most $\delta$.

> **Why is the Chernoff bound more conservative?**
> 
> The first bound $n \geq \frac{\ln(1/\delta)}{\varepsilon}$ asks: "How many samples to find **at least one** good item?"
> 
> The Chernoff bound $n \geq \frac{2\ln(2/\delta)}{\varepsilon^2}$ asks a stricter question: "How many samples so that the **empirical frequency** of good items is close to the true frequency $\varepsilon$?"
> 
> Here's the difference:
> 
> **First bound (coupon collector style):**
> - Let $X =$ number of good items found
> - We want $P(X \geq 1) \geq 1 - \delta$
> - This is asking: "Did we find anything at all?"
> - Result: $n \sim O(1/\varepsilon)$
> 
> **Chernoff bound (concentration style):**
> - Let $\hat{\varepsilon} =$ empirical fraction of good items in our sample
> - We want $P(|\hat{\varepsilon} - \varepsilon| \leq \text{some tolerance}) \geq 1 - \delta$
> - This is asking: "Is our sample frequency representative?"
> - Chernoff gives: $P(X = 0) \leq \exp(-n\varepsilon/2)$ when $\mathbb{E}[X] = n\varepsilon$
> - Setting this $\leq \delta$ gives $n \geq \frac{2\ln(1/\delta)}{\varepsilon}$
> 
> But actually for **one-sided** deviation (missing all), we use:
> $$P(X = 0) \leq \exp\left(-\frac{(n\varepsilon)^2}{2n}\right) = \exp\left(-\frac{n\varepsilon^2}{2}\right)$$
> 
> Wait, this gives $n \geq \frac{2\ln(1/\delta)}{\varepsilon^2}$, which is the $1/\varepsilon^2$ form!
> 
> **The key difference:**
> - Simple bound: Uses $(1-\varepsilon)^n \approx e^{-n\varepsilon}$ directly (tight for this specific question)
> - Chernoff: Gives a **general** concentration result that works for all deviations, not just the zero case. It's designed to bound $|X - \mathbb{E}[X]|$, which is overkill if you only care about $X = 0$.
> 
> **Bottom line:** For finding "at least one," use $O(1/\varepsilon)$. For getting **good empirical estimates** or **finding most/all** items, Chernoff's $O(1/\varepsilon^2)$ is more appropriate because it gives stronger concentration guarantees.

> **Where does this come from? PAC Learning or High-Dimensional Probability?**
>
> Great question! The answer is: **both**, and they're deeply connected.
>
> **From High-Dimensional Probability:**
> - Chernoff, Hoeffding, and other concentration inequalities are fundamentally results from **probability theory**
> - They study how random variables concentrate around their means
> - Classic reference: Boucheron, Lugosi, Massart - "Concentration Inequalities" (2013)
> - The $O(1/\varepsilon^2)$ scaling comes from the **variance** of Bernoulli random variables
>
> **From PAC Learning:**
> - PAC (Probably Approximately Correct) learning asks: "How many samples to learn a concept?"
> - The framework: Given error $\varepsilon$ and confidence $1-\delta$, find sample complexity
> - PAC sample complexity for finite hypothesis classes: $n = O\left(\frac{1}{\varepsilon^2}\log\frac{|H|}{\delta}\right)$
> - Notice the same $1/\varepsilon^2$ dependence!
> - Classic reference: Valiant (1984) - "A Theory of the Learnable"
>
> **The Connection:**
> - PAC learning **uses** concentration inequalities (especially Hoeffding) to derive sample complexity bounds
> - The typical PAC proof goes:
>   1. Use Hoeffding to bound error for a single hypothesis: $P(\text{error} > \varepsilon) \leq 2e^{-2n\varepsilon^2}$
>   2. Use Union Bound over all $|H|$ hypotheses: $P(\text{any bad hypothesis}) \leq 2|H|e^{-2n\varepsilon^2}$
>   3. Set this $\leq \delta$ and solve for $n$
> - This is exactly Pattern 3 (avoiding false positives) from below!
>
> **In the motif discovery context:**
> - We're essentially doing **finite hypothesis testing** (each configuration is a hypothesis)
> - We want to find all "good" hypotheses with high probability
> - This is very PAC-like: learn the set of significant configurations
> - But the underlying math is high-dimensional probability (concentration inequalities)
>
> **Historical lineage:**
> ```
> High-Dim Probability (1960s-70s)
>   ↓ (Chernoff, Hoeffding, etc.)
> Statistical Learning Theory (1980s-90s)
>   ↓ (Vapnik, Valiant - PAC framework)
> Modern ML Sample Complexity (2000s+)
>   ↓ (Applications to various domains)
> Your motif discovery problem (2020s)
> ```
>
> So the answer is: it's **probability theory at its core**, popularized and systematized by **PAC learning**, and now applied to **computational biology**!

**Key Insight:** Notice that $n$ **doesn't depend on $N$** (the total number of items)! It only depends on the density $\varepsilon$ of good items. This is why sampling can beat exhaustive search—even if $N = 1$ billion, if $\varepsilon = 0.01$, you only need a few thousand samples.

**Example:** 
- If 1% of configurations are biologically significant ($\varepsilon = 0.01$)
- Want 99% confidence ($\delta = 0.01$)
- Need roughly $n \approx 106,000$ samples (conservative bound) or $n \approx 300-500$ in practice

## Pattern 2: The "Finding All Needles" Pattern

**The Setup:** Same haystack, but now you want to find **all** $K = \varepsilon N$ good items, not just one.

**The Pattern:**
$$n \geq \frac{1}{\varepsilon} \cdot \left[\ln(N) + \ln(1/\delta)\right]$$

**Why It Works:** This is related to the **coupon collector problem**. To collect all coupons, you need $O(N \ln N)$ samples. But we only need to collect the $K = \varepsilon N$ good coupons, so we get:
$$n \approx \frac{N}{K} \cdot \ln(K) = \frac{1}{\varepsilon} \cdot \ln(\varepsilon N)$$

**Key Insight:** Now $N$ **does** appear, but only **logarithmically**! So even though finding all needles is harder than finding one, it's still way better than $O(N)$ exhaustive search.

**Scaling Comparison:**
- Finding one: $O(1/\varepsilon^2)$ — doesn't scale with $N$
- Finding all: $O((1/\varepsilon) \log N)$ — scales logarithmically with $N$  
- Exhaustive: $O(N)$ — scales linearly with $N$

## Pattern 3: The "Avoiding False Positives" Pattern

**The Setup:** You're sampling and testing items. Some items are actually bad but might randomly appear good in your sample (false positives). You want to control the probability that **any** bad item looks good.

**The Pattern (Union Bound + Hoeffding):**

If there are $N$ total items and you don't want any item with true frequency $p = 0$ to appear with empirical frequency $\hat{p} \geq \varepsilon$, you need:
$$n \geq \frac{\ln(N/\delta)}{2\varepsilon^2}$$

**Why It Works:** 
1. For a single item, Hoeffding's inequality bounds the probability of large deviations: $\mathbb{P}[|\hat{p} - p| > \varepsilon] \leq 2e^{-2n\varepsilon^2}$
2. Union bound over all $N$ items: $\mathbb{P}[\text{any false positive}] \leq N \cdot 2e^{-2n\varepsilon^2}$
3. Set this $\leq \delta$ and solve for $n$

**Key Insight:** This is where $N$ matters more directly. The more items you test, the more chances for false positives, so you need more samples to control them. But it's still logarithmic in $N$, not linear!

## Pattern 4: The "Combined Guarantee" Pattern

**The Setup:** You want to simultaneously (a) find all good items and (b) avoid false positives.

**The Pattern:**
$$n = \max\left\{\frac{2\ln(2/\delta)}{\varepsilon^2}, \frac{\ln(N/\delta)}{2\varepsilon^2}\right\}$$

**Why It Works:** Take the maximum of the two requirements. Usually, for large $N$ and small $\varepsilon$, the false positive bound dominates.

**Practical Note:** These bounds are often quite conservative! In practice, you can often get away with:
$$n_{\text{practical}} \approx \frac{k}{\varepsilon}$$
where $k = 3$ to $5$ gives you ~95-99% success rate.

## Pattern 5: The "Crossover Point" Pattern

**The Question:** When does sampling beat exhaustive enumeration?

**The Pattern:** Compare costs:
- Exhaustive: $N$
- Sampling: $(1/\varepsilon) \ln N$

Sampling wins when:
$$\frac{1}{\varepsilon} \ln N < N$$

Which simplifies to:
$$N > e^{1/\varepsilon}$$

**Example:** For $\varepsilon = 0.01$, crossover happens around $N \approx e^{100}$, but in practice with $q$-combinations, the crossover for motif discovery happens around $\alpha \approx 60$ (where $N = \binom{\alpha}{3}$).

## Pattern 6: The "Multiple Testing" Pattern

**The Setup:** You're doing the same sampling procedure across $M$ different datasets/sequences. Want guarantees to hold across all of them.

**The Pattern (Bonferroni Correction):**

For each dataset $i$, use failure probability:
$$\delta_i = \frac{\delta}{M}$$

Then overall guarantee holds with probability $\geq 1-\delta$.

**Why It Works:** Union bound again:
$$\mathbb{P}[\text{fail in at least one}] \leq \sum_{i=1}^M \mathbb{P}[\text{fail in } i] \leq M \cdot \frac{\delta}{M} = \delta$$

## Common Concentration Inequalities Used

### 1. **Chernoff Bound** (for finding rare events)
For sum of independent Bernoullis $X = \sum X_i$ where $\mathbb{E}[X] = \mu$:
$$\mathbb{P}[X = 0] \leq e^{-\mu}$$

More generally:
$$\mathbb{P}[X \leq (1-t)\mu] \leq e^{-t^2\mu/2}$$

**Use when:** You want to know if you'll find at least one good item.

### 2. **Hoeffding's Inequality** (for empirical frequencies)
For empirical average $\hat{p} = \frac{1}{n}\sum X_i$ where true mean is $p$:
$$\mathbb{P}[|\hat{p} - p| > \varepsilon] \leq 2e^{-2n\varepsilon^2}$$

**Use when:** You want to bound how far your sample frequency is from the true frequency.

### 3. **Union Bound** (for multiple events)
$$\mathbb{P}\left[\bigcup_i A_i\right] \leq \sum_i \mathbb{P}[A_i]$$

**Use when:** You want to control the probability that **any** of many bad things happens (like false positives).

## Putting It All Together: A Recipe

Here's the general workflow:

1. **Define your "success" event** (e.g., finding a significant configuration)

2. **Estimate the density** $\varepsilon$ of successes in your space

3. **Choose your confidence level** $1-\delta$ (typically 0.95 or 0.99)

4. **Pick the right pattern:**
   - Need at least one? Use Pattern 1
   - Need most/all? Use Pattern 2  
   - Worried about false positives? Combine with Pattern 3
   - Multiple datasets? Add Pattern 6

5. **Compute sample size** from the formulas

6. **Reality check:** The theoretical bounds are often 10-100× more conservative than needed in practice

## Why These Patterns Matter

The beautiful thing about concentration inequalities is they give you **distribution-free** guarantees. You don't need to know the exact distribution of your data—just some basic parameters like $\varepsilon$ (density of good items) and $N$ (size of space).

And the **sublinear scaling** (logarithmic in $N$ instead of linear) is what makes sampling practical for huge spaces. When $N = 20$ million configurations but you only need to sample 100,000, that's a 200× speedup!

## The Gap: Theory vs Practice

One last thing worth noting: there's often a big gap between theoretical bounds and practical needs.

**Theory says:** $n = \frac{2\ln(2/\delta)}{\varepsilon^2} \approx 106,000$ for $\varepsilon=0.01$, $\delta=0.01$

**Practice shows:** $n = \frac{k}{\varepsilon} \approx 300-500$ works great with $k=3-5$

Why the gap? The theoretical bounds are **worst-case guarantees** that work even in unlucky scenarios. But in the average case, you need far fewer samples.

**Adaptive strategy:** Start with practical estimates, and if you're not finding anything, gradually increase toward the theoretical bound. This way you get efficiency in the common case but still have guarantees for the worst case.

---

## The Elephant in the Room: The i.i.d. Assumption

You're absolutely right to call this out! Let's talk about what's really holding this whole framework together.

> **Why does all of this work? The i.i.d. assumption!**
>
> **The Foundation:**
> - All these concentration inequalities (Chernoff, Hoeffding, etc.) **fundamentally assume** samples are **independent and identically distributed (i.i.d.)**
> - Uniform sampling = every configuration has equal probability $1/N$
> - Independence = knowing you sampled config A tells you nothing about whether you'll sample config B next
> - This is what lets us write: $P(\text{miss all}) = (1-\varepsilon)^n$ (product of independent events)
>
> **Without i.i.d., the house of cards falls apart:**
> - If samples are **dependent**, concentration is weaker (or stronger, depending on correlation structure)
> - If sampling is **non-uniform**, you lose the clean probabilistic guarantees
> - The beautiful $O(\log N)$ scaling can disappear

> **Can we break free from uniform sampling without heuristics?**
>
> This is a deep question, and the answer is: **sort of, but it's tricky!** Here are the main approaches:
>
> **1. Importance Sampling (Still Principled!)**
> - **Idea:** Sample from a proposal distribution $q(x)$ instead of uniform $p(x) = 1/N$
> - **Key:** Weight samples by $w(x) = p(x)/q(x)$ to get unbiased estimates
> - **Example:** If you think certain filter combinations are more likely to be significant, sample them more heavily, then weight appropriately
> - **Pros:** 
>   - Still mathematically rigorous! Unbiased estimator
>   - Can dramatically reduce variance if you choose $q$ well
> - **Cons:** 
>   - Need to **know** or **estimate** a good proposal $q$ (this is the hard part)
>   - Concentration bounds now depend on the variance of the weights, which can be worse if $q$ is poorly chosen
>   - You're not really avoiding heuristics—choosing $q$ is itself a heuristic!
>
> **2. Stratified Sampling (Structured Non-Uniformity)**
> - **Idea:** Partition space into strata, sample uniformly **within** each stratum
> - **Example:** For motif discovery, stratify by which filters appear
> - **Key:** This is still uniform within strata, so concentration bounds apply per-stratum
> - **Pros:**
>   - Can improve efficiency if strata are natural
>   - Still principled—union bound over strata
> - **Cons:**
>   - How do you choose strata? (Heuristic alert!)
>   - If strata are unbalanced, doesn't help much
>
> **3. Adaptive/Sequential Sampling (Explore-Exploit)**
> - **Idea:** Use early samples to guide later samples (multi-armed bandits, Thompson sampling)
> - **Example:** If you find promising patterns in region A, sample more from region A
> - **Pros:**
>   - Can be very efficient in practice
>   - Some theoretical guarantees exist (regret bounds for bandits)
> - **Cons:**
>   - Samples are now **dependent**!
>   - Standard concentration inequalities don't apply directly
>   - Need specialized analysis (martingale concentration, etc.)
>   - This is **definitely** introducing heuristics in how you adapt
>
> **4. Active Learning / Query Complexity**
> - **Idea:** Intelligently choose which configurations to test based on maximizing information gain
> - **Pros:**
>   - Theoretically well-studied in some settings
>   - Can achieve better bounds than passive sampling
> - **Cons:**
>   - Need a model of what "informative" means (heuristic!)
>   - Often computationally expensive to compute information gain
>   - Breaks i.i.d. assumption
>
> **The Fundamental Tension:**
> 
> Here's the real issue: **Any non-uniform sampling strategy requires some kind of prior knowledge or belief about where the good stuff is.** And that's inherently a heuristic!
>
> ```
> Pure uniform sampling:
>   - No assumptions about where good items are
>   - Clean probabilistic guarantees
>   - May be inefficient if good items cluster
>
> Non-uniform sampling:
>   - Assumes you know something about the structure
>   - That assumption is a heuristic!
>   - But can be much more efficient
> ```
>
> **The Pragmatic Middle Ground:**
>
> In practice, you might do something like:
>
> 1. **Phase 1:** Uniform sampling to explore (small budget, say 5% of theoretical bound)
> 2. **Analyze:** Look at what you found—do patterns cluster in certain regions?
> 3. **Phase 2:** If you see clear structure, use importance/stratified sampling with theory-based weights
> 4. **Guarantee:** Use union bound over phases to maintain overall probabilistic guarantee
>
> This way you get:
> - Theoretical guarantees (from the uniform phase)
> - Practical efficiency (from the adaptive phase)
> - Some principled justification (importance sampling with reweighting)

> **What about the original motif discovery problem?**
>
> In your specific case with motif discovery:
>
> **Why uniform sampling makes sense:**
> - You genuinely don't know which filter combinations will be biologically meaningful
> - Biological signals might be subtle and not obvious from simple heuristics
> - The sparse coding already did the heavy lifting (learning the filters)
> - Configuration enumeration is just "which combinations co-occur?"
>
> **Where you might break uniformity:**
> - **Spatial constraints:** If filters must be nearby on the sequence (local patterns), you could sample locally-clustered configurations
>   - This is semi-principled: encode biological prior about locality
> - **Filter co-occurrence:** If you have training data showing certain filters often co-occur, use that as a prior for importance sampling
>   - Need to be careful: this might be circular (finding what you expect to find)
> - **Two-phase:** Coarse uniform sampling → identify hot regions → dense sampling in hot regions
>   - This is what the article's "adaptive two-phase sampling" does, and it's reasonable!
>
> **My Take:**
>
> The beauty of the uniform sampling approach is its **intellectual honesty**—it makes no assumptions beyond the sparse representation itself. Any heuristic you add is a claim about the structure of the problem, and you should:
> 1. Make that claim explicit
> 2. Justify why you believe it
> 3. Ideally, validate it empirically
>
> For a first principled approach, uniform sampling is the right starting point. Once you understand the structure of solutions in your domain, *then* you can introduce informed non-uniform sampling strategies. But at that point, you're no longer doing pure statistical sampling—you're doing domain-informed search, which is a different (and valid!) paradigm.

---

Hope this helps! The key insight is that these patterns give you the power to **trade exhaustive search for randomness + probability theory**, and concentration inequalities tell you exactly how much randomness you need to be confident in your results. But yes, the i.i.d. assumption is the bedrock—without it, you need much more sophisticated tools!