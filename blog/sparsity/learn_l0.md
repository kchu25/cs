@def title = "Learning Sparse Neural Networks Through L0 Regularization"
@def published = "10 October 2025"
@def tags = ["machine-learning", "sparsity"]
# A Conversational Guide to Bayesian Networks

# Learning Sparse Neural Networks Through L0 Regularization

**Authors:** Christos Louizos, Max Welling, Diederik P. Kingma  
**Published:** ICLR 2018

[https://arxiv.org/pdf/1712.01312](https://arxiv.org/pdf/1712.01312)

---

## Deep Dive: What Does "Disconnected" Actually Mean?

### The Question

> "Disconnected: how? Do you mean I can't go to good local optimum? Like every initialization is going to give you very distinct results?"

Great question! "Disconnected" is a precise topological concept that's crucial to understanding why L0 optimization is fundamentally different from typical neural network optimization.

### What "Disconnected" Means Geometrically

The constraint set $\mathcal{C}_k = \{\theta : \|\theta\|_0 \leq k\}$ is **disconnected** means it consists of **separate pieces that don't touch**.

#### Simple Example: 2D with k=1

Consider $\mathcal{C}_1 = \{\theta \in \mathbb{R}^2 : \|\theta\|_0 \leq 1\}$

This set consists of:
- The **x-axis**: $\{(\theta_1, 0) : \theta_1 \in \mathbb{R}\}$
- The **y-axis**: $\{(0, \theta_2) : \theta_2 \in \mathbb{R}\}$  
- The **origin**: $\{(0, 0)\}$

Picture this:
```
    |
    | y-axis (one piece)
    |
----+---- x-axis (another piece)
    |
    |
```

The point $(1, 0)$ and the point $(0, 1)$ are **both in the constraint set**, but there's **no continuous path** between them that stays within the set!

To go from $(1, 0)$ to $(0, 1)$ while staying in $\mathcal{C}_1$:
1. Walk along x-axis: $(1, 0) \to (0.5, 0) \to (0.1, 0) \to (0, 0)$
2. Now you're stuck! To reach $(0, 1)$, you'd have to move to $(0, 0.1)$, but that would temporarily take you through a point like $(0.001, 0.001)$ which has $\|\theta\|_0 = 2$ (violates the constraint!)

The x-axis and y-axis are **separate islands** connected only at the origin.

### Why This Breaks Gradient Descent

#### 1. You're Trapped in Your Component

If you initialize at $\theta = (2, 0)$, gradient descent can only move you along the x-axis. You can reach $(1, 0)$, $(0, 0)$, $(-5, 0)$, but **never** $(0, 1)$ or $(1, 1)$.

Think of it like being on an island with no bridges. Gradient descent can only walk on continuous ground - it can't jump across water.

#### 2. Different Initializations → Completely Different Solutions

**Yes, exactly!** Each initialization lands you in a different connected component, and you're stuck there:

```python
# 3 parameters, k=1 constraint
init_1 = [1.0, 0.0, 0.0]  # stuck on θ₁-axis forever
init_2 = [0.0, 1.0, 0.0]  # stuck on θ₂-axis forever  
init_3 = [0.0, 0.0, 1.0]  # stuck on θ₃-axis forever
```

Each initialization explores a **completely different subspace** with no way to transition between them (without violating the constraint).

#### 3. Contrast with Non-Convex but Connected Sets

This is fundamentally different from typical neural network optimization:

**Neural network loss landscape** (non-convex but connected):
```
    *peak*        *peak*
   /      \      /      \
  /   😊   \____/   😊   \
 /   local      valley    \
```

You can walk from one local minimum to another through high-loss regions. They're in the **same connected piece**.

**L0 constraint** (disconnected):
```
😊 island 1        😊 island 2        😊 island 3
   (θ₁-axis)         (θ₂-axis)         (θ₃-axis)

    🌊 ocean 🌊 ocean 🌊
```

You **literally cannot** walk from one island to another while staying in the feasible set.

### How Many Components Are There?

For $\|\theta\|_0 \leq k$ in $\mathbb{R}^n$, there are $\sum_{i=0}^{k} \binom{n}{i}$ connected components!

- For $k=1, n=1000$: That's $1 + 1000 = 1001$ separate components  
- For $k=2, n=1000$: That's $1 + 1000 + \binom{1000}{2} = 500,501$ separate components!

Each component corresponds to a different **support pattern** (which subset of parameters is non-zero).

### The Mathematical Definition

A set $S$ is **disconnected** if it can be written as $S = A \cup B$ where:
- $A$ and $B$ are non-empty
- $A$ and $B$ are open in $S$ (in the subspace topology)
- $A \cap B = \emptyset$

For the L0 constraint set, each "axis" or subspace is one such component.

### Why the Convex Hull Is Different

The **convex hull** of $\mathcal{C}_k$ fills in all the space between these disconnected pieces:

```
L0 ball (k=1):        Convex hull (L1 ball):
    |                      /\
    |                     /  \
----+----               /    \
    |                  /  😊  \
    |                 /________\
(disconnected)         (connected!)
```

The L1 ball is a **diamond** that's completely filled in - you can walk from any point to any other point along a straight line staying inside.

### Bottom Line

"Disconnected" means:
- ✅ The feasible set breaks into separate islands
- ✅ Gradient descent traps you on whichever island you start on  
- ✅ Different initializations give fundamentally different solutions with no way to compare/transition between them
- ✅ There's no continuous optimization path that explores all possibilities

This is why L0 optimization requires either:
1. Exhaustive search over all $\binom{n}{k}$ components (intractable!)
2. Smart relaxations like this paper's approach (smooth the discrete structure)

---

## The Big Idea

This paper tackles a fundamental problem in deep learning: neural networks are **massively overparametrized**. They have way more weights than they actually need, which wastes computation and can lead to overfitting. The authors propose a clever way to automatically prune networks *during training* using **L0 regularization** - essentially penalizing the network for every non-zero parameter it uses.

---

## Why L0 Norm?

Think of different regularization techniques as penalties on parameters:
- **L2 (weight decay)**: Shrinks large weights - encourages small values
- **L1 (Lasso)**: Also shrinks weights, can push some to zero
- **L0**: Simply counts non-zero parameters - no shrinkage, just sparsity!

The L0 norm is defined as:

$$\|\theta\|_0 = \sum_{j=1}^{|\theta|} \mathbb{I}[\theta_j \neq 0]$$

It's the "ideal" sparsity penalty because it directly counts how many parameters you're using, without caring about their actual values. Famous model selection criteria like AIC and BIC are actually special cases of L0 regularization!

---

## The Problem

Here's the catch: the L0 norm is **non-differentiable** and **combinatorial**. With $|\theta|$ parameters, there are $2^{|\theta|}$ possible on/off configurations to consider. You can't just throw gradient descent at it!

### Why is L0 Non-Differentiable?

The L0 norm uses the indicator function $\mathbb{I}[\theta_j \neq 0]$:

$\mathbb{I}[\theta_j \neq 0] = \begin{cases} 1 & \text{if } \theta_j \neq 0 \\ 0 & \text{if } \theta_j = 0 \end{cases}$

Let's try to compute the gradient:

$\frac{\partial}{\partial \theta_j} \mathbb{I}[\theta_j \neq 0] = \, ?$

**At $\theta_j \neq 0$:** The function equals 1 (constant), so the derivative is 0.

**At $\theta_j = 0$:** The function has a discontinuous jump from 0 to 1. The derivative is undefined!

**The gradient is almost everywhere zero**, which means gradient descent gets no signal about which direction to move parameters. Even worse, when you're exactly at zero, the gradient doesn't exist at all!

### The Combinatorial Explosion

Even if we could handle the non-differentiability, there's a deeper problem. The L0 penalty creates a **discrete optimization landscape**:

- Each parameter can be either "on" ($\theta_j \neq 0$) or "off" ($\theta_j = 0$)
- With $n$ parameters, there are $2^n$ possible configurations
- For a small network with 10,000 parameters: $2^{10000} \approx 10^{3010}$ configurations!

This is fundamentally a **combinatorial search problem** (like the knapsack problem or subset selection), which is NP-hard. You can't smoothly "walk" from one configuration to another - you have to make discrete jumps.

**Gradient descent fails because:**
1. The gradient provides no information (it's zero almost everywhere)
2. The search space is discrete, not continuous
3. Local changes in parameters don't smoothly affect the objective

### Mathematical Formalization

The L0 minimization problem can be written as:

$\min_{\theta \in \mathbb{R}^n} f(\theta) + \lambda \|\theta\|_0 \quad \Leftrightarrow \quad \min_{S \subseteq \{1,\ldots,n\}} \min_{\theta_S \in \mathbb{R}^{|S|}} f(\theta_S) + \lambda|S|$

where $S$ is the **support set** (indices of non-zero parameters) and $\theta_S$ are parameters indexed by $S$.

This is the **best subset selection problem**, proven NP-hard by:
- **Natarajan (1995)**: Showed sparse approximation is NP-hard
- **Davis et al. (1997)**: Proved subset selection is NP-hard even for convex loss functions

**Key theoretical results:**

1. **Computational complexity**: Even finding a $\rho$-approximation (within factor $\rho$ of optimal) is NP-hard for any $\rho < 2$ (Arora et al., 1998)

2. **Non-convexity**: The constraint set $\{\theta : \|\theta\|_0 \leq k\}$ is **non-convex** and **disconnected**:
   - Convex hull is $\{\theta : \|\theta\|_1 \leq \sqrt{k}\|\theta\|_2\}$ (different geometry!)
   - No continuous path between feasible points through feasible region

3. **Lipschitz continuity fails**: For any $\theta$ with $\|\theta\|_0 = k$:
   $\lim_{\epsilon \to 0^+} \|\theta + \epsilon e_j\|_0 = k+1 \neq k$
   where $e_j$ is a standard basis vector with $\theta_j = 0$. The function "jumps" discontinuously.

4. **Subdifferential is uninformative**: The subdifferential of $\|\theta\|_0$ at $\theta$ is:
   $\partial \|\theta\|_0 = \begin{cases} \{0\} & \text{if } \theta_j \neq 0 \\ [-\infty, \infty] & \text{if } \theta_j = 0 \end{cases}$
   When $\theta_j = 0$, any direction is a valid subgradient!

---

## The Solution: Stochastic Gates

The authors' clever trick involves introducing "gate" variables:

$$\theta_j = \tilde{\theta}_j \cdot z_j, \quad z_j \in \{0, 1\}$$

Each parameter $\tilde{\theta}_j$ has a binary gate $z_j$ that decides whether it's "on" (1) or "off" (0). The network then optimizes:

$$R(\tilde{\theta}, \pi) = \mathbb{E}_{q(z|\pi)}\left[\frac{1}{N}\sum_{i=1}^N \mathcal{L}(h(x_i; \tilde{\theta} \odot z), y_i)\right] + \lambda \sum_{j=1}^{|\theta|} \pi_j$$

where $q(z_j|\pi_j) = \text{Bernoulli}(\pi_j)$.

But wait - we still can't differentiate through discrete binary gates!

---

## The Hard Concrete Distribution

Here's where things get really clever. Instead of discrete gates, they use **continuous random variables** passed through a hard-sigmoid:

1. Sample $s \sim q(s|\phi)$ from a continuous distribution
2. Apply hard-sigmoid: $z = \min(1, \max(0, s))$
3. This allows $z$ to be exactly 0 or 1, while still being differentiable w.r.t. $\phi$!

The **hard concrete distribution** is created by:
- Starting with a binary concrete distribution (a smooth approximation of Bernoulli)
- "Stretching" it from $(0,1)$ to $(\gamma, \zeta)$ where $\gamma < 0$ and $\zeta > 1$
- Applying the hard-sigmoid rectification

The sampling procedure is:

$$u \sim U(0,1)$$
$$s = \text{Sigmoid}(\log u - \log(1-u) + \log \alpha)/\beta$$
$$\bar{s} = s(\zeta - \gamma) + \gamma$$
$$z = \min(1, \max(0, \bar{s}))$$

The probability of a gate being active is:

$$q(z \neq 0|\phi) = 1 - Q(s \leq 0|\phi)$$

This can be computed in closed form and differentiated!

---

## Why This Works

The hard concrete distribution has some really nice properties:
- **Exact zeros and ones**: Unlike softmax relaxations, you actually get sparse parameters during training
- **Differentiable**: The continuous part allows gradient-based optimization
- **Bimodal**: Like a Bernoulli, it concentrates mass at the endpoints (0 and 1)
- **Conditional computation**: Zero parameters don't need to be computed, enabling training speedups

---

## Results

### MNIST
On simple networks (MLP and LeNet-5), the method achieved:
- Competitive or better accuracy compared to other pruning methods
- Significant compression (e.g., LeNet: 20-50-800-500 → 20-25-45-462)
- Expected FLOPs reduction during training

### CIFAR-10/100 (Wide ResNets)
On modern architectures:
- **Better accuracy** than dropout baseline (3.83% vs 3.89% on CIFAR-10)
- **Faster training** through reduced FLOPs
- Automatic learning of which neurons to keep

---

## Key Advantages

1. **Trains sparse networks from scratch** - no need to train a dense network first
2. **Principled regularization** - direct connection to Bayesian inference with spike-and-slab priors
3. **Computational savings during training** - not just at inference time
4. **Simple to implement** - just add gates and optimize their parameters
5. **Group sparsity** - can easily do neuron or filter pruning by sharing gates

---

## The Math Connection

There's a beautiful theoretical connection: this approach is actually performing **variational inference** with spike-and-slab priors! The L0 penalty emerges naturally from:

$$p(z) = \text{Bernoulli}(\pi), \quad p(\theta|z=0) = \delta(\theta), \quad p(\theta|z=1) = \mathcal{N}(\theta|0, 1)$$

The method optimizes a variational lower bound where $\lambda$ represents the "code cost" - the amount of information each parameter encodes about the data.

---

## Practical Details

- Used $\gamma = -0.1$, $\zeta = 1.1$, $\beta = 2/3$
- Initialized $\log \alpha$ to match original dropout rates
- Single sample per minibatch (low variance, practical speedup)
- Can be combined with L2 regularization naturally
- Group sparsity by sharing gates across neuron/filter groups

---

## Limitations and Criticisms

While this paper presents an elegant solution, it has several important limitations:

### 1. **Gap Between Theory and Practice: Actual Speedup**

**The big caveat**: The paper shows *expected* FLOPs reduction and *theoretical* speedup, but:
- They only use **single samples** of gates per minibatch
- Actual hardware (GPUs) doesn't automatically speed up from having zeros - you need **structured sparsity**
- Modern hardware is optimized for dense operations; sparse operations can actually be *slower* due to irregular memory access patterns
- The paper doesn't provide **wall-clock time** measurements, only theoretical FLOP counts

**Reality check**: Unless you explicitly implement conditional computation (which they don't), training a network with stochastic gates can be *slower* than dense training due to sampling overhead.

### 2. **Gradient Variance from Single Samples**

The paper uses only **one sample** of $z$ per minibatch for "practical speedup":
- This introduces high variance in gradients
- Can lead to unstable training, especially early on
- The authors claim "this should not pose an issue" but provide limited empirical validation
- Most variance reduction techniques (multiple samples, control variates) would eliminate the speedup benefit

### 3. **Hyperparameter Sensitivity**

The method introduces several hyperparameters:
- $\lambda$: regularization strength (most critical)
- $\gamma, \zeta$: stretching parameters for hard concrete (-0.1, 1.1)
- $\beta$: temperature (2/3)
- Initialization of $\log \alpha$

**Issues**:
- $\lambda$ needs careful tuning and often requires **different values per layer** (as shown in their "λ sep." experiments)
- The "right" $\lambda$ depends on dataset size, architecture, task
- No principled way to set $\lambda$ a priori

### 4. **Limited Comparison with Concurrent Methods**

The paper compares mainly with:
- Magnitude pruning (Han et al., 2015)
- Bayesian methods (Louizos et al., 2017; Molchanov et al., 2017)

**Missing comparisons**:
- Lottery ticket hypothesis (Frankle & Carbin, 2019) - can you find winning tickets with L0?
- Modern pruning methods (magnitude pruning during training)
- Knowledge distillation approaches
- Quantization-aware training

### 5. **Sparsity Pattern May Not Be Optimal**

The method learns *which* parameters to prune, but:
- No guarantee this is the globally optimal sparsity pattern
- Different random seeds give different sparsity patterns
- The learned pattern depends heavily on initialization
- Post-hoc pruning methods can sometimes find better patterns by training dense first

### 6. **Group Sparsity Trade-offs**

While neuron/filter sparsity enables actual speedup, it's **less flexible** than weight sparsity:
- Removes entire features/neurons, which may be too coarse-grained
- Weight-level sparsity can achieve higher compression rates
- The paper doesn't explore more fine-grained structured sparsity (e.g., block sparsity, channel pruning patterns)

### 7. **Training Instability at High Sparsity**

The paper doesn't deeply explore:
- What happens with very high $\lambda$ (90%+ sparsity)?
- Training dynamics when many gates close early
- Whether the network can "recover" if it prunes important features too early
- The gates are stochastic during training but deterministic at test time - this train/test mismatch isn't fully analyzed

### 8. **Bayesian Interpretation Is Approximate**

The connection to spike-and-slab priors (Appendix A) makes strong approximations:
- Assumes fixed "code cost" $\lambda$ for all parameters
- Optimizes rather than integrates over $\theta$
- The variational bound is loose
- Not a true Bayesian treatment (despite the framing)

### 9. **Limited Analysis of What Gets Pruned**

The paper shows *how much* gets pruned but limited analysis of:
- *What* gets pruned (which features/patterns?)
- Are the learned sparse networks interpretable?
- Do different runs learn similar sparsity patterns?
- Sensitivity analysis of pruned vs. kept parameters

### 10. **Scalability Questions**

While tested on WideResNets, questions remain:
- Does it scale to very large models (BERT, GPT-scale)?
- How does it interact with other modern techniques (BatchNorm, LayerNorm, residual connections)?
- Performance on other domains (NLP, RL)?

### 11. **Hard-Sigmoid May Be Suboptimal**

The hard-sigmoid rectification $z = \min(1, \max(0, \bar{s}))$:
- Creates sparse gradients (zero gradient when $\bar{s} < 0$ or $\bar{s} > 1$)
- May lead to "dead gates" that never recover
- The clipping at 1 may be unnecessary (they could use $\max(0, \bar{s})$ only)

### What Follow-Up Work Addressed

Several papers have since improved on these limitations:
- **DiffRNN (Kuchaiev & Ginsburg, 2017)**: Better hardware-aware sparsity
- **Movement Pruning (Sanh et al., 2020)**: Pruning based on weight movement during training
- **Lottery Ticket Hypothesis (Frankle & Carbin, 2019)**: Shows dense training then pruning can work better
- **Soft threshold reparameterization (Kusupati et al., 2020)**: Alternative differentiable sparsity methods

---

## Has This Work Been Superseded by the Lottery Ticket Hypothesis?

**Short answer: No, but they address different problems with different philosophies.**

### The Lottery Ticket Hypothesis (Frankle & Carbin, 2019)

**Core claim**: Dense networks contain sparse "winning ticket" subnetworks that, when trained in isolation *from the same initialization*, can match the full network's performance.

**Method**:
1. Train dense network to completion
2. Prune smallest-magnitude weights
3. **Rewind** to original initialization
4. Retrain the sparse network from that initialization

**Key insight**: It's not just about *which* weights to keep, but also *how they're initialized*.

### Fundamental Differences

| Aspect | L0 Regularization (This Paper) | Lottery Tickets |
|--------|-------------------------------|-----------------|
| **Philosophy** | Learn sparsity *during* training | Find sparsity *after* training, then retrain |
| **Training cost** | Single training run | Train → prune → retrain (often iteratively) |
| **Initialization** | Standard initialization | Requires preserving "winning" initialization |
| **Mechanism** | Continuous relaxation + gates | Magnitude pruning + rewinding |
| **Sparsity decision** | Learned via gradient descent | Heuristic (magnitude-based) |
| **Theoretical grounding** | Variational inference, L0 penalty | Empirical observation |

### Why They're Complementary, Not Competing

**L0 Regularization strengths**:
- ✅ **Single training run** - more efficient if it works
- ✅ **Principled** - directly optimizes for sparsity
- ✅ **Differentiable** - can combine with other objectives
- ✅ **During-training benefits** - theoretical speedup while training

**Lottery Ticket strengths**:
- ✅ **Empirically robust** - works across many architectures/tasks
- ✅ **Simple** - easy to implement (no new distributions or gates)
- ✅ **Better final performance** - often achieves higher accuracy at same sparsity
- ✅ **Transferable insights** - reveals importance of initialization

### The Key Question: Which Approach Works Better?

**Empirically, Lottery Tickets often win** in terms of:
- Final accuracy at high sparsity levels (80-95%)
- Simplicity and reproducibility
- Breadth of applications

**But there's a cost**:
- Need to train the full dense network first (expensive!)
- Iterative pruning requires multiple full training runs
- Total compute can be 5-10x a single dense training run

**L0 Regularization is better when**:
- You want a single training run
- You need dynamic sparsity during training
- You want to combine sparsity with other learned objectives
- You need a differentiable sparsity mechanism

### The Current Landscape (2025 perspective)

Neither has "won" - both philosophies continue in modern work:

**Lottery Ticket descendants**:
- **Pruning at initialization** (Lee et al., 2019) - can we find tickets without training?
- **Early bird tickets** (You et al., 2020) - find tickets much earlier in training
- **Supermasks** (Zhou et al., 2019) - learn which weights to keep without training them

**L0/Differentiable sparsity descendants**:
- **STR (Soft Threshold Reparameterization)** (Kusupati et al., 2020)
- **Learned gradient sparsity** (Evci et al., 2020)
- **Dynamic sparse training** (Mocanu et al., 2018)

**Hybrid approaches**:
- Can you use L0 regularization to *find* lottery tickets faster?
- Can you use lottery ticket initialization with differentiable sparsity?

### My Take

The Lottery Ticket Hypothesis **shifted the paradigm** by showing:
1. Dense training has value (finds good initialization)
2. The initialization matters as much as the final weights
3. Simple magnitude pruning is surprisingly effective

But it **didn't make L0 regularization obsolete** because:
1. L0 is fundamentally about *learning* the sparsity pattern via gradient descent
2. Single-pass training is still valuable for efficiency
3. Differentiable sparsity enables things magnitude pruning can't (e.g., NAS, multi-objective optimization)

**Think of it this way**: Lottery Tickets showed that the "find sparse networks during training" problem might be approached backwards - train dense, then find the sparse network that was hiding inside. L0 regularization tried to learn the sparse network directly. Both perspectives have merit, and modern work often blends ideas from both.

## Bottom Line

This paper presents an elegant solution to neural network sparsification by making the intractable L0 norm optimization tractable through continuous relaxation. The hard concrete distribution provides the best of both worlds: exact sparsity and differentiable optimization. The method not only compresses networks but actually improves generalization and enables faster training - a rare win-win-win!

**However**, the gap between theoretical speedup and practical implementation, hyperparameter sensitivity, and limited large-scale validation are important caveats. The paper opened an important research direction, but practical deployment requires careful engineering and tuning.