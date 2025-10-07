@def title = "Learning to model the tail"
@def published = "6 October 2025"
@def tags = ["one-or-few-shots-learning"]

[https://www.ri.cmu.edu/app/uploads/2018/01/nips_2017_cameraready.pdf](https://www.ri.cmu.edu/app/uploads/2018/01/nips_2017_cameraready.pdf)

# Learning to Model the Tail

**Authors:** Yu-Xiong Wang, Deva Ramanan, Martial Hebert (CMU Robotics Institute)  
**Conference:** NIPS 2017

## The Gist

This paper tackles **long-tailed recognition** - learning accurate classifiers when classes have vastly imbalanced training data (some classes have thousands of examples, others have just a few). The key insight is to treat this as a meta-learning problem: learn how model parameters *evolve* as more training data becomes available, then use that knowledge to improve few-shot models for rare classes.

## Core Problem

Real-world datasets follow long-tailed distributions where:
- **Head classes**: abundant training data (hundreds/thousands of examples)
- **Tail classes**: scarce training data (as few as 1-10 examples)

Traditional approaches fail:
- **Over-sampling**: creates redundancy, leads to overfitting
- **Under-sampling**: loses critical information
- **Cost-sensitive weighting**: makes optimization difficult

## The Strategy: MetaModelNet

### What Networks Are Involved?

There are **THREE types of networks/models**:

#### **Network 1: Base Classifier $g(x; \theta)$**
- **What it is:** The actual task network (e.g., ResNet, AlexNet) that classifies images
- **Parameters:** $\theta$ (e.g., all CNN weights, or just final fully-connected layer)
- **Multiple instances:** You train MANY versions of this network with different amounts of data
- **Purpose:** Perform actual classification on images

#### **Network 2: Few-Shot Models $\theta_k$**
- **What it is:** Base classifier trained on only $k$ examples per class
- **Parameters:** Same architecture as Network 1, but trained with limited data
- **Examples:** 
  - $\theta_1$: classifier trained on 1 example per class
  - $\theta_2$: classifier trained on 2 examples per class
  - $\theta_4$: classifier trained on 4 examples per class
  - etc.
- **Purpose:** These are the "bad" models we want to improve

#### **Network 3: Many-Shot Model $\theta^*$**
- **What it is:** Base classifier trained on ALL available examples per class
- **Parameters:** Same architecture as Network 1, but trained with full dataset
- **Purpose:** This is the "gold standard" target - what we wish we had for tail classes

#### **Network 4: MetaModelNet $\mathcal{F}(·; w)$**
- **What it is:** A separate neural network that operates on model parameters (not images!)
- **Parameters:** $w$ (weights of the meta-network - completely separate from $\theta$)
- **Input:** Few-shot model parameters $\theta_k$ (e.g., a 4096-dim weight vector)
- **Output:** Predicted many-shot parameters $\hat{\theta}^*$ 
- **Purpose:** Learn to transform bad models → good models

### Key Distinction: Parameters vs Meta-Parameters

```
┌─────────────────────────────────────────────────┐
│ Base Classifier Network g(·; θ)                 │
│                                                  │
│ Input: Image x (e.g., 224×224×3)               │
│ Output: Class prediction (e.g., "living room") │
│ Parameters: θ ∈ ℝ^d (e.g., d=4096 for FC layer)│
└─────────────────────────────────────────────────┘
                      ↓
         These θ become the DATA for...
                      ↓
┌─────────────────────────────────────────────────┐
│ MetaModelNet F(·; w)                            │
│                                                  │
│ Input: Model parameters θ_k ∈ ℝ^d              │
│ Output: Transformed parameters θ̂* ∈ ℝ^d        │
│ Meta-Parameters: w (weights of F)               │
└─────────────────────────────────────────────────┘
```

### Main Idea
Instead of resampling data, transfer **meta-knowledge** about how models learn from head to tail classes. Specifically, learn the *trajectory* of model parameters as sample size increases.

### Three Key Components

#### 1. **Model Dynamics Learning**
Learn a meta-network $\mathcal{F}$ that predicts how few-shot model parameters $\theta_k$ (trained on $k$ examples) transform into many-shot parameters $\theta^*$ (trained on large datasets).

**Loss function:**
$\sum_{\theta \in k\text{-Shot}(\mathcal{H}_t)} \left[ \|\mathcal{F}(\theta; w) - \theta^*\|^2 + \lambda \sum_{(x,y) \in \mathcal{H}_t} \text{loss}_g(x; \mathcal{F}(\theta; w), y) \right]$

where:
- First term: regression loss (predict many-shot from few-shot)
- Second term: task performance loss (maintain accuracy)
- $\mathcal{H}_t$: head classes with $>t$ training examples
- **Important:** $\mathcal{F}(\theta; w)$ takes few-shot weights $\theta$ and outputs predicted many-shot weights

## Complete Training Pipeline: Step-by-Step

### Phase 1: Train Base Classifiers (No Meta-Learning Yet)

For **head classes only** (classes with lots of data):

**Step 1a: Train Many-Shot Models $\theta^*$**

For each head class $c \in \{1, 2, ..., C_{\text{head}}\}$:
- Use training set $\mathcal{D}_c = \{(x_i, y_i)\}_{i=1}^{N_c}$ where $N_c$ is large (100-1000+ examples)
- Optimize: $\theta^*_c = \arg\min_{\theta} \sum_{(x,y) \in \mathcal{D}_c} \ell(g(x; \theta), y)$
- Result: $\theta^*_c \in \mathbb{R}^d$ (the "gold standard" weights)

**Step 1b: Train Few-Shot Models $\theta_k$ for multiple k values**

For each head class $c$ and each $k \in \{1, 2, 4, 8, 16, 32, 64\}$:
- Randomly sample subset $\mathcal{D}_c^k \subset \mathcal{D}_c$ with exactly $k$ examples
- Optimize: $\theta^k_c = \arg\min_{\theta} \sum_{(x,y) \in \mathcal{D}_c^k} \ell(g(x; \theta), y)$
- Generate $S$ random samples (e.g., $S=1000$ for $k=1$, $S=200$ for $k=64$)
- Result: Multiple $\theta^k_{c,s}$ for $s \in \{1, ..., S\}$

**After Phase 1, you have:**
$\{\theta^*_1, \theta^*_2, ..., \theta^*_{C_{\text{head}}}\} \quad \text{(many-shot weights)}$
$\{\theta^k_{c,s} : c \in [C_{\text{head}}], k \in \{1,2,4,...,64\}, s \in [S]\} \quad \text{(few-shot weights)}$

### Phase 2: Train MetaModelNet (Recursive Training)

**MetaModelNet structure:** Chain of residual blocks
$\theta_1 \xrightarrow{f(\cdot; w_0)} \hat{\theta}_2 \xrightarrow{f(\cdot; w_1)} \hat{\theta}_4 \xrightarrow{f(\cdot; w_2)} \hat{\theta}_8 \xrightarrow{\cdots} \hat{\theta}^*$

Each block $i$ implements: $\mathcal{F}_i(\theta) = \mathcal{F}_{i+1}(\theta + f(\theta; w_i))$

**Training Order: BACK TO FRONT (largest k first)**

This means training the **residual blocks of MetaModelNet** from last to first, NOT the layers of ResNet!

**Iteration $N$ (Last Block): Handle $2^N$-shot $\rightarrow$ many-shot**

Threshold: $t = 2^{N+1}$, Training samples: $k = 2^N = t/2$

Select head classes: $\mathcal{C}_N = \{c : |\mathcal{D}_c| \geq t\}$

Objective for block $N$:
$\min_{w_N} \sum_{c \in \mathcal{C}_N} \sum_{s=1}^{S} \left[ \|\mathcal{F}_N(\theta^k_{c,s}; w_N) - \theta^*_c\|^2 + \lambda \sum_{(x,y) \in \mathcal{D}_c} \ell(g(x; \mathcal{F}_N(\theta^k_{c,s}; w_N)), y) \right]$

Since this is the last block: $\mathcal{F}_N(\theta; w_N) = \theta + f(\theta; w_N)$ (nearly identity)

**Iteration $N-1$: Handle $2^{N-1}$-shot $\rightarrow$ many-shot**

Threshold: $t = 2^N$, Training samples: $k = 2^{N-1}$

Select head classes: $\mathcal{C}_{N-1} = \{c : |\mathcal{D}_c| \geq t\}$ (larger set than before)

Now block $N-1$ feeds into already-trained block $N$:
$\mathcal{F}_{N-1}(\theta; w_{N-1}, w_N) = \mathcal{F}_N(\theta + f(\theta; w_{N-1}); w_N)$

Multi-task objective (train $w_{N-1}$, fine-tune $w_N$):
$\min_{w_{N-1}, w_N} \sum_{c \in \mathcal{C}_{N-1}} \sum_{k \in \{2^{N-1}, 2^N\}} \sum_{s} \mathcal{L}(\theta^k_{c,s}, \theta^*_c; w_{N-1}, w_N)$

where $\mathcal{L}(\theta_k, \theta^*; w) = \|\mathcal{F}(\theta_k; w) - \theta^*\|^2 + \lambda \sum_{(x,y)} \ell(g(x; \mathcal{F}(\theta_k; w)), y)$

**Continue recursively:** Iteration $i = N-2, N-3, ..., 1, 0$

At iteration $i$:
- Threshold $t = 2^{i+1}$
- Train $k = 2^i$-shot $\rightarrow$ many-shot mapping
- Multi-task across all $k \in \{2^i, 2^{i+1}, ..., 2^N\}$
- Update $w_i$, fine-tune $\{w_{i+1}, ..., w_N\}$

---

### 📘 **Clarification: "Back-to-Front" Means MetaModelNet Blocks, NOT ResNet Layers**

**Common Confusion:** Does "back-to-front" mean training ResNet from output layer to input layer?

**Answer: NO!** The base ResNet is trained normally (front-to-back via standard backpropagation).

**What "back-to-front" actually means:**
- Training the **residual blocks of MetaModelNet** in reverse order
- Block $N$ first (handles 64-shot → many-shot, easiest task)
- Block $0$ last (handles 1-shot → many-shot, hardest task)

**Visual Clarification:**

```
MetaModelNet Structure:
Block 0 → Block 1 → Block 2 → ... → Block N
(1→2)   (2→4)     (4→8)            (64→∞)

Training Order:
Step 1: Train Block N ←————————————— Start here (easiest)
Step 2: Train Block N-1, finetune N
Step 3: Train Block N-2, finetune N-1, N
  ...
Step N: Train Block 0, finetune all others ← End here (hardest)
```

**Why this order?**
1. **Easy to hard:** Last block transformation is nearly identity (64-shot already good)
2. **Curriculum learning:** Build up from well-trained models to poorly-trained models
3. **Stability:** Earlier blocks benefit from having later blocks already trained
4. **Compositionality:** Block 0 learns $1 \to 2$, then relies on Blocks $1...N$ for $2 \to \infty$

---

### Phase 3: Apply to Tail Classes (Inference)

For each **tail class** $c$ (with only a few examples):

**Step 3a: Train few-shot model**
```
Count examples n_c for tail class c
Train base classifier θ^{n_c}_c on these examples
```

**Step 3b: Find appropriate entry point**
```
If n_c = 3, use 2-shot pathway (k=2^1)
  → Feed θ^3_c into Block 1
  
If n_c = 10, use 8-shot pathway (k=2^3)  
  → Feed θ^10_c into Block 3

Output: θ̂*_c = transformed weights
```

**Step 3c: Replace weights**
```
Use θ̂*_c instead of θ^{n_c}_c for final classifier
```

---

### 🎯 **How MetaModelNet is Applied After Training (Detailed)**

After training is complete, you have:
1. ✅ Trained MetaModelNet with weights $w = \{w_0, w_1, ..., w_N\}$
2. ✅ All head classes already have good models (trained on abundant data)
3. ❓ Tail classes have poor models (trained on scarce data) - **Need improvement!**

#### **Application Process: Step-by-Step**

**Scenario:** Tail class "library" with only $n = 5$ training images.

**Step 1: Train the base few-shot model**

Optimize standard classification loss:
$\theta_5^{\text{library}} = \arg\min_{\theta} \sum_{i=1}^{5} \ell(g(x_i; \theta), y_i)$

where $y_i = \text{"library"}$ for all $i$, and $\theta_5^{\text{library}} \in \mathbb{R}^{2048}$ (e.g., final FC layer)

This model is **WEAK** - overfits to these 5 specific images.

**Step 2: Determine which MetaModelNet block to use**

Number of examples: $n = 5$

Find closest power of 2: $k = 2^{\lfloor \log_2(5) \rfloor} = 2^2 = 4$

Block index: $i^* = 2$ (use Block 2, trained for 4-shot $\to$ many-shot)

**Step 3: Feed through MetaModelNet starting at Block 2**

Initialize: $\theta^{(2)} = \theta_5^{\text{library}}$

Sequential transformation through residual blocks:
$\begin{align}
\theta^{(3)} &= \theta^{(2)} + f(\theta^{(2)}; w_2) \\
\theta^{(4)} &= \theta^{(3)} + f(\theta^{(3)}; w_3) \\
\theta^{(5)} &= \theta^{(4)} + f(\theta^{(4)}; w_4) \\
&\vdots \\
\theta^{(N+1)} &= \theta^{(N)} + f(\theta^{(N)}; w_N) \\
\hat{\theta}^*_{\text{library}} &= \theta^{(N+1)}
\end{align}$

Visual flow:
$\theta_5^{\text{library}} \xrightarrow{\text{Skip 0,1}} \boxed{\text{Block 2}} \xrightarrow{+f} \boxed{\text{Block 3}} \xrightarrow{+f} \cdots \xrightarrow{+f} \boxed{\text{Block N}} \rightarrow \hat{\theta}^*_{\text{library}}$

**Step 4: Replace the classifier weights**

Original weak classifier: $g_{\text{weak}}(x) = g(x; \theta_5^{\text{library}})$

Improved classifier: $g_{\text{improved}}(x) = g(x; \hat{\theta}^*_{\text{library}})$

Performance comparison:
$\begin{align}
\text{Accuracy}_{\text{weak}} &= \frac{1}{|\mathcal{T}|} \sum_{(x,y) \in \mathcal{T}} \mathbb{1}[g_{\text{weak}}(x) = y] \approx 0.35 \\
\text{Accuracy}_{\text{improved}} &= \frac{1}{|\mathcal{T}|} \sum_{(x,y) \in \mathcal{T}} \mathbb{1}[g_{\text{improved}}(x) = y] \approx 0.58
\end{align}$

where $\mathcal{T}$ is the test set.

#### **Complete Inference Algorithm**

**Input:** Few-shot weights $\theta_k$ from tail class with $k$ examples

**Output:** Predicted many-shot weights $\hat{\theta}^*$

**Algorithm:**
$\begin{align}
&\text{1. Find entry block: } i^* = \lfloor \log_2(k) \rfloor \\
&\text{2. Initialize: } \theta^{(i^*)} \leftarrow \theta_k \\
&\text{3. For } i = i^*, i^*+1, ..., N: \\
&\qquad \theta^{(i+1)} \leftarrow \theta^{(i)} + f(\theta^{(i)}; w_i) \\
&\text{4. Return: } \hat{\theta}^* = \theta^{(N+1)}
\end{align}$

**Usage for all tail classes:**

For each tail class $c \in \mathcal{C}_{\text{tail}}$:
$\begin{align}
&n_c = |\mathcal{D}_c| \quad \text{(count examples)} \\
&\theta^{n_c}_c = \text{train\_base\_classifier}(\mathcal{D}_c) \\
&\hat{\theta}^*_c = \text{apply\_metamodelnet}(\theta^{n_c}_c, n_c, \{w_0, ..., w_N\}) \\
&g_c(x) = g(x; \hat{\theta}^*_c) \quad \text{(final classifier)}
\end{align}$

#### **Why This Works: Mathematical Intuition**

**Before MetaModelNet:**

Few-shot model learns spurious correlations:
$\theta_5^{\text{library}} = \arg\min_{\theta} \sum_{i=1}^{5} \ell(g(x_i; \theta), y) \quad \Rightarrow \quad \text{Overfit to specific features}$

Example: If all 5 images have brown books, model learns $\theta$ such that:
$g(x; \theta_5) \approx \mathbb{1}[\text{color}(x) = \text{brown}]$

**After MetaModelNet:**

Transformation extrapolates to general pattern:
$\hat{\theta}^* = \mathcal{F}(\theta_5; w) \quad \Rightarrow \quad \text{Captures general "library" concept}$

The meta-network learned from head classes that few-shot $\to$ many-shot transformations involve:
- Increasing weight magnitudes: $\|\hat{\theta}^*\| > \|\theta_k\|$
- Reducing sensitivity to spurious features
- Amplifying features common across diverse examples

This is formalized through the meta-learning objective:
$w^* = \arg\min_{w} \sum_{c \in \mathcal{C}_{\text{head}}} \sum_{k} \|\mathcal{F}(\theta^k_c; w) - \theta^*_c\|^2$

which ensures $\mathcal{F}$ learns a universal transformation applicable to new classes.

#### **Entry Point Selection Rules**

| $n_c$ | $k = 2^{\lfloor \log_2(n_c) \rfloor}$ | Block $i^*$ | Transformation Intensity |
|-------|----------------------------------------|-------------|--------------------------|
| 1 | 1 | 0 | Maximum (extreme few-shot) |
| 2-3 | 2 | 1 | Very high |
| 4-7 | 4 | 2 | High |
| 8-15 | 8 | 3 | Moderate |
| 16-31 | 16 | 4 | Low |
| 32-63 | 32 | 5 | Minimal |
| 64+ | 64 | 6 | Nearly identity |

**Key principle:** Entry point at block $i^* = \lfloor \log_2(n_c) \rfloor$ ensures the transformation matches the training regime of that block. Earlier blocks apply more aggressive extrapolation, later blocks apply gentle refinement.

---
## Progressive Transfer with Residual Blocks

Rather than learning a single transformation, build a chain of residual blocks where each block handles a specific sample-size regime:

$\mathcal{F}_i(\theta) = \mathcal{F}_{i+1}(\theta + f(\theta; w_i))$

This creates transformations for: 1-shot → 2-shot → 4-shot → 8-shot → ... → many-shot

**Benefits:**
- Identity regularization: $\mathcal{F}_i \rightarrow I$ as $i \rightarrow \infty$ (for large sample sizes, no transformation needed)
- Compositionality: each block builds on previous ones
- Captures smooth dynamics across sample sizes

### Training Order: Why Back-to-Front?

**Intuition:** Easier tasks first, harder tasks later

**Last block (64-shot → many-shot):**
- Nearly identity mapping (already well-trained)
- Easy to learn
- Many training examples available

**First block (1-shot → many-shot):**  
- Dramatic transformation needed
- Hard to learn
- But can leverage all previously learned blocks
- Effectively learns: 1→2→4→8→...→many (composition)
Train blocks from back-to-front:
1. Start with last block (handles classes with most data)
2. Train with threshold $t = 2^{i+1}$, regressing $k = 2^i$-shot → many-shot
3. Move to next block with smaller threshold
4. Fine-tune all subsequent blocks in multi-task manner

This progressively transfers knowledge from data-rich to data-poor regimes.

## Architecture Details

**MetaModelNet Structure:**
```
Input: k-shot model θ_k
│
├─ Residual Block 0 (1-shot → 2-shot)
├─ Residual Block 1 (2-shot → 4-shot)
├─ Residual Block 2 (4-shot → 8-shot)
│  ...
└─ Residual Block N (many-shot)
│
Output: θ* (predicted many-shot model)
```

Each residual block:
- Batch Normalization
- Leaky ReLU activation
- Fully-connected layer
- Skip connection (ensures identity mapping for similar inputs)

## What the Meta-Network Learns

**Implicit Data Augmentation:** The network learns class-specific transformations that capture how parameters should change - effectively predicting the impact of augmentations without explicitly generating data.

**Class-Specific but Smooth:** Similar classes (e.g., "iceberg" and "mountain") have similar model parameters and transform similarly, while dissimilar classes transform differently.

**General Patterns:**
- Many-shot models have larger parameter magnitudes (higher confidence)
- Transformations capture domain-specific invariances

## Results

### SUN-397 Scene Classification
| Method | Accuracy |
|--------|----------|
| Plain baseline | 48.03% |
| Over-sampling | 52.61% |
| Under-sampling | 51.72% |
| Cost-sensitive | 52.37% |
| **MetaModelNet** | **57.34%** |

**Improvement:** +4.73% over best baseline, +9.31% over plain training

### Large-Scale Datasets
- **Places-205** (long-tail): 23.53% → 30.71% (+7.18%)
- **ImageNet-200** (merged classes): 68.85% → 73.46% (+4.61%)

## Actionable Lessons

### 1. **Rethink Transfer Learning for Imbalanced Data**
Don't just pre-train and fine-tune. Learn *how models evolve* with data, then hallucinate that evolution for rare classes.

### 2. **Logarithmic Sample Size Discretization**
Recognition accuracy improves logarithmically with data: design systems around 1-shot, 2-shot, 4-shot, 8-shot, ... rather than linear increments.

### 3. **Progressive Knowledge Transfer**
Transfer knowledge gradually from data-rich to data-poor regimes using curriculum learning principles. Don't force a single transformation.

### 4. **Identity Regularization is Critical**
For large sample sizes, transformations should approach identity. Use residual connections to enforce this - prevents harmful transformations for well-trained models.

### 5. **Model Space is Smoother Than Data Space**
Working in parameter space reveals smooth structure: similar tasks have similar parameters and transform similarly. This smoothness enables generalization.

### 6. **Joint Feature + Classifier Dynamics**
Don't freeze representations - progressively fine-tune features while learning classifier dynamics for best results (58.74% vs 54.99% with frozen features).

### 7. **Class-Specific Augmentation**
Different classes benefit from different augmentations. Meta-learning can discover these automatically rather than applying uniform strategies.

## Practical Implementation Steps

1. **Split dataset by sample size** into head/tail (threshold at median or percentiles)
2. **Train many-shot models** on head classes with full data
3. **Train k-shot models** on head classes with subsampled data (k = 1, 2, 4, 8, ...)
4. **Train MetaModelNet** recursively from largest to smallest k
5. **Apply to tail classes**: feed their k-shot models through appropriate residual blocks
6. **Fine-tune** entire network if computationally feasible

## Key Takeaway

**The meta-learning principle:** Don't learn to classify directly from limited data. Instead, learn *how to learn* from abundant data, then transfer that learning process to scarce data scenarios. This shifts the problem from "classifying with few examples" to "predicting parameter evolution with few examples" - a more tractable problem.