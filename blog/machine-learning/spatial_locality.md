@def title = "Spatial Locality as Inductive Bias"
@def published = "1 November 2025"
@def tags = ["machine-learning"]


# Spatial Locality as Inductive Bias

## The Central Question: What Makes CNNs Special?

When we talk about convolutional neural networks, we're really talking about a specific architectural choice that encodes a fundamental assumption: **nearby things matter more than distant things**. This is **spatial locality**, and it's one of the most powerful inductive biases in deep learning.

## What is Inductive Bias?

**Inductive bias** refers to the assumptions a learning algorithm makes about the structure of the problem. These assumptions restrict the hypothesis space—they tell the model "look for solutions of this kind, not that kind."

Without inductive bias, learning from limited data is nearly impossible. The bias guides the model toward solutions that match our understanding of the problem domain.

## Translation Equivariance: A Global Structural Constraint

Translation equivariance is indeed an inductive bias. Mathematically, a function $f$ is **translation equivariant** if:

$$f(T_\delta(x)) = T_\delta(f(x))$$

where $T_\delta$ is a spatial translation operation.

What's interesting here is that while translation equivariance operates on local features (edges, textures), it represents a **global constraint** on the model's behavior. It says: "the rules are the same everywhere in space."

### Comparing Degrees of "Globalness"

There's actually a spectrum:

1. **Feedforward/fully-connected layers**: Most local (position-specific)
   - Each position gets its own independent parameters
   - No spatial structure at all

2. **Convolutional layers**: Spatially consistent (translation equivariant)
   - Same rules applied everywhere via parameter sharing
   - Encodes global symmetry assumption

3. **Global pooling/full attention**: Truly global
   - Aggregates information from all positions
   - Position-independent operations

So yes, translation equivariance captures something "more global" than feedforward—it's a global constraint about spatial homogeneity.

---

## The Mechanisms: How CNNs Encode Spatial Locality

### 1. Parameter Sharing

This is the key mechanism that implements translation equivariance. Instead of learning different weights for each spatial position (like fully-connected layers), CNNs use the same filter weights at every location.

**Example:**
- **Fully-connected**: processing a 100×100 image means different weights for position (1,1), (1,2), (50,50)—each unique
- **Convolutional**: one 3×3 filter with 9 weights applied everywhere

Parameter sharing is a global architectural constraint that dramatically reduces parameters and enforces: "whatever patterns are important here are also important there."

### 2. Local Receptive Fields

The **receptive field** is the region of input that influences a particular output. This is absent in standard feedforward layers applied position-wise.

- Single conv layer: small receptive field (e.g., 3×3)
- Stacked conv layers: receptive field grows hierarchically
- Deep networks: can span entire input

This captures spatial locality bias: nearby pixels tend to form meaningful patterns.

### 3. Hierarchical Composition

By stacking layers:
- **Low levels** capture simple local patterns (edges, colors)
- **Middle levels** combine into textures, parts
- **High levels** form objects, scenes

---

## Mathematical Deep Dive: Receptive Field Calculation

### Definition

The **receptive field** of a neuron at layer $l$ is the spatial region in the input image that can influence its activation. Mathematically, for a neuron at position $(i, j)$ in layer $l$, the receptive field $RF_l$ defines the set of input positions:

$$RF_l = \{(x, y) \in \text{Input} \mid \text{Input}(x,y) \text{ affects Output}_l(i,j)\}$$

### Recursive Formula for Receptive Field Size

For a sequence of convolutional layers, we can calculate the receptive field size recursively. Let:

- $r_l$ = receptive field size at layer $l$
- $k_l$ = kernel size at layer $l$
- $s_l$ = stride at layer $l$
- $d_l$ = dilation at layer $l$

The **effective kernel size** with dilation is:

$$k_l^{\text{eff}} = k_l + (k_l - 1)(d_l - 1) = d_l(k_l - 1) + 1$$

The recursive formula for receptive field size:

$$r_l = r_{l-1} + (k_l^{\text{eff}} - 1) \cdot \prod_{i=1}^{l-1} s_i$$

With base case: $r_0 = 1$ (the input itself has receptive field of 1 pixel)

**Simplified form** when all strides are $s$:

$$r_l = r_{l-1} + (k_l^{\text{eff}} - 1) \cdot s^{l-1}$$

### Jump (Stride Product)

The **jump** $j_l$ represents the spacing between adjacent receptive field centers in the input space:

$$j_l = \prod_{i=1}^{l} s_i$$

This tells us how much we move in the input when we move 1 position in the feature map.

### Receptive Field Center Position

For a neuron at position $(i, j)$ in layer $l$, its receptive field center in the input is at:

$$\text{center}_x = \frac{r_l - 1}{2} + i \cdot j_l$$
$$\text{center}_y = \frac{r_l - 1}{2} + j \cdot j_l$$

### Detailed Example: 3-Layer CNN

Consider a network with:
- Layer 1: $k_1 = 3$, $s_1 = 1$, $d_1 = 1$
- Layer 2: $k_2 = 3$, $s_2 = 2$, $d_2 = 1$
- Layer 3: $k_3 = 3$, $s_3 = 1$, $d_3 = 1$

**Layer 0 (input):**
$$r_0 = 1, \quad j_0 = 1$$

**Layer 1:**
$$k_1^{\text{eff}} = 1(3-1) + 1 = 3$$
$$r_1 = 1 + (3-1) \cdot 1 = 3$$
$$j_1 = 1$$

**Layer 2:**
$$k_2^{\text{eff}} = 3$$
$$r_2 = 3 + (3-1) \cdot 1 = 5$$
$$j_2 = 1 \cdot 2 = 2$$

**Layer 3:**
$$k_3^{\text{eff}} = 3$$
$$r_3 = 5 + (3-1) \cdot 2 = 9$$
$$j_3 = 2 \cdot 1 = 2$$

So a neuron in layer 3 sees a **9×9 region** of the input.

### Impact of Dilation

Dilation increases the receptive field without adding parameters. For a single layer with dilation $d$:

$$r_{\text{dilated}} = 1 + (k-1) \cdot d$$

**Example**: $k=3$, $d=2$
$$r = 1 + (3-1) \cdot 2 = 5$$

The 3×3 kernel now covers 5×5 input pixels with gaps.

### Closed Form for Uniform Architecture

For $L$ layers with identical kernel size $k$ and stride $s$:

$$r_L = 1 + (k-1) \sum_{i=0}^{L-1} s^i = 1 + (k-1) \cdot \frac{s^L - 1}{s - 1}$$

When $s=1$ (no striding):
$$r_L = 1 + (k-1) \cdot L$$

**Example**: 10 layers of 3×3, stride 1
$$r_{10} = 1 + (3-1) \cdot 10 = 21$$

### Effective Receptive Field

The **theoretical receptive field** assumes all pixels contribute equally. In practice, the **effective receptive field** (ERF) follows a Gaussian-like distribution—central pixels matter much more.

Empirically, the ERF is approximately:

$$\text{ERF} \approx \sqrt{\frac{2r_L}{L}}$$

This is much smaller than the theoretical RF, especially in deep networks. It explains why:
- Distant pixels contribute negligibly
- Locality bias is stronger than theoretical RF suggests
- Skip connections (ResNets) expand ERF significantly

### Padding Considerations

Padding affects output dimensions but not the receptive field size. With padding $p$:

$$\text{Output size} = \left\lfloor \frac{\text{Input size} + 2p - k}{s} \right\rfloor + 1$$

But the receptive field calculation remains unchanged—it's determined by kernel size, stride, and dilation only.

### Practical Formula Summary

For quick calculation in standard CNNs (no dilation):

$$r_l = r_{l-1} + (k_l - 1) \cdot j_{l-1}$$
$$j_l = j_{l-1} \cdot s_l$$

Initialize with $r_0 = 1$, $j_0 = 1$.

**Python implementation:**
```python
def receptive_field(layers):
    r, j = 1, 1
    for k, s in layers:  # (kernel_size, stride) pairs
        r = r + (k - 1) * j
        j = j * s
    return r, j

# Example: [(3,1), (3,2), (3,1)]
rf, jump = receptive_field([(3,1), (3,2), (3,1)])
print(f"Receptive field: {rf}, Jump: {jump}")  # 9, 2
```

---

## The Quantitative Justification: Why Spatial Locality?

Why do local receptive fields work? Because natural images have quantifiable local structure:

### Statistical Correlation
Nearby pixels are highly correlated:

$$\text{Cov}(I(x), I(x+\delta)) \text{ is high when } |\delta| \text{ is small}$$

Empirically, adjacent pixels often have correlation >0.8, while distant pixels approach independence.

### Mutual Information
$I(X;Y)$ between pixels decays with spatial distance. Local windows capture most of the information.

### Structural Scales
Natural features (edges, textures) have characteristic spatial scales matched by small receptive fields (3×3, 5×5).

**The inductive bias:** the function we're learning has **locality of influence**—outputs at position $i$ depend strongly on inputs near $i$, weakly on distant inputs.

---

## Transformers: A Different Philosophy

Standard Transformers lack translation equivariance for a key reason: **positional encodings**.

Without positional encodings, self-attention is actually **permutation invariant**—it can't distinguish token orderings at all. Positional encodings break this symmetry, making position 1 fundamentally different from position 100.

### The Trade-off

- **CNNs**: Strong inductive bias (spatial locality, translation equivariance) → data-efficient for spatial data
- **Transformers**: Weaker inductive bias → need more data but more flexible, can learn long-range dependencies directly

This is why Vision Transformers originally needed much more training data than CNNs. They lack the built-in spatial locality bias, but with enough data, they can learn approximately equivariant representations if useful.

---

## Beyond Convolutions: Other Operations Encoding Locality

Spatial locality isn't unique to convolutions. Other operations encode similar assumptions:

### Recurrent Networks (RNNs, LSTMs)
- Temporal locality: current state depends on recent past
- Markovian assumption: limited temporal receptive field
- Same principle, different domain (time vs space)

### Graph Neural Networks (GNNs)
- Locality defined by graph structure, not Euclidean distance
- Message passing from immediate neighbors
- Stacking layers expands to k-hop neighborhoods

### Local/Windowed Attention
- Swin Transformers: attention within spatial windows
- Longformer, BigBird: sparse attention patterns
- Hybrid approach: some locality bias + attention flexibility

### Locally Connected Layers
- Local receptive fields WITHOUT parameter sharing
- Spatial locality but no translation equivariance
- Each position learns its own local filter

### The Common Thread

All these operations restrict the computational graph so each output depends on a limited neighborhood of inputs. The definition of "neighborhood" varies:

- **CNNs**: Euclidean spatial distance
- **RNNs**: Temporal proximity
- **GNNs**: Graph connectivity
- **Windowed attention**: Constrained receptive field

The alternative—fully-connected layers or full attention—assumes every input is equally relevant to every output. No locality bias.

---

## Key Takeaway

**Spatial locality** is a powerful inductive bias that says: the world has local structure, and nearby things influence each other more than distant things.

CNNs encode this through:
1. **Parameter sharing** (global constraint: same rules everywhere)
2. **Local receptive fields** (local processing of neighborhoods)
3. **Hierarchical composition** (simple → complex features)

This bias makes CNNs incredibly data-efficient for spatial data, but it's also a constraint. Transformers show us the alternative: weaker assumptions, more data needed, but potentially more flexibility.

**The art of architecture design is choosing the right inductive biases for your problem domain.**

---

> ## Controversial and Underappreciated Aspects of CNNs
> 
> ### 1. Texture Bias vs Shape Bias - The Uncomfortable Truth
> 
> CNNs are heavily **texture-biased**, not shape-biased like humans:
> - ImageNet-trained CNNs often classify based on texture, not object shape
> - A cat-shaped object with elephant texture gets classified as elephant
> - Humans use shape primarily; CNNs learned a "wrong" solution that still works
> - **Why it matters**: Adversarial vulnerability, poor generalization to stylized images
> - **The debate**: Is texture bias a bug or feature? Maybe textures ARE more informative statistically?
> 
> ### 2. Aliasing and the Strided Convolution Problem
> 
> This is getting more attention recently but still underappreciated:
> - **Stride > 1 causes aliasing** - violates Nyquist sampling theorem
> - Downsampling without proper anti-aliasing loses high-frequency information incorrectly
> - **The fix**: Blur before downsampling (BlurPool, Lipschitz constraints)
> - **Controversial**: Most architectures ignore this, yet it improves shift-equivariance significantly
> - Papers like "Making Convolutional Networks Shift-Invariant Again" show huge gains from proper anti-aliasing
> 
> ### 3. The Effective Receptive Field Scandal
> 
> The ERF revelation is actually shocking:
> - Theoretical RF of ResNet-50 final layer: **427×427 pixels**
> - Effective RF (where gradients actually flow): **~70×70 pixels**
> - **The controversy**: Deep networks don't actually use their theoretical receptive fields!
> - Central pixels dominate; peripheral pixels contribute almost nothing
> - Skip connections help but don't solve this completely
> - **Implication**: Maybe we don't need such deep networks? Or we're not training them optimally?
> 
> ### 4. Border Effects and Padding - The Silent Performance Killer
> 
> Zero-padding creates **boundary artifacts** that propagate through networks:
> - Edges of feature maps have different statistics than centers
> - Networks learn to "know" where borders are, breaking true translation equivariance
> - **Controversial solution**: Circular/reflection padding, or crop away borders
> - Most papers ignore this; some claim 10%+ accuracy gains from proper padding strategies
> 
> ### 5. Weight Standardization vs Batch Normalization
> 
> BatchNorm is everywhere, but:
> - **Weight Standardization** (normalizing weights, not activations) can be better
> - Less sensitive to batch size, more stable training
> - **Controversial**: Challenges the BatchNorm hegemony
> - GroupNorm + Weight Standardization often outperforms BatchNorm in small-batch regimes
> 
> ### 6. The Lottery Ticket Hypothesis - Implications
> 
> Finding that **sparse subnetworks** exist from initialization that train to full accuracy:
> - Suggests over-parameterization might be about finding good initialization paths
> - **Controversial view**: Maybe we don't understand what training does at all
> - Are we learning, or searching through random features?
> - Connections to neural tangent kernels and lazy training
> 
> ### 7. Checkerboard Artifacts from Transposed Convolutions
> 
> When doing upsampling (GANs, segmentation):
> - Transposed convolutions with stride create **checkerboard patterns**
> - Caused by uneven pixel overlap
> - **The fix**: Resize + convolution, or careful kernel/stride choices
> - Many papers still use naive transposed convs and get artifacts
> 
> ### 8. Data Augmentation as Regularization - It's Doing More Than You Think
> 
> Modern training relies heavily on augmentation, but:
> - Augmentation changes the **implicit prior** of the model
> - You're not just preventing overfitting; you're encoding invariances
> - **Controversial**: Different augmentations create fundamentally different learned representations
> - MixUp, CutMix change the loss landscape itself - are we still solving the same problem?
> 
> ### 9. The Fourier Bias of Neural Networks
> 
> Networks learn **low-frequency functions first** (spectral bias):
> - High-frequency details are learned slowly or not at all
> - Explains why CNNs struggle with fine-grained textures
> - **Implication**: Maybe explicit frequency representations (Fourier features) should be standard
> - Used in NeRF and other coordinate-based networks, but rare in vision
> 
> ### 10. Group Equivariance - The Obvious Generalization Nobody Uses
> 
> Translation equivariance is just one type of symmetry:
> - **Rotation equivariance**: Objects look the same rotated
> - **Scale equivariance**: Should be resolution-independent
> - **Group Convolutional Networks** exist but are rarely used in practice
> - **The controversy**: Why don't we use stronger geometric priors?
> - Counter-argument: Data augmentation achieves approximate equivariance cheaper
> 
> ### 11. Implicit vs Explicit Bias
> 
> The distinction between:
> - **Explicit bias**: Architecture (convolutions, pooling)
> - **Implicit bias**: What optimization finds (gradient descent's inductive bias)
> - **Controversial finding**: SGD has its own biases independent of architecture
> - Flat minima, margin maximization, feature learning order
> - We might be attributing to architecture what's actually from optimization
> 
> ### 12. The Death of Pooling?
> 
> Max pooling was standard, now:
> - Strided convolutions often work better
> - **Stochastic pooling** was promising but died out
> - **Learnable pooling** (soft attention) rarely used
> - **The debate**: Is pooling a form of lossy compression we shouldn't do?
> - Or does its discrete nature help with robustness?
> 
> ### 13. Coordinate Convolutions - Adding Back Position Information
> 
> CNNs lose absolute position through translation equivariance, but:
> - Adding **coordinate channels** (x, y pixel positions) as extra inputs helps
> - Seems to contradict the equivariance principle
> - **Useful for**: Segmentation, detection, anything needing spatial reasoning
> - **Controversial**: Are we admitting CNNs are missing something fundamental?
> 
> ### 14. The Batch Size - Learning Rate Scaling Mystery
> 
> Linear scaling rule: `lr ∝ batch_size` but:
> - Only works up to a point (critical batch size)
> - Large batch training requires careful warmup, different optimization
> - **Controversial**: Is there a fundamental limit to parallelization?
> - Connects to generalization - large batches reach "sharper" minima
> 
> ### 15. Neural ODEs and Continuous Depth
> 
> Viewing ResNets as discrete ODE solvers:
> - $h_{t+1} = h_t + f(h_t)$ is Euler discretization
> - **Neural ODEs**: Make depth continuous
> - **Controversial**: Changes how we think about depth vs width
> - Adaptive computation depth based on input difficulty
> 
> ---
> 
> ### Most Practically Useful but Overlooked:
> 
> 1. **Proper anti-aliasing** (BlurPool) - easy implementation win
> 2. **Reflection/circular padding** - fixes boundary artifacts  
> 3. **Weight standardization** - better than BatchNorm in some regimes
> 4. **Coordinate convolutions** - helps spatial tasks significantly
> 5. **Careful upsampling** - eliminates checkerboard artifacts
> 
> ### Most Theoretically Important but Ignored:
> 
> 1. **Effective receptive field** being tiny - questions depth benefits
> 2. **Texture bias** - CNNs solve problems differently than humans
> 3. **Implicit bias of optimization** - it's not just architecture
> 4. **Spectral bias** - explains many failure modes