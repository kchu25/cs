@def title = "Regularizing Final vs First Layer Embeddings"
@def published = "16 November 2025"
@def tags = ["machine-learning"]

# Regularizing Final vs First Layer Embeddings

## The Claim

**Yes, this is generally true** — regularizing final layer embeddings is typically easier and more effective than regularizing first layer embeddings in neural networks.

## Why This Is True

### 1. **Semantic Coherence**

Final layer embeddings exist in a **semantically meaningful space** where:
- Similar inputs have been mapped to nearby representations
- The network has learned task-relevant features
- Distances and directions carry interpretable meaning

First layer embeddings, by contrast, are often arbitrary initial representations with less inherent structure.

### 2. **Gradient Flow and Training Dynamics**

**Final layers** receive clearer training signals:
- Direct gradients from the loss function
- Regularization penalties directly influence the optimization
- Faster convergence to regularized solutions

**First layers** face:
- Diluted gradients through many layers (vanishing gradient problem)
- Regularization effects are indirect and weaker
- Slower adaptation to regularization constraints

### 3. **Dimensionality and Manifold Structure**

The transformation through the network typically involves:

$$\text{Input} \xrightarrow{\text{many layers}} \text{Low-dim manifold} \xrightarrow{\text{final layers}} \text{Output}$$

Final embeddings often lie on a **lower-dimensional manifold** that's easier to regularize. Common techniques like:
- L2 normalization: $\mathbf{z} \leftarrow \frac{\mathbf{z}}{\|\mathbf{z}\|}$
- Hypersphere constraints
- Contrastive losses

work well because the meaningful structure has already been extracted.

## Mathematical Intuition

Consider a loss function with regularization:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{R}(\mathbf{h})$$

where $\mathbf{h}$ is an embedding layer.

### For Final Layer $\mathbf{h}_L$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_L} = \frac{\partial \mathcal{L}_{\text{task}}}{\partial \mathbf{h}_L} + \lambda \frac{\partial \mathcal{R}}{\partial \mathbf{h}_L}$$

Both terms are **direct and strong**.

### For First Layer $\mathbf{h}_1$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}_{\text{task}}}{\partial \mathbf{h}_L} \cdot \frac{\partial \mathbf{h}_L}{\partial \mathbf{h}_{L-1}} \cdots \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1} + \lambda \frac{\partial \mathcal{R}}{\partial \mathbf{h}_1}$$

The regularization gradient is **diluted** through the chain rule.

## Practical Examples

### Effective Final Layer Regularization:
- **Face recognition**: Cosine similarity on final embeddings
- **Metric learning**: Triplet loss with normalized embeddings
- **Contrastive learning**: SimCLR, MoCo use final layer regularization

### Challenges with First Layer Regularization:
- Word embeddings in NLP: Often need pre-training or careful initialization
- Image inputs: Raw pixel regularization is less meaningful
- Requires architectural considerations (e.g., ResNets) to propagate regularization effects

---

> ### 📝 Side Note: Why L1 Regularization Often Fails on First Layers
>
> **The Problem:** Applying L1 regularization to first layer embeddings:
>
> $$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \|\mathbf{h}_1\|_1$$
>
> often **fails to induce sparsity**. Here's why:
>
> #### 1. **Gradient Competition (When λ is Small)**
>
> The gradient for first layer weights $\mathbf{W}_1$:
>
> $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \underbrace{\frac{\partial \mathcal{L}_{\text{task}}}{\partial \mathbf{W}_1}}_{\text{task gradient}} + \underbrace{\lambda \cdot \text{sign}(\mathbf{h}_1) \cdot \frac{\partial \mathbf{h}_1}{\partial \mathbf{W}_1}}_{\text{sparsity gradient}}$
>
> When λ is too small, the **task gradient dominates**:
> - It carries information from the final loss through all layers
> - It's typically much larger in magnitude
> - L1 penalty contributes only a constant (±λ) per parameter
> - Task gradients can be orders of magnitude larger
> - **Result**: Network ignores sparsity, trains normally but isn't sparse
>
> #### 1b. **Gradient Conflict (When λ is Large)**
>
> When λ is too large, you see a different failure: **embeddings become unoptimizable**:
>
> $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \text{small varied task signal} + \text{large constant sparsity signal}$
>
> **What happens:**
> - The sparsity gradient is **uniform and strong**: always pushes toward zero
> - The task gradient is **diverse and weak**: different directions for different inputs
> - L1 "drowns out" the task signal, like noise overwhelming a weak radio signal
> - Updates become dominated by: $\Delta \mathbf{W}_1 \approx -\eta \lambda \cdot \text{sign}(\mathbf{h}_1)$
>
> **Why training fails:**
> - All embeddings get pushed toward zero uniformly
> - Network loses capacity to represent different inputs differently
> - Task loss stops decreasing or increases
> - Gradients become uninformative (everything is being zeroed)
> - Network is "trapped" - can't escape because any non-zero value gets immediately penalized
>
> **This is specific to first layers because:**
> - Task gradients are already weak (diluted through backprop)
> - The uniform sparsity pressure has nothing to balance against
> - First layers need to preserve **diversity** of representations
> - For final layers: uniform pressure toward zero can be okay (feature selection)
> - For first layers: uniform pressure toward zero destroys information flow
>
> #### Why Final Layers Don't "Drown Out" as Easily
>
> Even with the same large λ, final layers remain trainable:
>
> $\Delta \mathbf{W}_L = -\eta\left(\frac{\partial \mathcal{L}_{\text{task}}}{\partial \mathbf{W}_L} + \lambda \cdot \text{sign}(\mathbf{h}_L) \cdot \frac{\partial \mathbf{h}_L}{\partial \mathbf{W}_L}\right)$
>
> **Key differences:**
>
> **1. Task gradients are stronger (no dilution)**
> - $\frac{\partial \mathcal{L}_{\text{task}}}{\partial \mathbf{W}_L}$ is direct from the loss
> - Magnitude is comparable to or larger than $\lambda \cdot \text{sign}(\mathbf{h}_L)$
> - Can actually compete with sparsity pressure
>
> **2. Task gradient is also more uniform**
> - Final layer features have converged to stable, task-relevant representations
> - Less "diversity" needed - the network has already decided what matters
> - Task gradient has **consensus**: "these features are important, keep them"
> - Sparsity gradient: "make everything zero"
> - When they conflict on important features, task wins. When they agree on unimportant features, sparsity wins.
>
> **3. Partial sparsity is acceptable**
> - If L1 zeros out 50% of final features, the network can still function
> - Remaining features carry the essential information
> - For first layers: zeroing out 50% of raw input information is catastrophic
>
> **4. Natural "selection" equilibrium**
> - Important features develop large task gradients that resist L1
> - Unimportant features have weak task gradients, get zeroed by L1
> - A stable equilibrium emerges: sparse but functional
> - For first layers: all raw features might be "important" for some inputs, no clear selection
>
> **Analogy:**
> - **First layer L1**: Like trying to have a conversation (weak signal) next to a jackhammer (uniform noise)
> - **Final layer L1**: Like a negotiation between two parties of similar strength - they reach a compromise
>
> #### 2. **Later Layers Can Compensate**
>
> Even if $\mathbf{h}_1$ becomes sparse, subsequent layers compensate:
>
> $$\mathbf{h}_2 = f(\mathbf{W}_2 \mathbf{h}_1)$$
>
> The network learns larger weights in $\mathbf{W}_2$ to amplify remaining features and maintain expressiveness—the network's adaptive capacity works against your regularization goal.
>
> #### 3. **Vanishing Sparsity Signal**
>
> The chain rule dilutes the sparsity signal:
>
> $$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L} \prod_{i=2}^{L} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} + \lambda \cdot \text{sign}(\mathbf{h}_1)$$
>
> - The product of Jacobians can be very small (vanishing gradients)
> - The constant $\lambda$ term fights against exponentially larger task gradients
> - Even with careful $\lambda$ tuning, the balance is unstable
>
> #### 4. **Optimization Landscape Issues**
>
> L1's non-smoothness at zero causes problems:
> - SGD oscillates around zero without reaching it
> - Momentum prevents embeddings from becoming exactly sparse
> - Network finds local minima where features are small but non-zero
>
> #### 5. **Information Bottleneck Conflict**
>
> First layers must encode all information needed downstream. Sparsity reduces capacity, creating a bottleneck that conflicts with task performance.
>
> #### Why Final Layer L1 Works Better
>
> For final embeddings $\mathbf{h}_L$, the gradient looks similar:
>
> $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_L} = \frac{\partial \mathcal{L}_{\text{task}}}{\partial \mathbf{W}_L} + \lambda \cdot \text{sign}(\mathbf{h}_L) \cdot \frac{\partial \mathbf{h}_L}{\partial \mathbf{W}_L}$$
>
> **So why does it work here?** The crucial differences:
>
> **1. Gradient Magnitudes are Comparable**
> - Task gradient $\frac{\partial \mathcal{L}_{\text{task}}}{\partial \mathbf{W}_L}$ is direct (no chain through many layers)
> - It's typically **smaller** than first layer task gradients because:
>   - No accumulation through backprop
>   - Final features are often already refined and stable
> - The sparsity penalty λ can actually compete on equal footing
>
> **2. Task and Sparsity Can Align**
> - Final layer often performs feature selection naturally
> - Sparse representations **help** the task (remove noise, improve generalization)
> - The network "wants" to ignore irrelevant features
> - L1 pushes in a direction the task already favors
>
> **3. No Downstream Compensation**
> - If $\mathbf{h}_L$ becomes sparse, there are no more layers to "undo" it
> - The output layer directly uses the sparse representation
> - Sparsity is enforced all the way to the prediction
>
> **4. Information Processing is Complete**
> - By the final layer, the network has already extracted what it needs
> - Sparsity doesn't create a bottleneck—it prunes **redundancy**
> - For first layers: sparsity removes raw information (bad)
> - For final layers: sparsity removes processed redundancy (good)
>
> **5. Smaller Effective Dimensionality**
> - Final embeddings often live in lower-dimensional manifolds
> - Fewer dimensions → each dimension matters more
> - Sparsity gradient per dimension has larger relative impact
>
> #### Practical Solutions for First Layer Sparsity
>
> If you really need first-layer sparsity:
> 1. **Structured sparsity**: Group Lasso to zero out entire feature groups
> 2. **Explicit gating**: Learnable binary masks with straight-through estimators
> 3. **Pruning post-training**: Train dense, then prune and fine-tune
> 4. **Architecture changes**: Build sparsity into the architecture (e.g., sparse attention)
> 5. **Much larger λ**: Dramatically increase penalty (but risk underfitting)

---

## Key Takeaway

The neural network progressively **distills structure** from data. By the final layers, this structure is refined and task-aligned, making it responsive to regularization. Early layers still process raw, high-dimensional, unstructured data where regularization constraints are harder to enforce and less meaningful.

**For L1 sparsity specifically**: First layers face overwhelming task gradients, adaptive compensation by later layers, and vanishing regularization signals—making sparse regularization ineffective without architectural or algorithmic interventions.