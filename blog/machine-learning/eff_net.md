@def title = "EfficientNet: Compound Scaling & MBConv Blocks - Complete Guide"
@def published = "22 November 2025"
@def tags = ["machine-learning"]

# EfficientNet: Compound Scaling & MBConv Blocks - Complete Guide

## Initial Question: How Are They Related?

Great question! Let me break this down in a conversational way because these concepts are actually quite related, even though they operate at different levels of the network design.

### The Compound Scaling Coefficient

First, let's talk about the **compound scaling coefficient**. In EfficientNet, the authors proposed scaling up networks in a balanced way using a compound coefficient **φ** (phi). Instead of just making networks deeper OR wider OR using higher resolution images (which is what people typically did), they scale all three dimensions together:

- **Depth**: d = α^φ
- **Width**: w = β^φ  
- **Resolution**: r = γ^φ

Where α, β, and γ are constants determined through a grid search, and they must satisfy: **α · β² · γ² ≈ 2**

This constraint means that for every increment of φ by 1, your total computational cost (FLOPs) roughly doubles. The authors found that α = 1.2, β = 1.1, and γ = 1.15 work well.

> **💡 The Insight Behind Compound Scaling**
>
> **Why scale all three dimensions together?** Before EfficientNet, people would scale networks by picking just one dimension—making them deeper, wider, OR using bigger images. But these dimensions are interdependent!
>
> - **Higher resolution images** have more detail → need **more channels** to capture it → need **more layers** to combine features
> - Scaling just one dimension creates imbalanced networks (like a 100-story building that's 1 foot wide!)
>
> **The key insight**: Networks need balanced growth. Just like building a bigger house means more floors (depth), wider floors (width), AND a bigger foundation (resolution).
>
> **Why α · β² · γ² ≈ 2?** This is about computational budget. CNN FLOPs scale as: **FLOPs ∝ depth × width² × resolution²** (squared because convolutions multiply channels and images are 2D). By setting this constraint, each φ increment roughly doubles your FLOPs, giving you a predictable accuracy-efficiency trade-off.
>
> The authors found α = 1.2, β = 1.1, γ = 1.15 through grid search—notice depth grows fastest since adding layers is relatively "cheap" compared to making everything wider or higher-res.

### The MBConv Block Structure

Now, the **MBConv block** (Mobile Inverted Bottleneck Convolution) is the fundamental building block used throughout EfficientNet. Here's its basic structure:

1. **Expansion phase**: 1×1 conv that expands channels by a factor (typically 6)
2. **Depthwise convolution**: 3×3 or 5×5 depthwise conv
3. **Squeeze-and-Excitation**: attention mechanism
4. **Projection phase**: 1×1 conv that projects back down
5. **Skip connection** (if input and output dimensions match)

> **🔍 Width vs. Expansion Ratio**
>
> **Width and expansion are two different things!**
>
> - **Width (scaled by β^φ)**: The number of **input/output channels** of the entire MBConv block
> - **Expansion ratio**: How much you **temporarily expand** channels **inside** the block
>
> **Example with 40-channel input (MBConv6)**:
> ```
> Input: 40 channels (this is the "width")
>    ↓
> 1×1 Conv (expansion): 40 × 6 = 240 channels (expansion ratio = 6)
>    ↓
> Depthwise 3×3: still 240 channels
>    ↓
> Squeeze-and-Excitation: operates on 240 channels
>    ↓
> 1×1 Conv (projection): back to 40 channels
>    ↓
> Output: 40 channels (same "width" as input)
> ```
>
> When you scale width with β^φ:
> - B0: 40 input/output channels, 40 × 6 = 240 internal
> - B1: 44 input/output channels, 44 × 6 = 264 internal
> - The **expansion ratio stays 6**, but absolute numbers change

> **⚙️ Squeeze-and-Excitation Reduction Ratio**
>
> The SE block learns channel attention weights. It has a **reduction ratio r** that controls compression:
>
> ```
> Input: C channels
>    ↓
> Global Average Pool: C → C values (one per channel)
>    ↓
> FC layer 1 (squeeze): C → C/r (where r is the reduction ratio)
>    ↓
> ReLU
>    ↓
> FC layer 2 (excitation): C/r → C
>    ↓
> Sigmoid → Scale original input
> ```
>
> **Why Sigmoid (not Softmax)?**
>
> SE blocks use **Sigmoid** for independent channel-wise attention, not Softmax:
>
> - **Sigmoid**: Each channel gets an independent weight [0, 1]
>   - Channel 1: 0.9 ← "Very important!"
>   - Channel 2: 0.8 ← "Also very important!"
>   - Channel 3: 0.1 ← "Not useful here"
>   - **All channels can be important simultaneously**
>
> - **Softmax**: Channels compete (weights sum to 1)
>   - Channel 1: 0.7 ← "Most important"
>   - Channel 2: 0.25 ← "Second place"
>   - Channel 3: 0.05 ← "Least important"
>   - **If one goes up, others must go down**
>
> **The rationale**: SE blocks do **recalibration, not selection**. Different feature channels (edges, textures, colors) can all be important at the same time. For a brick wall image, you want to emphasize BOTH edges AND textures—Sigmoid allows this, while Softmax would force you to choose.
>
> **Analogy**: Sigmoid is like a music mixer where each instrument has its own volume knob. Softmax is like having a fixed total volume budget where turning up guitar forces you to turn down drums.
>
> Softmax makes sense for mutual exclusion (classification, token selection), but for channel recalibration where multiple features matter simultaneously, Sigmoid is the right choice.
>
> **What should r be?**
> - **r = 4** (EfficientNet default): Best balance between cost and expressiveness
> - **r = 16**: Lighter, faster, but less expressive
> - **r = 2**: More parameters, potentially more expressive but diminishing returns
>
> **Why r = 4?** From the original SE-Net paper, experiments showed:
> - r = 16 was too compressed (lost information)
> - r = 2 added parameters without much benefit  
> - r = 4 hit the sweet spot
>
> **Intuition**: The SE block learns "how much attention to pay to each channel." Smaller r = more complex attention mechanism. Larger r = simpler, faster. **Use r = 4 as default** unless you have extreme computational constraints.

### How They're Related

Here's where it gets interesting! **The compound scaling coefficient determines HOW you scale the MBConv blocks as you go from EfficientNet-B0 to B1, B2, etc.**

When you increase φ:

- **Depth scaling (α^φ)** means you repeat MBConv blocks more times in each stage
- **Width scaling (β^φ)** means you increase the number of channels in each MBConv block
- **Resolution scaling (r^φ)** means you feed in larger input images

So the MBConv block is the **what** (the architectural component), while the compound coefficient is the **how** (the scaling strategy).

### A Concrete Example

Let's say you have a stage in EfficientNet-B0 with:
- 3 MBConv blocks repeated
- 40 output channels
- Input resolution 224×224

When you scale to B1 with φ = 1:
- Depth: 3 → 3 × 1.2¹ ≈ 4 blocks (rounded)
- Width: 40 → 40 × 1.1¹ ≈ 44 channels
- Resolution: 224 → 224 × 1.15¹ ≈ 240

The **structure** of each MBConv block stays the same (expansion → depthwise → SE → projection), but you have more of them, they're wider, and they process larger images.

### Why This Matters

The beauty is that the compound scaling ensures balanced growth. If you only made the network deeper (more MBConv blocks) without making it wider, you'd have a very deep but skinny network. If you only made it wider without increasing resolution, the network might not benefit from all those extra parameters since it's still seeing small images.

By scaling all three together with the compound coefficient, EfficientNet maintains a good balance that empirically works better than scaling any dimension alone.

### Key Takeaway

Think of it this way: **MBConv is your LEGO brick, and the compound scaling coefficient is your recipe for how many bricks to use, how big they should be, and what size base plate you're building on!**

---

## How Do You Decide φ for Each Stage?

Great follow-up question! Let me clarify because there's an important distinction here:

**φ (phi) is a global parameter for the entire network, not something you decide per stage.**

### How φ Works

When you choose a φ value, it scales the **entire network** uniformly. For example:

- **EfficientNet-B0**: φ = 0 (baseline)
- **EfficientNet-B1**: φ = 1
- **EfficientNet-B2**: φ = 2
- **EfficientNet-B3**: φ = 3
- ...and so on up to **B7**: φ = 7

Each increment of φ roughly doubles the computational cost (FLOPs) across the whole network.

### What About Individual Stages?

Here's the key insight: **You don't decide φ per stage. Instead, each stage scales according to the same global φ, but the scaling manifests differently.**

Let me show you with math. If stage *i* in the baseline (B0) has:
- **L_i** layers (depth)
- **C_i** channels (width)

Then in a scaled version with compound coefficient φ:
- **Depth becomes**: L_i × α^φ
- **Width becomes**: C_i × β^φ

Notice that **every stage** uses the same α^φ and β^φ multipliers, but because each stage starts with different baseline values (L_i and C_i), they end up with different absolute numbers.

### Another Concrete Example

Let's say EfficientNet-B0 has:
- **Stage 1**: 1 block, 16 channels
- **Stage 2**: 2 blocks, 24 channels
- **Stage 3**: 2 blocks, 40 channels

When you scale to B1 (φ = 1, with α = 1.2, β = 1.1):
- **Stage 1**: 1 × 1.2 ≈ **1 block**, 16 × 1.1 ≈ **18 channels**
- **Stage 2**: 2 × 1.2 ≈ **2 blocks**, 24 × 1.1 ≈ **26 channels**
- **Stage 3**: 2 × 1.2 ≈ **2 blocks**, 40 × 1.1 ≈ **44 channels**

See how the **same** scaling factors (1.2 and 1.1) apply everywhere, but each stage grows from its own baseline?

### The Real Design Decision

The actual design decision happens when creating the **baseline architecture (B0)**:
1. Through neural architecture search (NAS), they found a good base architecture
2. They determined α, β, γ through a grid search on B0 to find what works best
3. Then they just increment φ to create B1, B2, B3, etc.

So you're not deciding "how much to scale each stage" individually—you're just turning one global knob (φ) and letting it proportionally scale everything!

The elegance is in its simplicity: one parameter controls the whole network's size.

---

## Does φ Affect the Number of Stages?

Excellent question! The short answer is: **No, the compound scaling coefficient φ does NOT change the number of stages.**

### What Stays Fixed

The **architecture's structure** remains constant across all EfficientNet variants (B0 through B7):
- Number of stages: **fixed**
- Which operations go in each stage (e.g., MBConv1, MBConv6): **fixed**
- Kernel sizes (3×3 or 5×5): **fixed**
- Stride patterns: **fixed**

### What Changes with φ

Only these three dimensions scale:
- **Depth (d = α^φ)**: How many blocks you repeat **within** each stage
- **Width (w = β^φ)**: Number of channels in each block
- **Resolution (r = γ^φ)**: Input image size

### Why Not Scale the Number of Stages?

This is actually a deliberate design choice! Here's the reasoning:

1. **Stages have semantic meaning**: Early stages capture low-level features (edges, textures), middle stages capture mid-level features (parts, patterns), and late stages capture high-level features (objects, concepts). This hierarchical structure is fundamental to CNNs.

2. **Stride patterns matter**: Each stage typically has one stride-2 operation that downsamples the spatial resolution. If you added more stages, you'd need to decide where to put strides, which would fundamentally change the feature hierarchy.

3. **Architectural search determined the best structure**: The baseline B0 architecture was found through Neural Architecture Search (NAS). The number of stages (7 in EfficientNet) was optimized as part of that search.

### The EfficientNet Stage Structure

For reference, EfficientNet has this fixed structure:

```
Stage 1: MBConv1, k3×3
Stage 2: MBConv6, k3×3  
Stage 3: MBConv6, k5×5
Stage 4: MBConv6, k3×3
Stage 5: MBConv6, k5×5
Stage 6: MBConv6, k5×5
Stage 7: MBConv6, k3×3
```

When you go from B0 → B7, these stages remain, but each one gets:
- **Deeper** (more repeated blocks within the stage)
- **Wider** (more channels)
- Processing **higher resolution** features

### Think of It This Way

Scaling the number of stages would be like adding more floors to a building's foundation-to-roof hierarchy. Instead, compound scaling makes each existing floor taller (depth), wider (width), and built with higher resolution materials—but the fundamental multi-floor structure stays the same!

---

## Summary: What You Decide vs. What φ Decides

**Exactly! Here's the final clarification:**

### What YOU Decide (Architecture Design)

1. **Number of stages**: You decide this when designing the base architecture (B0)
2. **Which type of MBConv to use in each stage**: MBConv1, MBConv6, etc.
3. **Kernel sizes**: 3×3 or 5×5 for each stage
4. **Baseline depth**: How many blocks in each stage for B0
5. **Baseline width**: How many channels in each stage for B0

In EfficientNet, these were found through Neural Architecture Search (NAS).

### What φ (Compound Coefficient) Determines

Once you have your baseline architecture, φ **only** controls:

1. **How many MBConv blocks to repeat in each stage** (depth scaling: α^φ)
2. **How wide each block is** (width scaling: β^φ)  
3. **Input resolution** (resolution scaling: γ^φ)

### The Formula in Practice

For any stage in your network:

```
Number of blocks = (baseline blocks) × α^φ
Number of channels = (baseline channels) × β^φ
```

So if Stage 3 in B0 has **2 blocks** with **40 channels**, and you set φ = 2:

```
Stage 3 in B2:
- Blocks: 2 × (1.2)² ≈ 2 × 1.44 ≈ 3 blocks
- Channels: 40 × (1.1)² ≈ 40 × 1.21 ≈ 48 channels
```

### The Key Insight

The **architecture** (number of stages, what goes in each) is a design-time decision.

The **compound coefficient φ** is a scale-time decision—it's just a knob you turn to make the whole thing bigger or smaller while keeping the architecture's "shape" intact.

**Think of it like a blueprint for a house**: you design the blueprint once (number of rooms, their purposes), but φ is like scaling that blueprint to 80%, 100%, 120%, 150% size—the layout stays the same, just everything gets proportionally bigger! 🎯