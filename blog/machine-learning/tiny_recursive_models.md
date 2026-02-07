@def title = "Less is More: Recursive Reasoning with Tiny Networks"
@def published = "7 February 2026"
@def tags = ["machine-learning"]

# Less is More: Recursive Reasoning with Tiny Networks

**Paper**: [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) (Jolicoeur-Martineau, Samsung SAIL Montréal, October 2025)

**Code**: [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

---

## The Punchline

A **7 million parameter** network with only **2 layers** beats DeepSeek-R1 (671B parameters), o3-mini, and Gemini 2.5 Pro on hard puzzle benchmarks like ARC-AGI. That's less than 0.01% of the parameters. The trick? Instead of making the network bigger, you run it **recursively** --- feeding its output back as input, over and over, letting it iteratively refine its answer.

The core message: **memory (storing and reusing intermediate state) can substitute for compute (more parameters and deeper networks).**

---

## Prerequisites: What You Need to Know

### What is a Transformer?

If you know the basics of LLMs, you know transformers. A transformer layer takes a sequence of vectors (shape $[L, D]$ where $L$ is sequence length and $D$ is embedding dimension), applies self-attention and a feed-forward network, and outputs vectors of the same shape. Stacking many layers gives you depth.

Key components used in this paper:
- **Self-attention**: lets each position attend to every other position. Good for long sequences but expensive
- **MLP (feed-forward)**: a position-wise neural network applied independently to each position. Cheap when $L$ is small
- **RMSNorm**: a normalization layer (simpler variant of LayerNorm)
- **SwiGLU**: a gated activation function (a better variant of ReLU for transformers)
- **Rotary embeddings (RoPE)**: encodes position information into the attention mechanism

### What is "Reasoning" in Current AI?

In LLMs, "reasoning" usually means **chain-of-thought (CoT)**: the model generates step-by-step text before its final answer. This is done **auto-regressively** --- one token at a time, left to right.

The problem: a single wrong token can derail the entire reasoning chain. And generating thousands of reasoning tokens is expensive.

This paper proposes a fundamentally different kind of reasoning: **recursive refinement of a latent state** rather than sequential text generation.

---

## The Setup: Supervised Learning on Puzzles

This is **not** an LLM. It's a supervised learning model:

- **Input**: a puzzle (e.g., a Sudoku grid with blanks, a maze, an ARC-AGI task)
- **Output**: the solution (e.g., the filled grid, the path, the answer grid)
- Both input and output are tokenized into sequences of shape $[B, L]$, where $B$ is the batch size and $L$ is the context length

The model is trained on **tiny datasets** (~1000 examples) with heavy data augmentation (shuffling, rotations, flips, color permutations). This is a regime where large models would massively overfit.

---

## The Architecture: What Does a "Reasoning Model" Look Like Here?

### The Predecessor: Hierarchical Reasoning Model (HRM)

To understand TRM, you first need HRM (Wang et al., 2025). HRM uses **two** small transformer networks:
- $f_L$ ("low-level"): runs at high frequency, updates a latent feature $z_L$
- $f_H$ ("high-level"): runs at low frequency, updates a latent feature $z_H$

These two networks recurse: $f_L$ runs multiple times, then $f_H$ runs once, then repeat. The biological motivation is that the brain processes information at different temporal frequencies.

HRM has 27M parameters (two 4-layer transformers) and achieves 40% on ARC-AGI-1.

### TRM: Strip It Down to the Essentials

The key insight of TRM is that HRM is **overcomplicated**. The author identifies that:

1. $z_H$ is just the **current predicted answer** $y$ (you can decode it back to tokens and it looks like a solution)
2. $z_L$ is a **latent reasoning state** $z$ (decoding it gives nonsense --- it's internal working memory)
3. You don't need two networks. One network can do both jobs, since the tasks are distinguished by **which inputs are provided**

So TRM uses:
- **One network** $f$ with only **2 layers** (not two networks with 4 layers each)
- **Two state variables**: the current answer $y$ and the latent reasoning state $z$
- **Recursive application** of the same network

### The Core Loop (Pseudocode)

Here's the entire TRM algorithm:

```python
def latent_recursion(x, y, z, n=6):
    """One round of recursive reasoning"""
    for i in range(n):
        # Update latent reasoning given question, current answer, current reasoning
        z = net(x, y, z)
    # Update answer given current answer and reasoning
    y = net(y, z)
    return y, z

def deep_recursion(x, y, z, n=6, T=3):
    """Multiple rounds, backprop only through the last one"""
    # T-1 rounds without gradients (cheap refinement)
    with torch.no_grad():
        for j in range(T - 1):
            y, z = latent_recursion(x, y, z, n)
    # 1 round with gradients (learning signal)
    y, z = latent_recursion(x, y, z, n)
    return y.detach(), z.detach(), output_head(y)
```

That's it. The entire "reasoning" is:

1. **Latent recursion** ($n$ times): update the reasoning state $z$ by repeatedly applying the network with inputs $(x, y, z)$
2. **Answer update** (once): update the answer $y$ by applying the network with inputs $(y, z)$
3. **Repeat** ($T$ times): do steps 1--2 multiple times, but only backpropagate through the last repetition

### Why the Input Distinction Matters

Notice that when updating $z$, the network receives $(x, y, z)$ --- the question is included. When updating $y$, the network receives $(y, z)$ --- the question is **not** included. This input difference tells the single network which task to perform. It's an elegant trick that replaces the need for two separate networks.

---

## The Secret Sauce: Deep Supervision

The recursive loop above runs within **one supervision step**. But the real power comes from **deep supervision**: running multiple supervision steps in sequence, carrying over the latent state.

```python
# Deep supervision (the outer training loop)
for x_input, y_true in dataloader:
    y, z = y_init, z_init  # start from scratch
    
    for step in range(N_sup):  # up to 16 steps
        x = input_embedding(x_input)
        (y, z), y_hat, q = deep_recursion(x, y, z)
        
        loss = cross_entropy(y_hat, y_true)
        loss += halting_loss(q, y_hat == y_true)
        
        z = z.detach()  # cut gradients between supervision steps
        loss.backward()
        optimizer.step()
```

Each supervision step:
1. Takes the previous $(y, z)$ as initialization
2. Runs the full deep recursion (recursive reasoning + answer update)
3. Computes a loss against the true answer
4. **Detaches** the latent state (cuts the gradient graph)
5. Backpropagates and updates weights

The detachment in step 4 is crucial: it means we never backpropagate through the entire chain of 16 supervision steps. Each step is trained independently, but each step *receives* the output of the previous step. This is what allows the model to simulate enormous depth without the memory cost of backpropagating through all of it.

### Effective Depth

With 2 layers, $n = 6$ recursions, $T = 3$ repetitions, and $N_{\text{sup}} = 16$ supervision steps, the effective depth per supervision step is:

$$T \times (n + 1) \times n_{\text{layers}} = 3 \times 7 \times 2 = 42 \text{ layers}$$

Over all 16 supervision steps, that's $42 \times 16 = 672$ effective layers of processing. From a 2-layer network with 7M parameters.

---

## Why Tiny Networks? The "Less is More" Result

Here's the surprising finding: **reducing** the network from 4 layers to 2 layers and **increasing** the number of recursions $n$ proportionally (to keep the total compute similar) **improves** generalization.

From the ablation on Sudoku-Extreme:

| Configuration | Test Accuracy | Parameters |
|:---|:---:|:---:|
| TRM (2 layers, single net, full backprop) | **87.4%** | **5M** |
| + separate $f_L$ and $f_H$ networks | 82.4% | 10M |
| + 4 layers (like HRM) | 79.5% | 10M |
| + self-attention instead of MLP | 74.7% | 7M |
| + 1-step gradient (like HRM) | 56.5% | 5M |
| HRM (original) | 55.0% | 27M |

Every step that makes the model **smaller** and **simpler** improves generalization. This is counterintuitive, but makes sense in the small-data regime: with only ~1000 training examples, a larger model overfits. Recursion provides depth without adding parameters, so it doesn't increase overfitting.

---

## The Memory-Compute Trade-off

This is the core conceptual contribution. Let's make it precise.

### Standard Deep Networks

A standard $K$-layer transformer:
- **Parameters**: $O(K \times D^2)$ (each layer has its own weights)
- **Memory for backprop**: $O(K)$ (store activations for all layers)
- **Depth**: $K$

To get more depth, you need more layers, which means more parameters and more memory.

### TRM's Approach

TRM with a 2-layer network recursed $n$ times, repeated $T$ times:
- **Parameters**: $O(D^2)$ (same 2 layers reused everywhere)
- **Memory for backprop**: $O(n)$ (only backprop through one repetition)
- **Effective depth**: $T \times (n + 1) \times 2$

The trade-off:
- **More recursions** = more effective depth = more compute per example
- **Same parameters** = no additional overfitting risk
- **Stored latent state** $(y, z)$ across supervision steps = "memory" that carries information without backprop cost

In essence, TRM trades **FLOPs** (repeated forward passes through the same small network) for **parameters** (a large single-pass network). Since FLOPs are cheap but overfitting on small data is deadly, this is a very favorable trade.

### Comparison with LLM Reasoning

| Property | LLM + CoT | TRM |
|:---|:---|:---|
| Reasoning mechanism | Generate text tokens sequentially | Recurse latent states |
| Reasoning medium | Natural language | Learned latent vectors |
| Reasoning is | Explicit (human-readable) | Implicit (not decodable) |
| Error propagation | One wrong token can derail everything | Iterative refinement can self-correct |
| Parameters | Billions | Millions |
| Data needed | Massive pretraining corpus | ~1000 task-specific examples |
| Generality | General-purpose | Task-specific |

---

## Key Design Decisions Explained

### Why Not Use the Implicit Function Theorem?

HRM justified only backpropagating through the last 2 recursions by claiming the latent states converge to a **fixed point**, then invoking the Implicit Function Theorem. The TRM author shows this assumption is shaky --- the residuals never actually reach zero in practice.

TRM's solution: just backpropagate through all $n + 1$ recursions within one repetition. This is only feasible because the network is tiny (2 layers), so the memory cost is manageable. The result: going from the 1-step gradient approximation to full backprop improved Sudoku-Extreme from 56.5% to 87.4%.

### Why MLP Instead of Self-Attention?

For tasks with small fixed grids (Sudoku is 81 cells, i.e., $L = 81$ and $D = 512$), self-attention is overkill. An MLP applied along the sequence dimension (inspired by MLP-Mixer) has $[L, L]$ parameters, which is much smaller than the attention mechanism when $L \leq D$. This reduced model capacity, further combating overfitting.

However, for larger grids (30x30 mazes, ARC-AGI), self-attention works better due to its inductive bias for relational reasoning.

### Why EMA?

The model is trained on so little data that it tends to overfit and then diverge. Exponential Moving Average (EMA) of the weights (common in GANs and diffusion models) smooths out training and prevents sharp collapse. Decay of 0.999 worked well.

### Why Adaptive Computational Time (ACT)?

With 16 supervision steps, spending all 16 steps on every training example is wasteful --- many examples are solved early. ACT learns a **halting probability**: "is the current answer already correct?" If so, move on to the next example. During training, this means spending an average of ~2 steps per example instead of 16, allowing much better data coverage. At test time, all 16 steps are used to maximize accuracy.

TRM simplifies HRM's ACT (which required Q-learning and a second forward pass) to a simple binary cross-entropy loss on whether the current prediction matches the target.

---

## Results: How Valid Are They?

| Method | Params | Sudoku-Extreme | Maze-Hard | ARC-AGI-1 | ARC-AGI-2 |
|:---|:---:|:---:|:---:|:---:|:---:|
| DeepSeek R1 | 671B | 0.0% | 0.0% | 15.8% | 1.3% |
| o3-mini-high | ? | 0.0% | 0.0% | 34.5% | 3.0% |
| Gemini 2.5 Pro | ? | 0.0% | 0.0% | 37.0% | 4.9% |
| HRM | 27M | 55.0% | 74.5% | 40.3% | 5.0% |
| **TRM-Att** | **7M** | 74.7% | **85.3%** | **44.6%** | **7.8%** |
| **TRM-MLP** | **5M** | **87.4%** | 0.0% | 29.6% | 2.4% |

### The Honest Assessment

**What's genuinely impressive:**
- On Sudoku-Extreme and Maze-Hard, LLMs (even the best reasoning models) score **0.0%**. TRM scores 87.4% and 85.3%. This isn't marginal --- it's a qualitative difference
- On ARC-AGI-1, 44.6% with 7M parameters outperforms most frontier LLMs
- The parameter efficiency is extraordinary: 0.01% of the parameters, trained from scratch on ~1000 examples

**Caveats to keep in mind:**
- This is **task-specific supervised learning**, not a general-purpose model. A TRM trained on Sudoku can't solve mazes. An LLM can attempt both (even if it fails)
- The comparisons with LLMs are apples-to-oranges: LLMs are general-purpose; TRM is a specialist
- The MLP variant (best on Sudoku) completely fails on Maze-Hard, showing that architecture choices are task-dependent
- ARC-AGI-2 accuracy of 7.8% is better than most LLMs but still far from human-level
- The top ARC-AGI-1 scores (Bespoke/Grok-4 at 79.6%) still far exceed TRM, though they use 1.7T parameters and likely enormous test-time compute

**What it really shows:** For structured reasoning problems with small data, **recursive depth via weight-sharing** is a dramatically better inductive bias than scaling parameters. The LLMs aren't failing because they lack knowledge --- they're failing because autoregressive token generation is a poor fit for constraint satisfaction problems.

---

## Why This Matters for Academics (The Real Takeaway)

Here's the honest truth: **you are not competing with Samsung or DeepSeek or OpenAI on resources**. You have a single GPU (maybe two), a few weeks per semester, and datasets you can build by hand. TRM is written for you.

### The Specific Advantages for Academia

**1. You can train it from scratch in days, not months**

- Sudoku-Extreme: 18 hours on 1 L40S GPU
- You don't need a cluster
- You don't need pretraining
- No fine-tuning: direct supervised learning from scratch on your task

Compare this to fine-tuning any LLM:
- DeepSeek-R1: requires massive compute to train from scratch, and even the open weights are prohibitively expensive to fine-tune
- You can't even *run* o3-mini or Gemini 2.5 locally; you pay by the token

**2. You can actually own and modify the code**

The entire training loop is ~500 lines of PyTorch. It's simple enough to understand completely, modify, and experiment with. You're not blocked by a closed API or a 671B parameter model that takes hours to generate one example.

**3. Your dataset size becomes an advantage, not a liability**

In the LLM paradigm, small datasets are bad news — your model will just memorize or fail. In the TRM paradigm, small datasets are fine. The recursive structure prevents overfitting. You curate 1000 examples, augment them, and you're done. No need to scrape the entire internet.

**4. Parameter efficiency = computational democracy**

A 5M-parameter model:
- Fits in GPU memory with headroom
- Trains in hours
- You can run 10 different architectures in parallel on modest hardware
- You can afford to do ablations and hyperparameter sweeps

A 671B-parameter model:
- You might not be able to run it at all
- Each experiment costs moneeeey
- You can do maybe one or two runs before the grant money runs out

**5. You can actually publish novel research**

TRM shows a very specific insight: **weight-sharing defeats overfitting better than model capacity**. This is a *generalizable principle*, not a task-specific hack. That means:
- You can apply this to your own structured reasoning problems (scheduling, planning, constraint satisfaction, protein folding, symbolic reasoning)
- You can publish novel architectures that explore this trade-off
- You're not trying to beat LLMs at their game (in which case you lose); you're solving a different problem with better tools

### A Concrete Research Scenario

Suppose you're at a mid-tier university and want to work on reasoning in biology: predicting protein secondary structure from a small labeled dataset (5000 examples).

**The old way:**
- Fine-tune ESMFold or OmegaFold (2.7B parameters each)
- Hope your GPU doesn't run out of memory
- Cross your fingers and hope the pretrained weights transfer
- Publish a "fine-tuning" paper, which is incremental

**The TRM way:**
- Build a 2-layer transformer with recursive refinement
- Train from scratch on your 5000 examples in a few days
- Ablate aggressively: test MLP vs attention, test different recursion depths, test different loss functions
- Discover which architectural choices matter for *your* problem specifically
- Publish a paper on "recursive refinement for protein structure prediction" or "how much depth do you actually need for secondary structure"? That's novel.

### The Brutal Honesty

Yes, TRM is currently evaluated only on puzzles (Sudoku, mazes, ARC-AGI). That's a limitation. But it's not because puzzles are the only domain where weight-sharing beats model capacity --- it's because those were convenient benchmarks with ground truth, easy evaluation, and easy data augmentation.

The principles apply anywhere:
- You have a task with a clear ground truth
- Your data is small (< 100k examples)
- Your problem is sufficiently complex that bigger = worse (due to overfitting)
- You can afford to run the model multiple times at test time

That covers a LOT of academic research.

---

## Trying It Yourself

The code is open source. For Sudoku-Extreme on a single GPU:

```bash
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels
cd TinyRecursiveModels
pip install -r requirements.txt

# Build dataset
python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug-1000 \
    --subsample-size 1000 --num-aug 1000

# Train TRM (MLP variant, ~18h on 1 L40S)
python pretrain.py \
    arch=trm \
    data_paths="[data/sudoku-extreme-1k-aug-1000]" \
    evaluators="[]" \
    epochs=50000 eval_interval=5000 \
    lr=1e-4 weight_decay=1.0 \
    arch.mlp_t=True arch.pos_encodings=none \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=6 \
    ema=True
```

Expected: ~87% test accuracy with 5M parameters.

---

## The Bigger Picture

TRM is part of an emerging theme in AI: **you don't always need to scale up**. Specifically:

1. **Test-time compute** (reasoning models like DeepSeek-R1, o1): let a large model think longer instead of making it bigger
2. **Recursive refinement** (TRM, HRM): let a tiny model iterate instead of making it deeper
3. **Weight sharing** (deep equilibrium models, universal transformers): reuse the same parameters across depth

All three trade FLOPs for parameters. The difference is that TRM pushes this to an extreme: 2 layers, 7M parameters, 672 effective layers. The implication is provocative --- for structured problems, maybe the right scaling law isn't "bigger models" but "more iterations of small models."

Whether this extends beyond puzzles to broader reasoning tasks remains an open question. The author notes that TRM is currently a supervised learning method (deterministic single answer) and extending it to generative settings would be interesting future work.

But the core insight is already clear: **depth through recursion, not parameters, is what enables reasoning on hard problems with limited data.**
