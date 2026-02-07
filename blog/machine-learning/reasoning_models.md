@def title = "How Reasoning Models Work"
@def published = "7 February 2026"
@def tags = ["machine-learning"]

# How Reasoning Models Work

This post builds directly on [The Transformer Loss Function](/blog/machine-learning/llm_loss/). If you haven't read that, the short version: a transformer predicts the next token at each position by projecting its hidden state to logits over a vocabulary of size $V$, then minimizing cross-entropy:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log P(y_i \mid y_1, \ldots, y_{i-1}; \theta)$$

where $P(y_i \mid \cdot)$ comes from a softmax over logits $\mathbf{z}_i = \mathbf{W}_{\text{vocab}} \mathbf{t}_i \in \mathbb{R}^V$.

Reasoning models modify *what the model generates* and *how it's trained*, but the underlying loss machinery is the same. Let's see exactly how.

---

## Part I: Reasoning via Chain of Thought

### The Core Idea

In `llm_loss.md`, we saw the standard autoregressive factorization for any sequence of tokens:

$$P(\text{sequence}; \theta) = \prod_{i=1}^{N} P(\text{token}_i \mid \text{token}_1, \ldots, \text{token}_{i-1}; \theta)$$

For reasoning tasks, we can think of the input as having two parts: a **prompt** or **question** $x$, and the **response** $y$ that the model generates. In this view, a standard LLM takes a question $x$ and directly generates an answer $y$:

$$P(y \mid x; \theta) = \prod_{i=1}^{|y|} P(y_i \mid x, y_1, \ldots, y_{i-1}; \theta)$$

This is still the same autoregressive factorization — we're just explicitly separating the "given" part ($x$) from the "generated" part ($y$). Each factor $P(y_i \mid x, y_1, \ldots, y_{i-1}; \theta)$ is computed via the same logit $\rightarrow$ softmax $\rightarrow$ cross-entropy pipeline from `llm_loss.md`, where the conditioning context includes both the original prompt tokens and the previously generated response tokens.

A reasoning model inserts a **thinking trace** $t$ between the question and the answer:

$$P(y \mid x; \theta) = \sum_t P(t \mid x; \theta) \, P(y \mid x, t; \theta)$$

The trace $t = (t_1, t_2, \ldots, t_M)$ is a sequence of tokens --- the model's "scratchpad." It might look like:

```
x: "What is 27 * 34?"
t: "I need to multiply 27 by 34. 27 * 30 = 810. 27 * 4 = 108. 810 + 108 = 918."
y: "918"
```

Here's the crucial point: **mechanically, generating $t$ and $y$ uses the exact same next-token prediction**. The model doesn't know that $t$ is "reasoning" and $y$ is "answer." It just generates a sequence of tokens, left to right, each one predicted from all previous tokens via softmax over logits. The only difference is that we've structured the training data (or the RL reward) so that generating intermediate steps before the final answer leads to better answers.

### Why Does a Longer Trace Help?

This is the **test-time compute** insight. Consider the autoregressive factorization of the full generation $(t, y)$:

$$P(t, y \mid x; \theta) = \underbrace{\prod_{j=1}^{M} P(t_j \mid x, t_1, \ldots, t_{j-1}; \theta)}_{\text{generating the trace}} \cdot \underbrace{\prod_{i=1}^{|y|} P(y_i \mid x, t, y_1, \ldots, y_{i-1}; \theta)}_{\text{generating the answer}}$$

When the model generates $y_1$, it conditions on the *entire* trace $t$. That means:
- A model with $M = 0$ trace tokens computes $y$ from $x$ alone --- one "shot" of the transformer
- A model with $M = 200$ trace tokens has effectively run the transformer 200 additional forward steps, each time updating the key-value cache with new information

Each trace token is a forward pass through the network. More trace tokens = more serial computation before committing to an answer. The trace acts as **external working memory** that the model writes to and reads from through the attention mechanism.

This is why a 7B model "thinking" for 200 tokens can outperform a 70B model answering in 1 token: the 7B model does more total serial computation per problem.

---

## Part II: Training Reasoning Models

There are two main approaches, and they are complementary.

### Approach 1: Distillation (Supervised Fine-Tuning)

The simplest recipe. Take a powerful model, generate $(x, t, y)$ triples, and fine-tune a smaller model on them using the standard cross-entropy loss:

$$\mathcal{L}_{\text{distill}} = -\frac{1}{M + |y|} \sum_{j=1}^{M+|y|} \log P(s_j \mid s_1, \ldots, s_{j-1}; \theta)$$

where $s = (t_1, \ldots, t_M, y_1, \ldots, y_{|y|})$ is the concatenated trace-then-answer sequence.

**Key point: No architectural changes needed.** This is *identical* to standard language model training. You're using the exact same transformer architecture, the exact same forward pass, the exact same backpropagation. The only difference is the training data — instead of training on raw text, you train on sequences that include reasoning traces. The loss function, the attention mechanism, the vocabulary projection: all unchanged from `llm_loss.md`.

**What does "training" mean here?**

You typically start with a pretrained base model (e.g., Llama-3-7B, Qwen-2.5-1.5B) and **fine-tune** it on reasoning traces. You don't train from scratch. The procedure:

1. Use a teacher model to generate $(x, t, y)$ triples: prompt $x$, reasoning trace $t$, final answer $y$
2. Concatenate them into a single sequence: $s = [x, t, y]$
3. Fine-tune the student model on these sequences with standard cross-entropy loss
4. The model learns to predict reasoning tokens just like it learned to predict any other tokens during pretraining

**But wait — doesn't this require labels/answers?**

Yes! This is a key difference from standard unsupervised pretraining (which only requires raw text). For reasoning model distillation, you need:
- **Prompts** $x$ (questions like "What is 27 * 34?")
- **Ground-truth answers** $y$ (like "918")

The teacher model generates the reasoning trace $t$, but you still need to know the correct answer to verify the teacher's output is valid and to provide the final answer token for the student to learn.

In practice, this means you're working with **supervised datasets** — collections of question-answer pairs like GSM8K (math problems), MATH (competition math), APPS (coding problems), etc. Standard pretraining doesn't need these labels; distillation does.

**Practical notes from the DeepSeek-R1 distillation experiments:**
- Don't use sample packing --- reasoning traces are long, and splitting them across chunks destroys coherence
- Use a larger learning rate (4e-5 vs. the usual 2e-5) --- each doubling gained ~10 points on coding benchmarks
- Prefill the prompt with a thinking token --- distilled models sometimes skip reasoning unless nudged

### Approach 2: Reinforcement Learning (GRPO)

Distillation requires a teacher. What if you want the model to *discover* reasoning on its own?

This is where RL comes in. **Again, no architectural changes needed.** You take the same transformer that was doing standard next-token prediction, but instead of training with cross-entropy on $(x, t, y)$ triples, you let the model generate its own traces, then **reward or penalize** based on whether the final answer is correct.

The key insight: the model architecture doesn't "know" it's doing reasoning. From the model's perspective, it's still just predicting the next token given all previous tokens. The RL training teaches it that certain token patterns (step-by-step reasoning) tend to lead to higher rewards than others (direct answers).

**What does "training" mean here?**

You typically start with a pretrained and instruction-tuned base model, then apply RL:

1. Sample a prompt $x$ from your dataset
2. **The model generates** a full response $o$ (which may or may not include reasoning)
3. Extract the final answer and check if it's correct → compute reward $R(o, x)$
4. Update the model's weights using the GRPO loss to increase probability of high-reward responses

**Do you generate traces during training?** Yes! This is the key difference from distillation. During each training step, the model generates multiple candidate outputs (the "group" of size $G$), and those outputs are scored by the reward function. Over time, the model learns that generating step-by-step reasoning before the final answer tends to produce correct answers more often, so it starts doing that spontaneously.

**But wait — doesn't this require labels/answers?**

Yes! The reward function $R(o, x)$ needs to verify if the model's answer is correct, which means you need ground-truth answers. This is why RL-based reasoning training focuses on **verifiable domains**:
- **Math**: check if the numerical answer matches the ground truth
- **Code**: run test cases and check if they pass
- **Formal reasoning**: verify logical consistency

Standard unsupervised pretraining doesn't need these labels. GRPO does. The key difference from distillation is that GRPO only needs the final answer labels — it doesn't need the reasoning traces $t$ to be provided. The model discovers those on its own.

#### From Policy Gradients to GRPO

Let's build this up carefully.

**The RL framing.** We treat the language model as a **policy** $\pi_\theta$ that maps a prompt $x$ (the "state") to a generated sequence $o$ (the "action"). A reward function $R(o, x)$ scores the output (e.g., $R = 1$ if the math answer is correct, $R = 0$ otherwise).

We want to maximize the expected reward:

$$J(\theta) = \E_{x \sim \mathcal{D}} \left[\E_{o \sim \pi_\theta(\cdot \mid x)} [R(o, x)] \right]$$

The standard policy gradient (REINFORCE) gives us the gradient:

$$\nabla_\theta J = \E \left[ R(o, x) \nabla_\theta \log \pi_\theta(o \mid x) \right]$$

This says: increase the log-probability of outputs that got high reward, decrease it for low reward. The problem is **high variance** --- a single sample $o$ is a noisy estimate of what the model should do.

**PPO (Proximal Policy Optimization)** fixes this with two tricks:
1. Use a **baseline** (value function) to reduce variance: replace $R$ with the advantage $A = R - V(x)$
2. **Clip** the probability ratio to prevent the policy from changing too much in one step

The PPO objective for a single output $o$ is:

$$\mathcal{L}_{\text{PPO}} = \min\left(\frac{\pi_\theta(o \mid x)}{\pi_{\theta_{\text{old}}}(o \mid x)} A, \; \text{clip}\left(\frac{\pi_\theta(o \mid x)}{\pi_{\theta_{\text{old}}}(o \mid x)}, 1-\epsilon, 1+\epsilon\right) A \right)$$

The clipping ensures: if the advantage is positive (good output), don't increase the probability ratio beyond $1 + \epsilon$. If negative (bad output), don't decrease it beyond $1 - \epsilon$. This keeps training stable.

**The problem with PPO for LLMs:** you need a value function $V(x)$ --- a separate neural network that estimates the expected reward for each prompt. For large language models, this means training *another* model of similar size, roughly doubling memory requirements.

**GRPO (Group Relative Policy Optimization)** eliminates the value function entirely. Instead of estimating $V(x)$ with a neural network, GRPO:

1. For each prompt $x$, sample a **group** of $G$ outputs: $\{o_1, o_2, \ldots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)$
2. Score each with the reward function: $r_i = R(o_i, x)$
3. Compute the advantage **within the group** (normalize by group statistics):

$$A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

This is clever: the group itself serves as the baseline. If 6 out of 8 outputs are correct ($r = 1$) and 2 are wrong ($r = 0$), then the correct ones get a positive advantage and the wrong ones get a negative advantage. No separate value model needed.

4. The GRPO loss:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\E_{x} \left[\frac{1}{G}\sum_{i=1}^{G} \min\left(\frac{\pi_\theta(o_i \mid x)}{\pi_{\theta_{\text{old}}}(o_i \mid x)} A_i, \; \text{clip}(\cdot) A_i \right) - \beta \, D_{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

The KL divergence term $D_{KL}(\pi_\theta \| \pi_{\text{ref}})$ penalizes the model for straying too far from a reference policy (typically the initial supervised model). This prevents "reward hacking" --- the model finding degenerate outputs that exploit the reward function.

#### What the Probability Ratio Actually Means

Let's be concrete. The ratio $\frac{\pi_\theta(o_i \mid x)}{\pi_{\theta_{\text{old}}}(o_i \mid x)}$ is a product of per-token ratios:

$$\frac{\pi_\theta(o_i \mid x)}{\pi_{\theta_{\text{old}}}(o_i \mid x)} = \prod_{k=1}^{|o_i|} \frac{P_\theta(o_{i,k} \mid x, o_{i,1}, \ldots, o_{i,k-1})}{P_{\theta_{\text{old}}}(o_{i,k} \mid x, o_{i,1}, \ldots, o_{i,k-1})}$$

Each factor is a ratio of softmax probabilities --- computed via the same logits-to-probabilities pipeline from `llm_loss.md`. The only difference is that instead of comparing against a ground-truth token, we're comparing the *current* model's probability of a token against the *old* model's probability of the same token.

In practice, we work in log-space to avoid numerical issues:

$$\log \frac{\pi_\theta(o_i \mid x)}{\pi_{\theta_{\text{old}}}(o_i \mid x)} = \sum_{k=1}^{|o_i|} \left[\log P_\theta(o_{i,k} \mid \cdot) - \log P_{\theta_{\text{old}}}(o_{i,k} \mid \cdot) \right]$$

Each $\log P_\theta(o_{i,k} \mid \cdot)$ is exactly the log-softmax of logits at position $k$ --- the same quantity that appears in the cross-entropy loss, just evaluated at the *sampled* token instead of a *ground-truth* token.

#### The Remarkable Finding: Emergent Reasoning (R1-Zero)

When you apply GRPO directly to a base model (no supervised fine-tuning, no reasoning traces in the training data), the model spontaneously develops reasoning behaviors:

- **Self-verification**: "Let me check this answer..."
- **Backtracking**: "Wait, that's wrong. Let me try again..."
- **Strategy exploration**: trying multiple approaches before committing

These behaviors **emerge** because the RL objective rewards correct final answers. The model discovers on its own that "thinking before answering" is a strategy that increases reward. Nobody programmed these patterns --- the optimization landscape favors them.

#### Distillation vs. RL: The Trade-off

| Property | Distillation | GRPO |
|:---|:---|:---|
| Training loss | Cross-entropy (standard LM loss) | Clipped policy gradient + KL penalty |
| Requires | Teacher model traces | Reward function (verifier) |
| What model learns | To imitate teacher's reasoning | Its own reasoning strategies |
| Data efficiency | Needs many $(x, t, y)$ examples | Needs prompts $x$ + reward signal |
| Compute cost | Standard fine-tuning | Higher (generate $G$ samples per step) |
| Surprise factor | Predictable (copies teacher) | Can discover novel strategies |

The DeepSeek-R1 pipeline uses both: RL first to develop reasoning ability, then distillation to transfer that ability to smaller models.

---

## Part III: A Different Kind of Reasoning --- Recursive Refinement

Everything above is about **autoregressive** reasoning: the model generates tokens left-to-right, and the trace is expressed in natural language. But there's a completely different approach, explored in the [TRM paper](/blog/machine-learning/tiny_recursive_models/) (Jolicoeur-Martineau, 2025).

### The Problem with Autoregressive Reasoning

Chain-of-thought reasoning has a fundamental fragility: it's **sequential and irreversible**. Each token is generated conditioned on all previous tokens. If token $t_{j}$ is wrong, every subsequent token $t_{j+1}, t_{j+2}, \ldots$ is conditioned on that error. The model can try to "backtrack" in text ("Wait, that's wrong..."), but it's generating more tokens to fix earlier tokens --- it can't actually rewrite the past.

Also, reasoning in natural language is *expensive*. Each reasoning token requires a full forward pass through the model. A 7B model generating 2000 reasoning tokens does $2000 \times 7\text{B} = 14$ trillion multiply-adds just for thinking.

### Recursive Refinement: Think in Vectors, Not Words

TRM (Tiny Recursive Model) takes a radically different approach. **This requires major architectural changes.** Instead of generating a text trace autoregressively, it maintains **latent state vectors** and iteratively refines them by running the same tiny network over and over.

The setup abandons language modeling entirely in favor of supervised learning. Given an input $x$ (a puzzle) and target $y^*$ (the solution), both tokenized into sequences:

$$x \in \{1, \ldots, V\}^{L_x}, \quad y^* \in \{1, \ldots, V\}^{L_y}$$

**Why the same length $L$ in practice?** For simplicity, TRM assumes $x$ and $y$ have the same sequence length. If the natural input and output have different lengths, you pad the shorter one. For example:
- **Sudoku**: Input is a 9×9 grid with blanks (81 tokens), output is the completed grid (81 tokens) — naturally the same length
- **Mazes**: Input is the maze layout, output is the solution path — pad whichever is shorter to match
- **ARC-AGI**: Input and output grids may differ in size — pad to a common maximum length

This is a **supervised learning** setup, not a generative language modeling task. You're training the network to map fixed-size inputs to fixed-size outputs, like image-to-image translation. The padding is just a practical detail to make the matrix operations work.

**Architectural differences from standard transformers:**

1. **No autoregressive generation**: The model doesn't generate tokens left-to-right
2. **Stateful operation**: The model maintains evolving state vectors $(y, z)$ across multiple forward passes
3. **Weight sharing**: The same 2-layer transformer is applied recursively, rather than having separate layers
4. **Input conditioning**: The network receives different inputs at different steps — sometimes $(x, y, z)$, sometimes just $(y, z)$

The model maintains three variables as continuous embeddings:
- $x \in \mathbb{R}^{L \times D}$ --- the input (fixed, embedded from discrete tokens)
- $y \in \mathbb{R}^{L \times D}$ --- the current answer estimate (evolves, lives in embedding space)
- $z \in \mathbb{R}^{L \times D}$ --- the latent reasoning state (evolves, lives in a **different learned space**)

**Critical point about $z$:** The reasoning trace $z$ is **not** in the same space as the token embeddings. Here's the distinction:

- $x$ and $y$: These start as discrete token IDs, get embedded into $\mathbb{R}^D$, and can be decoded back to tokens via $\mathbf{W}_{\text{vocab}}$
- $z$: This is a **pure latent vector** that the network learns to use as working memory. It's not tied to any vocabulary. If you project $z$ through $\mathbf{W}_{\text{vocab}}$ and decode it, you get gibberish — it was never meant to be interpretable as tokens

Think of it this way:
- In CoT models, reasoning happens in **token space** (natural language you can read)
- In TRM, reasoning happens in **latent space** (learned representations with no direct token correspondence)

This is why TRM can't explain its reasoning in human language — the trace $z$ is optimized purely for computational utility, not linguistic interpretability.

**What does "training" mean here?**

You build and train this custom network **from scratch** (no pretrained base model). For each training example:

1. Start with random initializations for $y$ and $z$
2. Run the recursive refinement loop (latent updates, answer updates, repeated across multiple supervision steps)
3. After each supervision step, compute cross-entropy between the predicted answer and ground truth
4. Backpropagate the loss and update the 2-layer network's weights

**Do you generate traces during training?** Not in natural language. The "reasoning" happens entirely in the latent vector $z$, which is never decoded to text. The model refines $z$ over many forward passes, and eventually uses it to produce the final answer $y$. But if you tried to decode $z$ to tokens at any intermediate step, you'd get nonsense — it's internal working memory, not a linguistic trace.

This is fundamentally different from CoT models, where the reasoning is explicitly generated as text tokens that you can read.

**But wait — doesn't this require labels/answers?**

Absolutely! TRM is **fully supervised**. For every training example, you need:
- Input puzzle $x$ (e.g., a Sudoku grid with blanks)
- Ground-truth solution $y^*$ (the completed grid)

This is pure supervised learning, like training an image classifier. You optimize the model to produce outputs that match the labels. No unsupervised pretraining phase — just direct optimization on $(x, y^*)$ pairs.

The TRM paper uses ~1000 labeled examples per task (Sudoku, mazes, ARC-AGI), with heavy data augmentation to prevent overfitting.

A single 2-layer transformer $f_\theta$ is applied recursively:

**Latent update** ($n$ times): refine the reasoning state given the full context

$$z \leftarrow f_\theta([x; y; z])$$

**Answer update** (once): update the answer from the reasoning state

$$y \leftarrow f_\theta([y; z])$$

where $[a; b; c]$ denotes concatenation along the sequence dimension.

Notice the key difference: when updating $z$, the network sees the input $x$. When updating $y$, it does not. This input distinction lets one network play two roles --- "reasoner" (sees the question) and "summarizer" (produces the answer).

**Why not just one forward pass through a transformer?**

You might ask: "A transformer already has self-attention that lets position 2 look at position 40. Why not just do one forward pass $x \rightarrow y$ and be done?"

Great question! The issue is **depth of reasoning** vs **width of information gathering**.

One forward pass through a transformer gives you:
- ✓ **Information gathering**: Position 2 can attend to all other positions (rows, columns, blocks in Sudoku)
- ✗ **Iterative refinement**: You only get one "shot" at solving the puzzle

But constraint satisfaction problems require **multiple rounds of inference**:

```
Round 1: "Position 5 can't be 3 or 7 (those are in the same row)"
Round 2: "Now that I know position 5 isn't 3, position 8 must be 3"
Round 3: "Since position 8 is 3, position 12 can't be 3..."
```

This is **serial reasoning** — each conclusion enables the next one. A single forward pass, even with perfect attention, can't do this because all positions are computed in parallel from the input.

**TRM's recursion** solves this by running the same small transformer **many times**:

$$\begin{aligned}
\text{Pass 1:} & \quad z^{(1)} = f_\theta([x; y^{(0)}; z^{(0)}]) \quad \text{(initial constraints)} \\
\text{Pass 2:} & \quad z^{(2)} = f_\theta([x; y^{(0)}; z^{(1)}]) \quad \text{(propagate conclusions)} \\
& \quad \vdots \\
\text{Pass 672:} & \quad z^{(672)} = f_\theta([x; y^{(*)}, z^{(*)}]) \quad \text{(final answer)}
\end{aligned}$$

Each pass updates $z$ based on the conclusions from the previous pass. After 672 applications of the same 2-layer network, you've done 672 "rounds" of constraint propagation.

**Why not just make the transformer deeper?** You could build a 672-layer transformer and do one pass. But:
1. **Parameter explosion**: 672 separate layers = hundreds of millions of parameters → overfits on 1000 examples
2. **No weight sharing**: Each layer learns different computations, so you need massive data
3. **No recurrence**: Can't naturally model "repeat this reasoning step until converged"

TRM's recursion gives you the **depth** (672 effective layers) with the **parameter efficiency** (5M parameters, reused 672 times). It's depth through iteration, not through stacking layers.

**The analogy**: Think of solving a Sudoku:
- **One forward pass**: Looking at the entire grid once and writing down all answers simultaneously
- **TRM recursion**: Looking at the grid, making one deduction, looking again with that new information, making another deduction, repeat 672 times

Humans solve Sudoku the second way. TRM does too.

**What does the latent state $z$ actually learn?** The intermediate "reasoning" stored in $z$ across the 672 passes encodes **partial solutions and constraints** that make the final answer easier to arrive at. For example, in Sudoku:

- Early passes: $z$ might encode "position 5 can be 2, 4, or 6" (eliminating impossible values)
- Middle passes: $z$ might encode "if position 5 is 2, then position 8 must be 7" (conditional constraints)
- Late passes: $z$ converges to "position 5 is definitely 4" (commitment)

The final answer $y$ is computed from this refined $z$ state. The loss at each supervision step encourages $z$ to progressively refine toward states that make correct predictions easier.

**Important clarification: $z$ is deterministic, not generated.** Unlike CoT traces, $z$ is not generated autoregressively with randomness. Each update is:

$$z^{(s,t,i)} = f_\theta(\text{concat}(x, y^{(s,t,i-1)}, z^{(s,t,i-1)}))$$

This is a **deterministic forward pass**. You feed in concatenated vectors, the transformer processes them, and out comes the new $z$. No sampling, no choice — just computation. The number of iterations $(n, T, N_{\text{sup}})$ is **fixed beforehand** (typically $n=6$, $T=3$, $N_{\text{sup}}=16$).

This is exactly analogous to CoT reasoning in language models — the trace $t$ ("27 × 30 = 810, 27 × 4 = 108, 810 + 108 = ...") stores intermediate results that make the final answer ("918") easier to produce. The difference is:
- **CoT**: Intermediate steps are explicit tokens, generated autoregressively (with sampling at inference)
- **TRM**: Intermediate steps are latent vectors, computed deterministically (no sampling)

### The Mathematics of TRM

Let's be completely explicit about what happens. Given a training example with discrete tokens:

$$x_{\text{discrete}} \in \{1, \ldots, V\}^L, \quad y^*_{\text{discrete}} \in \{1, \ldots, V\}^L$$

**Step 1: Embedding.** Convert discrete tokens to continuous vectors via an embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times D}$:

$$x = \mathbf{E}[x_{\text{discrete}}] \in \mathbb{R}^{L \times D}$$

where $\mathbf{E}[k]$ retrieves the $k$-th row of $\mathbf{E}$ (the embedding for token $k$).

**Step 2: Initialize states.** Start with random or zero initialization:

$$y^{(0)} = \mathbf{0} \in \mathbb{R}^{L \times D}, \quad z^{(0)} = \mathbf{0} \in \mathbb{R}^{L \times D}$$

**Step 3: Recursive refinement.** This is where the notation gets dense. Let me unpack it carefully.

We have three levels of nesting:
- **Supervision steps** $s = 1, \ldots, N_{\text{sup}}$: how many times we snapshot the state and compute a loss
- **Outer repetitions** $t = 1, \ldots, T$: how many times we repeat the latent-update loop
- **Inner latent updates** $i = 1, \ldots, n$: how many times we refine $z$ before updating $y$

The superscripts $(s, t, i)$ track where we are in this nested loop. Here's the **complete forward pass as pseudocode**:

```
z ← 0, y ← 0  // Initialize states

for s = 1 to N_sup:  // Supervision loop (e.g., 16 times)
    
    for t = 1 to T:  // Outer repetition loop (e.g., 3 times)
        
        for i = 1 to n:  // Inner latent update loop (e.g., 6 times)
            z ← f_θ([x; y; z])  // Refine latent state
        
        y ← f_θ([y; z])  // Update answer from refined latent state
    
    // After all T repetitions, compute loss
    ŷ = y @ W_vocab^T  // Project to vocabulary space
    L_s = CrossEntropy(ŷ, y*)  // Compare to ground truth
    
    // Carry forward to next supervision step (but detach gradients)
    z ← stop_gradient(z)
    y ← stop_gradient(y)
```

**What this means:**

1. **Inner loop ($i = 1, \ldots, n$)**: Refine $z$ a total of $n$ times. $y$ stays fixed during these updates.
2. **Middle loop ($t = 1, \ldots, T$)**: After each $n$-pass refinement of $z$, update $y$ once. Repeat this $(z \text{ refinement} + y \text{ update})$ pair $T$ times.
3. **Outer loop ($s = 1, \ldots, N_{\text{sup}}$)**: Repeat the entire middle+inner loops $N_{\text{sup}}$ times, computing a loss after each outer iteration.

**Concrete example** with $n=2$, $T=2$, $N_{\text{sup}}=2$:

```
=== Supervision step s=1 ===
  Repetition t=1:
    Latent update i=1: z^(1,1,1) = f_θ([x; y^(1,1,0); z^(1,1,0)])
    Latent update i=2: z^(1,1,2) = f_θ([x; y^(1,1,0); z^(1,1,2)])  // Note: y unchanged
    Answer update:     y^(1,1)   = f_θ([y^(1,1,0); z^(1,1,2)])
  
  Repetition t=2:
    Latent update i=1: z^(1,2,1) = f_θ([x; y^(1,1); z^(1,2,0)])    // Carry over y from t=1
    Latent update i=2: z^(1,2,2) = f_θ([x; y^(1,1); z^(1,2,2)])
    Answer update:     y^(1,2)   = f_θ([y^(1,1); z^(1,2,2)])
  
  Loss computation: L_1 = CrossEntropy(y^(1,2) @ W_vocab, y*)

=== Supervision step s=2 ===
  Start with y ← stop_gradient(y^(1,2)), z ← stop_gradient(z^(1,2,2))
  
  Repetition t=1:
    Latent update i=1: z^(2,1,1) = f_θ([x; y^(2,1,0); z^(2,1,0)])
    Latent update i=2: z^(2,1,2) = f_θ([x; y^(2,1,0); z^(2,1,2)])
    Answer update:     y^(2,1)   = f_θ([y^(2,1,0); z^(2,1,2)])
  
  Repetition t=2:
    Latent update i=1: z^(2,2,1) = f_θ([x; y^(2,1); z^(2,2,0)])
    Latent update i=2: z^(2,2,2) = f_θ([x; y^(2,1); z^(2,2,2)])
    Answer update:     y^(2,2)   = f_θ([y^(2,1); z^(2,2,2)])
  
  Loss computation: L_2 = CrossEntropy(y^(2,2) @ W_vocab, y*)

Total loss: L = L_1 + L_2
```

**Key details:**

- **$z$ gets overwritten in the inner loop**: Each latent update $i$ overwrites $z$ with a new value. You don't accumulate or concatenate — you replace it. The new $z^{(s,t,i)}$ is the output of $f_\theta(\text{concat}(x, y, z^{(s,t,i-1)}))$, which overwrites the old $z^{(s,t,i-1)}$.
- **$y$ doesn't change during inner loop**: The latent updates refine $z$ while $y$ stays at $y^{(s,t,0)}$.
- **$y$ updates carry forward**: After updating $y^{(s,t)}$, the next repetition $t+1$ uses this new $y$ value as $y^{(s,t+1,0)}$.
- **$z$ resets at each repetition**: Within a supervision step, $z$ starts fresh at each $t$ with $z^{(s,t,0)} = z^{(s,t-1,n)}$ (the final refined state from the previous repetition).
- **Gradients detached between supervision steps**: After computing $L_s$, we detach gradients, so supervision step $s+1$ doesn't backprop through step $s$.

**The memory picture:**

```
Iteration i=1: z^(s,t,0) --[f_θ]--> z^(s,t,1)   // Old z discarded, new z stored
Iteration i=2: z^(s,t,1) --[f_θ]--> z^(s,t,2)   // Old z discarded, new z stored
Iteration i=3: z^(s,t,2) --[f_θ]--> z^(s,t,3)   // Old z discarded, new z stored
...
Iteration i=n: z^(s,t,n-1) --[f_θ]--> z^(s,t,n)  // Final refined z
```

Only the **final** $z^{(s,t,n)}$ is kept and used to update $y^{(s,t)}$.

**Step 4: Decode to logits.** Project the final answer embedding back to vocabulary space:

$$\hat{y}^{(s)} = y^{(s,T)} \mathbf{W}_{\text{vocab}}^\mathsf{T} \in \mathbb{R}^{L \times V}$$

where $\mathbf{W}_{\text{vocab}} \in \mathbb{R}^{V \times D}$ is the vocabulary projection matrix (same as in standard transformers).

**Step 5: Compute loss.** For each position $\ell = 1, \ldots, L$ and each supervision step $s$:

$$\mathcal{L}_s = -\frac{1}{L}\sum_{\ell=1}^{L} \log \frac{\exp(\hat{y}^{(s)}[\ell, y^*_{\text{discrete}}[\ell]])}{\sum_{v=1}^{V} \exp(\hat{y}^{(s)}[\ell, v])}$$

This is cross-entropy between the predicted token distribution at position $\ell$ and the ground-truth token $y^*_{\text{discrete}}[\ell]$.

**Step 6: Total loss.** Sum over all supervision steps:

$$\mathcal{L}_{\text{total}} = \sum_{s=1}^{N_{\text{sup}}} \mathcal{L}_s$$

with gradients detached between steps (backprop only through the last $T$ repetition within each step).

**Key observations:**

1. **$z$ never touches $\mathbf{W}_{\text{vocab}}$**: Only $y$ gets projected to logits. The latent reasoning state $z$ is never decoded to tokens.

2. **Same network, different inputs**: $f_\theta$ is applied with different concatenations — sometimes $[x; y; z]$, sometimes $[y; z]$ — but it's the same 2-layer transformer weights.

3. **Effective depth from repetition**: Even though $f_\theta$ has only 2 layers, applying it $(n+1) \times T \times N_{\text{sup}}$ times creates enormous effective depth through recurrence.

### The Recursion Structure (Summary)

One "supervision step" works as follows. Define the inner recursion:

$$\text{latent\_recursion}(x, y, z) : \quad \begin{cases} z \leftarrow f_\theta([x; y; z]) & \text{repeated } n \text{ times} \\ y \leftarrow f_\theta([y; z]) & \text{once} \end{cases}$$

Then the outer recursion repeats this $T$ times, but only backpropagates through the last repetition:

$$\text{deep\_recursion}(x, y, z): \quad \begin{cases} (y, z) \leftarrow \text{latent\_recursion}(x, y, z) & T - 1 \text{ times, no gradients} \\ (y, z) \leftarrow \text{latent\_recursion}(x, y, z) & \text{once, with gradients} \end{cases}$$

Finally, **deep supervision** runs $N_{\text{sup}}$ supervision steps in sequence, computing a loss after each one:

$$\hat{y} = \mathbf{W}_{\text{vocab}} \, y, \quad \mathcal{L} = \text{CrossEntropy}(\hat{y}, \; y^*)$$

The latent state $(y, z)$ carries over between supervision steps, but gradients are detached. This means each step trains independently while benefiting from the accumulated reasoning of previous steps.

### Effective Depth

With $n = 6$ latent updates, $T = 3$ repetitions, 2 layers, and $N_{\text{sup}} = 16$ supervision steps:

- **Per supervision step**: $T \times (n + 1) \times 2 = 3 \times 7 \times 2 = 42$ effective layers
- **Total**: $42 \times 16 = 672$ effective layers

All from a 2-layer, 7-million-parameter network. The same weights are reused for every single one of those 672 effective layers.

### Why Tiny Networks Win Here

The ablation tells the story:

| Change from TRM | Test Accuracy | Parameters |
|:---|:---:|:---:|
| TRM (2 layers, single network, full backprop) | **87.4%** | **5M** |
| + separate reasoning and answer networks | 82.4% | 10M |
| + 4 layers per network | 79.5% | 10M |
| + self-attention (instead of MLP) | 74.7% | 7M |
| + 1-step gradient approximation | 56.5% | 5M |

Every increase in model capacity **hurts**. Why? With only ~1000 training examples, larger models overfit. Recursion provides depth (and therefore expressiveness) without adding parameters. It's the ideal inductive bias for the low-data regime.

### TRM Loss vs. LLM Loss

Let's make the connection explicit. Recall from `llm_loss.md`:

**Standard LLM (next-token prediction):**
$$\mathcal{L}_{\text{LLM}} = -\frac{1}{N}\sum_{i=1}^{N} \log P(y_i \mid y_1, \ldots, y_{i-1}; \theta)$$

One forward pass, one loss computation, trained on billions of tokens.

**Reasoning LLM (GRPO):**

Same next-token prediction for generating traces, but trained with:
$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G} \text{clip}\left(\frac{\pi_\theta(o_i \mid x)}{\pi_{\theta_{\text{old}}}(o_i \mid x)}\right) A_i + \beta \, D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

Still autoregressive, still predicting tokens, but optimized for downstream reward rather than teacher imitation.

**TRM (recursive refinement):**
$$\mathcal{L}_{\text{TRM}} = \sum_{s=1}^{N_{\text{sup}}} \text{CrossEntropy}\left(\mathbf{W}_{\text{vocab}} \cdot y^{(s)}, \; y^*\right)$$

where $y^{(s)}$ is the answer state after $s$ supervision steps, each involving $T \times (n + 1)$ forward passes through the same 2-layer network. Not autoregressive. No trace in natural language. Just iterative vector refinement.

---

## Part IV: The Two Philosophies Side by Side

| | Autoregressive Reasoning (CoT) | Recursive Refinement (TRM) |
|:---|:---|:---|
| **Architecture changes?** | **No** — same transformer | **Yes** — custom recursive network |
| **Training changes?** | **Yes** — different data or RL | **Yes** — supervised + deep supervision |
| **Start from pretrained model?** | **Yes** — fine-tune a base LLM | **No** — train from scratch |
| **Requires ground-truth labels?** | **Yes** — need correct answers for both distillation and RL | **Yes** — fully supervised $(x, y^*)$ pairs |
| **Generate reasoning during training?** | Distillation: No (provided). RL: Yes (sampled). | No — reasoning stays in latent vectors |
| **Reasoning medium** | Natural language tokens | Learned latent vectors |
| **Reasoning is** | Explicit, human-readable | Implicit, not decodable |
| **Architecture** | Large transformer (7B--671B) | Tiny transformer (2 layers, 7M) |
| **Training paradigm** | Distillation (supervised) or RL (reward-based) | Fully supervised |
| **Data needed** | Billions of tokens (pretraining) + labeled Q&A | ~1000 task-specific labeled examples |
| **Depth** | Sequential token generation | Recursive weight-shared forward passes |
| **Generality** | General-purpose | Task-specific |
| **Error recovery** | Textual backtracking (fragile) | Iterative refinement (robust) |
| **Where it wins** | Language, broad knowledge | Structured puzzles, constraint satisfaction |

### What are Constraint Satisfaction Problems?

A constraint satisfaction problem (CSP) is a **search problem** where you need to find values for variables that satisfy a set of constraints. Unlike optimization problems (which have objectives to maximize/minimize), CSPs just ask: "Is there a valid assignment?"

**Examples:**

- **Sudoku**: Variables are the 81 cells. Constraints are "each row has digits 1-9", "each column has digits 1-9", "each 3×3 block has digits 1-9". Goal: find an assignment of digits to cells that satisfies all constraints.
  
- **Graph coloring**: Variables are nodes. Constraints are "adjacent nodes must have different colors". Goal: color the graph with $k$ colors.

- **Scheduling**: Variables are time slots for tasks. Constraints are "task A must finish before task B", "tasks C and D can't run simultaneously". Goal: find a valid schedule.

- **Boolean satisfiability (SAT)**: Variables are boolean. Constraints are logical clauses like $(x_1 \vee \neg x_2 \vee x_3)$. Goal: find an assignment that makes all clauses true.

**Why TRM is perfect for CSPs:** Each forward pass does one round of **constraint propagation**. When you know cell (1,1) is 5, you can eliminate 5 from the possibilities of all cells in row 1, column 1, and its 3×3 block. Those eliminations cascade — if a cell now has only one possible value, you can fix it and propagate further.

The 2-layer network learns to:
1. Read the current state (which values are still possible for each cell)
2. Apply constraints (eliminate impossible values based on filled cells and constraint logic)
3. Output a refined state for the next iteration

After 672 iterations, all logical implications are exhausted and the puzzle is solved. This is fundamentally different from language modeling — there's no generation, just iterative refinement toward a fixed solution.

### The Common Thread

Despite their differences, both approaches share the same core insight: **reasoning requires depth, and depth can come from iteration rather than parameters.**

For autoregressive models, "iteration" means generating more tokens (test-time compute). For TRM, it means running the same network more times (recursive compute). Both trade FLOPs for effective depth. Both allow smaller models to solve problems that larger, shallower models cannot.

The question is which kind of iteration suits your problem:
- If you need **general-purpose** reasoning with **natural-language** explanations: autoregressive CoT
- If you need **task-specific** reasoning on **structured problems** with **limited data**: recursive refinement

---

## Practical Summary

**If you want to use reasoning models (inference):**

The loss function during inference is irrelevant --- you just sample tokens. But understanding the training loss tells you *why* the model reasons:
- Distilled models reason because they were trained on reasoning traces (cross-entropy)
- RL-trained models reason because reasoning increases reward (GRPO)
- TRM reasons because iterative refinement reduces supervised loss (deep supervision)

**If you want to train reasoning models:**

| Goal | Method | Compute needed |
|:---|:---|:---|
| Add reasoning to a small LLM | Distillation from a teacher | Standard fine-tuning (a few GPUs) |
| Train reasoning from scratch (LLM) | GRPO with verifiable rewards | High (sample $G$ outputs per step) |
| Solve structured puzzles | TRM-style recursive refinement | Low (single GPU, hours) |

**If you're in academia with limited compute:**

TRM is the most accessible. A 5M-parameter model trained in 18 hours on a single L40S that beats 671B-parameter models on puzzle benchmarks. The [TRM blog post](/blog/machine-learning/tiny_recursive_models/) covers the full details, including code to reproduce the results.

---

## Key References

- **DeepSeek-R1**: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) --- GRPO, emergent reasoning, distillation pipeline
- **TRM**: [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) --- recursive refinement with tiny networks
- **The Transformer Loss Function**: [llm_loss](/blog/machine-learning/llm_loss/) --- the foundation this post builds on
- **Open-R1**: [github.com/huggingface/open-r1](https://github.com/huggingface/open-r1) --- open-source reproduction of R1
- **TRL**: [github.com/huggingface/trl](https://github.com/huggingface/trl) --- GRPO implementation
