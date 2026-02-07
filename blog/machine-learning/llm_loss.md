@def title = "The Transformer Loss Function"
@def published = "7 February 2026"
@def tags = ["machine-learning"]

# The Transformer Loss Function

## The Setup

Most tutorials skip the details of how transformers actually compute loss during training. The confusion typically arises because tokens are *vectors* (groups of neurons), yet we need to predict *discrete* token IDs from a vocabulary. Let's clarify this.

### Data Structures

Following the notation from transformers literature, we represent a sequence of tokens as a matrix where each token is a row:

$$\mathbf{T} = \begin{bmatrix}
\mathbf{t}_1^\mathsf{T}\\
\vdots \\
\mathbf{t}_N^\mathsf{T}\\
\end{bmatrix} \in \mathbb{R}^{N \times d}$$

where each $\mathbf{t}_i \in \mathbb{R}^{d \times 1}$ is a token's code vector with dimensionality $d$.

### The Output Layer

The transformer processes these tokens through $L$ layers of attention and MLPs. At the final layer, we have:

$$\mathbf{T}_{\text{out}} = \text{transformer}(\mathbf{T}_{\text{in}}) \quad \quad \triangleleft \quad \mathbf{T}_{\text{out}} \in \mathbb{R}^{N \times d}$$

But how do we go from these $d$-dimensional token vectors to predictions over our vocabulary of size $V$?

## From Tokens to Logits

We add a **linear projection layer** that maps each token to logit scores over the vocabulary:

$$\mathbf{z}_i = \mathbf{W}_{\text{vocab}} \mathbf{t}_i \quad \quad \triangleleft \quad \mathbf{W}_{\text{vocab}} \in \mathbb{R}^{V \times d}$$

For the full sequence, this is:

$$\mathbf{Z} = \mathbf{T}_{\text{out}} \mathbf{W}_{\text{vocab}}^\mathsf{T} \quad \quad \triangleleft \quad \mathbf{Z} \in \mathbb{R}^{N \times V}$$

Each row $\mathbf{z}_i^\mathsf{T}$ contains unnormalized log-probabilities (logits) for all $V$ possible tokens at position $i$.

### What Are Logits?

**Logits** are simply the raw, unnormalized scores before applying softmax. Think of them as "pre-probabilities":

- **Dimension of a single logit vector**: $\mathbf{z}_i \in \mathbb{R}^{V \times 1}$ — one score for each token in the vocabulary
- **Dimension of all logits**: $\mathbf{Z} \in \mathbb{R}^{N \times V}$ — one logit vector for each position in the sequence

For example, if your vocabulary has $V = 50{,}000$ tokens, then $\mathbf{z}_i$ is a vector of 50,000 numbers:

$$\mathbf{z}_i = \begin{bmatrix}
2.3 \quad \leftarrow \text{score for token "the"}\\
-1.5 \quad \leftarrow \text{score for token "cat"}\\
4.7 \quad \leftarrow \text{score for token "sat"}\\
\vdots\\
0.2 \quad \leftarrow \text{score for token "zebra"}
\end{bmatrix}$$

These raw scores can be any real numbers (positive, negative, large, small). We convert them to probabilities via **softmax**:

$$P(\text{token}_j) = \frac{e^{z_i[j]}}{\sum_{k=1}^{V} e^{z_i[k]}}$$

The softmax ensures:
- All probabilities are between 0 and 1
- All probabilities sum to 1
- Higher logits → higher probabilities

## The Training Task: Next-Token Prediction

For language modeling, we use **causal (masked) attention** to predict each token from only previous tokens. Given an input sequence, we simultaneously predict:

$$\begin{aligned}
\mathbf{t}_1 &\rightarrow \mathbf{t}_2\\
\{\mathbf{t}_1, \mathbf{t}_2\} &\rightarrow \mathbf{t}_3\\
&\vdots\\
\{\mathbf{t}_1, \ldots, \mathbf{t}_{N-1}\} &\rightarrow \mathbf{t}_N
\end{aligned}$$

This means our targets are simply the input sequence **shifted by one position**.

### Handling Variable-Length Sequences

In practice, different sentences have different lengths, but neural networks require **fixed-size tensors**. Here's how this is handled:

**Option 1: Padding (most common)**

Choose a maximum sequence length $N_{\text{max}}$ and pad shorter sequences:

```
Sentence 1: "The cat sat"           → length 3
Sentence 2: "The dog ran away"      → length 4

After padding to N_max = 5:
Sentence 1: ["The", "cat", "sat", "<pad>", "<pad>"]
Sentence 2: ["The", "dog", "ran", "away", "<pad>"]
```

In matrix form, both become $\mathbf{T} \in \mathbb{R}^{5 \times d}$. We then use an **attention mask** to ignore padded positions:

$$\mathbf{A}_{ij} = \begin{cases}
\text{softmax}(\mathbf{q}_i^\mathsf{T}\mathbf{k}_j / \sqrt{m}) & \text{if position } j \text{ is not padded}\\
0 & \text{if position } j \text{ is padded}
\end{cases}$$

And critically, we **don't compute loss** on padded positions:

$$\mathcal{L} = -\frac{1}{N_{\text{actual}}} \sum_{i=1}^{N_{\text{actual}}} \log P(y_i \mid \mathbf{t}_1, \ldots, \mathbf{t}_{i-1})$$

where $N_{\text{actual}}$ is the true sequence length (before padding).

**Option 2: Batching by Length**

Group sequences of similar length together, minimizing wasted computation on padding:

```
Batch 1 (length 3-4):  N_max = 4
Batch 2 (length 10-12): N_max = 12
```

**Option 3: Dynamic Batching**

Some frameworks (like modern PyTorch with `torch.nn.utils.rnn.pack_padded_sequence`) can handle variable-length sequences more efficiently, but under the hood it's still padding with masking.

### Concrete Example

Let's say we're training on the sequence: `"The cat sat"`

After tokenization, we have token vectors representing each word. The training setup looks like:

**Input sequence:** $[\mathbf{t}_{\text{The}}, \mathbf{t}_{\text{cat}}, \mathbf{t}_{\text{sat}}]$  
**Target sequence:** $[\mathbf{t}_{\text{cat}}, \mathbf{t}_{\text{sat}}, \mathbf{t}_{\text{end}}]$ ← shifted by one!

If we pad to $N_{\text{max}} = 5$:

**Input (padded):** $[\mathbf{t}_{\text{The}}, \mathbf{t}_{\text{cat}}, \mathbf{t}_{\text{sat}}, \mathbf{t}_{\text{pad}}, \mathbf{t}_{\text{pad}}]$  
**Target (padded):** $[\mathbf{t}_{\text{cat}}, \mathbf{t}_{\text{sat}}, \mathbf{t}_{\text{end}}, \mathbf{t}_{\text{pad}}, \mathbf{t}_{\text{pad}}]$

At each position, we predict the next token:
1. **Position 1:** Given $[\mathbf{t}_{\text{The}}]$, predict token ID for `"cat"`
2. **Position 2:** Given $[\mathbf{t}_{\text{The}}, \mathbf{t}_{\text{cat}}]$, predict token ID for `"sat"`
3. **Position 3:** Given $[\mathbf{t}_{\text{The}}, \mathbf{t}_{\text{cat}}, \mathbf{t}_{\text{sat}}]$, predict token ID for `"end"`
4. **Positions 4-5:** Ignored (padding, no loss computed)

The transformer outputs token vectors at each position, then $\mathbf{W}_{\text{vocab}}$ projects these to logits over all vocabulary tokens.

## The Loss Function

For each position $i$, we convert logits to probabilities via softmax, then compute cross-entropy with the true next token $y_i \in \{1, \ldots, V\}$:

$$\mathcal{L}_i = -\log \frac{e^{z_i[y_i]}}{\sum_{j=1}^{V} e^{z_i[j]}} = -\log P(y_i \mid \mathbf{t}_1, \ldots, \mathbf{t}_{i-1})$$

where $z_i[y_i]$ is the logit corresponding to the correct token ID.

### Concrete Worked Example

Let's work through this with tiny numbers. Suppose we have a vocabulary of just $V = 4$ tokens:
- Token 0: `"the"`
- Token 1: `"cat"` 
- Token 2: `"sat"`
- Token 3: `"end"`

At position $i=1$, after processing `"The"`, our model outputs a token vector $\mathbf{t}_1 \in \mathbb{R}^{d \times 1}$. The vocabulary projection gives us logits:

$$\mathbf{z}_1 = \mathbf{W}_{\text{vocab}} \mathbf{t}_1 = \begin{bmatrix}
0.5 \quad \leftarrow \text{logit for "the" (token 0)}\\
2.0 \quad \leftarrow \text{logit for "cat" (token 1)}\\
-1.0 \quad \leftarrow \text{logit for "sat" (token 2)}\\
0.1 \quad \leftarrow \text{logit for "end" (token 3)}
\end{bmatrix}$$

The correct next token is `"cat"`, so $y_1 = 1$. Now we compute the loss:

**Step 1: Compute softmax denominators**

$$\sum_{j=0}^{3} e^{z_1[j]} = e^{0.5} + e^{2.0} + e^{-1.0} + e^{0.1} = 1.65 + 7.39 + 0.37 + 1.11 = 10.52$$

**Step 2: Compute probability of correct token**

$$P(y_1 = 1) = \frac{e^{z_1[1]}}{\sum_{j=0}^{3} e^{z_1[j]}} = \frac{e^{2.0}}{10.52} = \frac{7.39}{10.52} \approx 0.70$$

**Step 3: Compute loss**

$$\mathcal{L}_1 = -\log(0.70) \approx 0.36$$

The model assigned 70% probability to the correct token `"cat"`, giving us a fairly low loss.

**If the model had been wrong:**

Suppose it gave logits $\mathbf{z}_1 = [2.0, -1.0, 0.5, 0.1]^\mathsf{T}$. Then:
- Correct token `"cat"` gets $e^{-1.0} / \sum e^{z_j} \approx 0.05$ probability (only 5%!)
- Loss becomes $\mathcal{L}_1 = -\log(0.05) \approx 3.0$ (much higher!)

### What This Means for a Single Position

If the correct token at position $i$ is token #42 in our vocabulary (say, the token ID for `"cat"`):

$$\mathcal{L}_i = -\log\left(\frac{e^{z_i[42]}}{\sum_{j=1}^{V} e^{z_i[j]}}\right)$$

This loss is:
- **Small** when the model assigns high probability to token #42 (correct!)
- **Large** when the model assigns low probability to token #42 (incorrect)
- The $\log$ makes gradients smooth and well-behaved for optimization

The **full loss** averages over all positions:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_i = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid \mathbf{t}_1, \ldots, \mathbf{t}_{i-1})$$

This is equivalently written as:

$$\boxed{\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[\mathbf{z}_i[y_i] - \log \sum_{j=1}^{V} e^{\mathbf{z}_i[j]}\right]}$$

## Why Cross-Entropy?

The cross-entropy loss naturally arises from maximum likelihood estimation. We're finding parameters $\theta$ that maximize:

$$P(\mathbf{y} \mid \mathbf{T}_{\text{in}}; \theta) = \prod_{i=1}^{N} P(y_i \mid \mathbf{t}_1, \ldots, \mathbf{t}_{i-1}; \theta)$$

Taking the negative log gives us our loss (and turns products into sums):

$$\mathcal{L} = -\log P(\mathbf{y} \mid \mathbf{T}_{\text{in}}; \theta)$$

The softmax ensures probabilities sum to 1, and the log makes gradients well-behaved.

## Implementation

In practice, we compute this efficiently via a single matrix operation:

```python
# Z: (batch, seq_len, vocab_size) - logits for each position
# y: (batch, seq_len) - target token IDs

# Flatten for cross-entropy computation
Z_flat = Z.view(-1, V)          # (batch * seq_len, vocab_size)
y_flat = y.view(-1)             # (batch * seq_len,)

# Cross-entropy does softmax + negative log-likelihood internally
loss = F.cross_entropy(Z_flat, y_flat)
```

The key insight: while tokens are continuous $d$-dimensional vectors throughout the network, the final layer projects them to discrete predictions over the vocabulary. The loss measures how well these predictions match the true next tokens.