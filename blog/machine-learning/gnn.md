@def title = "Graph Neural Networks: Step-by-Step with Simple Examples"
@def published = "9 October 2025"
@def tags = ["machine-learning"]

# Graph Neural Networks: Step-by-Step with Simple Examples

## Simple Example: Two Connected Nodes

Let's say we have a graph with **2 nodes** that are **connected** to each other.

### The Input

**Node Features** - Each node has a feature vector:
- Node 1: $h_1 = [1, 2]$ (a 2-dimensional vector)
- Node 2: $h_2 = [3, 4]$ (a 2-dimensional vector)

We stack these into a **feature matrix**:

$$H = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

Each row is a node's features!

### Edge Information: The Adjacency Matrix

**This is how we encode the graph structure!**

$$A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$

Reading this matrix:
- $A_{12} = 1$ means node 1 connects to node 2
- $A_{21} = 1$ means node 2 connects to node 1  
- $A_{11} = 0$ means node 1 doesn't connect to itself (no self-loop)

**The adjacency matrix IS how edge information is handled!** It tells us who talks to whom.

## Step-by-Step: One GNN Layer

### Step 1: Aggregate Neighbor Features

Multiply the adjacency matrix by the feature matrix:

$$AH = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix}$$

**What just happened?**
- Row 1 of $AH$ is $[3, 4]$ = the features of node 1's neighbor (node 2)
- Row 2 of $AH$ is $[1, 2]$ = the features of node 2's neighbor (node 1)

Each node now has its **neighbor's information**!

### Step 2: Transform the Aggregated Features

Apply a learnable weight matrix. Let's say:

$$W = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix}$$

Then:

$$(AH)W = \begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix} = \begin{bmatrix} 1.5 & 2 \\ 0.5 & 1 \end{bmatrix}$$

This transforms the neighbor features through learned weights.

### Step 3: Apply Nonlinearity

$$H^{(1)} = \sigma\left(\begin{bmatrix} 1.5 & 2 \\ 0.5 & 1 \end{bmatrix}\right)$$

If $\sigma$ is ReLU (which keeps positive values), we get:

$$H^{(1)} = \begin{bmatrix} 1.5 & 2 \\ 0.5 & 1 \end{bmatrix}$$

**These are the new node features after one GNN layer!**

## Adding Self-Loops: A More Realistic Version

Usually we want nodes to keep their **own information** too, not just neighbor info!

Add self-loops by modifying the adjacency matrix:

$$\tilde{A} = A + I = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} + \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$$

Now let's redo step 1:

$$\tilde{A}H = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 4 & 6 \\ 4 & 6 \end{bmatrix}$$

**What happened?**
- Row 1: $[1,2] + [3,4] = [4,6]$ (node 1's features + node 2's features)
- Row 2: $[1,2] + [3,4] = [4,6]$ (node 2's features + node 1's features)

Each node now aggregates **itself and its neighbors**!

## Complete Formula

The full GNN layer operation is:

$H^{(l+1)} = \sigma\left(\tilde{A} H^{(l)} W^{(l)}\right)$

Where:
- $H^{(l)}$ = node features at layer $l$
- $\tilde{A}$ = adjacency matrix (with self-loops) - **this is the edge information!**
- $W^{(l)}$ = learnable weight matrix
- $\sigma$ = activation function

---

### 📘 Side Note: What Does $HW$ Mean Mathematically?

When we multiply features by weights: $H^{(l)} W^{(l)}$

**Dimensions:**
- $H^{(l)} \in \mathbb{R}^{n \times d}$ (n nodes, d features per node)
- $W^{(l)} \in \mathbb{R}^{d \times d'}$ (transforms d features to d' features)
- Result: $H^{(l)}W^{(l)} \in \mathbb{R}^{n \times d'}$

**Operation-wise, for each node $i$:**

$\left(HW\right)_i = h_i W = \sum_{j=1}^{d} h_{ij} W_{j:}$

This is a **linear combination** of the columns of $W$, weighted by the node's features!

**Concrete Example:**

Say node 1 has features $h_1 = [2, 3]$ and we have:

$W = \begin{bmatrix} 0.5 & 1 \\ 0.2 & 0.8 \end{bmatrix}$

Then:

$h_1 W = [2, 3] \begin{bmatrix} 0.5 & 1 \\ 0.2 & 0.8 \end{bmatrix} = [2 \cdot 0.5 + 3 \cdot 0.2, \; 2 \cdot 1 + 3 \cdot 0.8] = [1.6, 4.4]$

**What's happening?**
- We're projecting the 2D features into a new 2D space
- Each output dimension is a weighted sum of input features
- The weights in $W$ are **learned during training** to extract useful patterns

**Intuition:**
- Think of $W$ as a "feature mixer" or "lens"
- It learns which combinations of input features are important
- Similar to a fully-connected layer in a neural network
- If $d' < d$: dimensionality reduction (compression)
- If $d' > d$: dimensionality expansion (learning more complex representations)

**Why is this useful in GNNs?**

After $\tilde{A}H$ aggregates neighbor features, $W$ learns:
- Which aggregated patterns matter for the task
- How to transform raw features into more abstract representations
- Different "channels" or "filters" (like in CNNs, but for graphs)

---

### 🔍 Side-Side Note: Order Matters! $\tilde{A}(HW)$ vs $(\tilde{A}H)W$

Wait, these give the **same result** by associativity of matrix multiplication:

$\tilde{A}(HW) = (\tilde{A}H)W$

But they have **different interpretations**:

**Option 1: $(\tilde{A}H)W$ — Aggregate then Transform**
1. First: $\tilde{A}H$ aggregates raw neighbor features
2. Then: multiply by $W$ to transform the aggregated features

**Option 2: $\tilde{A}(HW)$ — Transform then Aggregate**
1. First: $HW$ transforms each node's features independently
2. Then: $\tilde{A}(...)$ aggregates the transformed features from neighbors

**Implications:**

| Aspect | Aggregate→Transform | Transform→Aggregate |
|--------|---------------------|---------------------|
| **Computation** | Same result! | Same result! |
| **Interpretation** | Pool raw features, learn from pooled | Each node transforms first, then share |
| **Parameters** | One $W$ for all nodes | One $W$ for all nodes |
| **Expressiveness** | Equivalent | Equivalent |

**But here's the KEY insight:**

Transform-then-aggregate $(\tilde{A}(HW))$ is more intuitive:
- Each node prepares a "message" by transforming its features: $m_i = h_i W$
- Then neighbors aggregate these messages: $\sum_{j \in \mathcal{N}(i)} m_j$

This is the **message passing** view! Each node:
1. Creates an outgoing message from its features
2. Receives and sums messages from neighbors
3. Updates its representation

**More expressive architectures:**

What if we use **different weight matrices** for self vs. neighbors?

$h_i^{(l+1)} = \sigma\left(W_{\text{self}} h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} W_{\text{neighbor}} h_j^{(l)}\right)$

Now we can't write it as a simple $\tilde{A}HW$ anymore! We need:
- Separate transformation for the node itself
- Separate transformation for aggregated neighbors

This is what **GraphSAGE** and other advanced GNNs do!

**Computational considerations:**

$\tilde{A}(HW)$ can be more efficient:
- Compute $HW$ once (cheap: $O(nd \cdot dd')$)
- Then do sparse $\tilde{A} \times (...)$ (depends on edges)

vs $(\tilde{A}H)W$:
- Sparse $\tilde{A}H$ first (depends on edges)  
- Then dense $(...)W$ (same cost)

For sparse graphs, the order doesn't matter much. For dense graphs, transform-first can be faster!

**🤔 Remark: Why multiply by $\tilde{A}$ if $W$ already linearly combines features?**

Great question! Here's the crucial difference:

**$W$ operates on features (columns)** — it mixes feature dimensions:
- Takes feature 1 and feature 2 and creates new combined features
- Same transformation for EVERY node
- Example: $[x_1, x_2] W$ → combines the two features into new features

**$\tilde{A}$ operates on nodes (rows)** — it mixes information ACROSS nodes:
- Takes node 1's features and node 2's features and combines them
- Different aggregation for EACH node (depends on graph structure)
- Example: $\tilde{A} H$ → node $i$ gets a weighted sum of its neighbors' feature vectors

**Without $\tilde{A}$:** Just $H^{(l+1)} = \sigma(HW)$
- This is a standard fully-connected layer
- Each node updates independently
- **No information flows through the graph!**
- Equivalent to treating each node in isolation

**With $\tilde{A}$:** $H^{(l+1)} = \sigma(\tilde{A}HW)$
- The adjacency matrix enforces graph structure
- Only connected nodes exchange information
- This is what makes it a **Graph** Neural Network!

**Analogy:**
- $W$: A personal translator that translates your own thoughts into a new language (same for everyone)
- $\tilde{A}$: A phone network that connects you to specific people — you can only hear from those you're connected to

You need BOTH:
1. The network ($\tilde{A}$) to route information
2. The transformation ($W$) to make that information useful

---

## Three Node Example

Let's do one more! Three nodes in a line: 1 — 2 — 3

**Initial features:**
$$H^{(0)} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$

(Each node has a 1D feature for simplicity)

**Adjacency with self-loops:**
$$\tilde{A} = \begin{bmatrix} 1 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 1 \end{bmatrix}$$

- Node 1 connects to: itself, node 2
- Node 2 connects to: itself, node 1, node 3
- Node 3 connects to: itself, node 2

**After aggregation:**
$$\tilde{A}H^{(0)} = \begin{bmatrix} 1 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 3 \\ 6 \\ 5 \end{bmatrix}$$

Breaking it down:
- Node 1: $1 + 2 = 3$ (itself + neighbor 2)
- Node 2: $1 + 2 + 3 = 6$ (itself + neighbors 1 and 3)
- Node 3: $2 + 3 = 5$ (itself + neighbor 2)

**Node 2 has the most information because it has the most neighbors!**

## Key Insights

1. **Node features** = rows in matrix $H$
2. **Edge information** = encoded in adjacency matrix $A$
3. **$AH$ operation** = each node aggregates neighbor features
4. **Weight matrix $W$** = learnable transformation
5. **Multiple layers** = information flows further across the graph

## What About Edge Features?

Sometimes edges have their own features (e.g., distance, weight, type). For this:

- **Simple approach**: Use weighted adjacency matrix where $A_{ij}$ = edge weight
- **Advanced approach**: Message Passing Neural Networks (MPNNs) that explicitly compute:

$$m_{ij} = \phi(h_i, h_j, e_{ij})$$

where $e_{ij}$ is the edge feature. Then aggregate these messages.

But the basic GNN doesn't use edge features—just whether edges exist (in $A$)!