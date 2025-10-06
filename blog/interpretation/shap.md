@def title = "SHAP and DeepLIFT: A Technical Summary"
@def published = "6 October 2025"
@def tags = ["interpretation"]


# SHAP and DeepLIFT: A Technical Summary

## The Big Picture

SHAP (SHapley Additive exPlanations) gives you a principled way to explain any model's predictions by assigning each feature an importance value—its Shapley value. For deep learning, vanilla SHAP becomes computationally nightmarish because you'd need to evaluate the model on exponentially many feature coalitions. That's where **DeepSHAP** comes in, using **DeepLIFT** as its computational engine.

## What Are Shapley Values Again?

The Shapley value for feature $i$ is:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$

where $F$ is the set of all features, and $S$ is a subset not containing $i$. This measures feature $i$'s marginal contribution across all possible coalitions.

The problem? For $n$ features, you have $2^n$ subsets. Even for modest networks, this is absurd.

## Enter DeepLIFT

DeepLIFT (Deep Learning Important FeaTures) is a backpropagation-based method that computes feature attributions by comparing each neuron's activation to a "reference activation." Think of the reference as a baseline input—often all zeros or the dataset mean.

**The central goal**: DeepLIFT ultimately just linearizes your neural network output as a function of the input. It expresses:

$f(x) - f(r) = \sum_i C_{\Delta x_i \Delta f} = \sum_i \phi_i \cdot (x_i - r_i)$

where $\phi_i$ are the contribution coefficients. So despite all the complexity, you're just decomposing the output change as a weighted sum of input changes. All the machinery below is just computing these weights efficiently through the network structure.

### The Core Idea: Multipliers

DeepLIFT defines **contribution scores** $C_{\Delta x_i \Delta t}$ that measure how much an input feature $x_i$'s difference from reference contributes to an output neuron $t$'s difference from reference.

The key insight is the **multiplier**:

$$m_{\Delta x_i \Delta t} = \frac{C_{\Delta x_i \Delta t}}{\Delta x_i}$$

where $\Delta x_i = x_i - x_i^{\text{ref}}$ is the difference from reference.

### The Chain Rule for Contributions

For a deep network, DeepLIFT uses a modified chain rule. If neuron $t$ depends on neurons $\{s_1, s_2, \ldots\}$, then:

$C_{\Delta x_i \Delta t} = \sum_{j} C_{\Delta x_i \Delta s_j} \cdot m_{\Delta s_j \Delta t}$

This lets you backpropagate contributions through the network.

**What's really happening**: You're finding coefficients $\phi_i$ such that $\Delta t = \sum_i \phi_i \cdot \Delta x_i$. The multipliers are just the tool to compute these coefficients layer by layer, respecting the network's structure.

### Computing Multipliers for Common Layers

**Linear layers**: If $t = \sum_j w_j s_j + b$, then:

$$m_{\Delta s_j \Delta t} = w_j$$

Simple! The weight is the multiplier.

**ReLU (and other activations)**: This is where it gets interesting. For ReLU, if both $s$ and $s^{\text{ref}}$ are positive (or both negative), you get:

$$m_{\Delta s \Delta t} = \frac{\Delta t}{\Delta s}$$

But if they have different signs, DeepLIFT uses the "rescale rule" to handle the discontinuity at zero gracefully.

**MaxPool and other operations**: DeepLIFT distributes contributions based on which neurons were actually responsible for the max operation.

## DeepSHAP: Marrying DeepLIFT with Shapley

DeepSHAP uses DeepLIFT's computational framework but interprets it through the lens of Shapley values. Here's the clever bit:

1. **Multiple references**: Instead of a single reference input, DeepSHAP samples a distribution of reference inputs (often from your training data).

2. **Expected contribution**: For each feature, DeepSHAP computes:

$$\phi_i = \frac{1}{|R|} \sum_{r \in R} C_{\Delta x_i \Delta t}(x, r)$$

where $R$ is your set of reference samples.

3. **Why this works**: Lundberg showed that if you average DeepLIFT contributions over a reference distribution, you approximate the Shapley value under certain assumptions (specifically, when features are independent given the reference).

## The Math That Trips People Up

### Linearization Assumption

DeepLIFT essentially linearizes the model locally around the reference. The multipliers $m_{\Delta s_j \Delta t}$ act like local "effective gradients" but aren't quite the same as actual gradients.

For a function $f$, the contribution satisfies:

$\Delta f = \sum_i C_{\Delta x_i \Delta f}$

This is the **summation-to-delta** property. The contributions sum to exactly the difference in output.

**The key insight**: Even though your neural network $f(x)$ is highly nonlinear, DeepLIFT finds a linear approximation that's exact for the pair $(x, r)$:

$f(x) - f(r) = \sum_i \phi_i(x, r) \cdot (x_i - r_i)$

The coefficients $\phi_i$ depend on both $x$ and $r$, but the relationship is linear. This is different from a Taylor expansion (which uses gradients and is only locally accurate)—DeepLIFT's linearization is exact for that specific input-reference pair by construction.

### The Reference Distribution Trick

When you average over references, you're computing:

$$\phi_i(f, x) = \mathbb{E}_{r \sim \mathcal{R}}[C_{\Delta x_i \Delta f}(x, r)]$$

This expectation connects to the Shapley value through the formula:

$$\phi_i = \mathbb{E}_{S \sim \pi}[f(S \cup \{i\}) - f(S)]$$

where $\pi$ is a specific distribution over coalitions. The reference distribution approximates this coalition sampling.

### Why Not Just Use Gradients?

Gradients $\frac{\partial f}{\partial x_i}$ tell you the local rate of change, but:
- They don't satisfy the summation-to-delta property
- They can be zero even when a feature is important (e.g., in saturated regions)
- They don't naturally handle interactions between features

DeepLIFT's contributions respect the actual function behavior between reference and input.

## Practical Computation

Here's the algorithm flow:

1. **Forward pass**: Compute activations for your input $x$
2. **Reference forward pass**: Compute activations for reference $r$
3. **Compute differences**: $\Delta s_j = s_j - s_j^{\text{ref}}$ for each neuron
4. **Backward pass with multipliers**: Starting from the output, backpropagate using $C_{\Delta x_i \Delta s_j} = \sum_k C_{\Delta x_i \Delta s_k} \cdot m_{\Delta s_k \Delta s_j}$
5. **Average over references**: Repeat for multiple $r$ and average

The beauty is that this is a single backward pass per reference—vastly cheaper than $2^n$ model evaluations.

## Caveats and Nuances

- **Reference choice matters**: Your baseline significantly affects the attributions. Common choices: zero input, dataset mean, or adversarial samples.
- **Approximation quality**: DeepSHAP is exact for linear models but approximate for nonlinear ones. The approximation introduces bias relative to true interventional Shapley values, trading accuracy for computational speed. **Crucially, the approximation quality scales with the number of reference samples**—more references give you a better estimate of the true Shapley value, but each reference requires a full forward+backward pass. This is the fundamental tradeoff: accuracy vs. computational cost.
- **Independence assumption**: The theoretical connection to Shapley values holds exactly when features are independent conditioned on the reference distribution. In practice, this assumption is often violated, but the approximation tends to remain reasonable for smooth networks.
- **No formal convergence guarantees**: Unlike some Monte Carlo methods, there aren't tight convergence bounds on how many references you need for a given approximation error. The quality depends on the reference distribution, network architecture, and local smoothness.
- **Interactions**: Deep networks have feature interactions. DeepLIFT/DeepSHAP naturally handles these through the network's structure.

## The Takeaway

SHAP gives you theoretically grounded feature importance via Shapley values. DeepSHAP makes this tractable for neural networks by using DeepLIFT's backpropagation-style contribution computation, averaged over reference inputs. The multipliers are the secret sauce—they let you efficiently trace how input differences flow through the network to produce output differences, all while maintaining mathematical guarantees about how contributions sum up.

The Lundberg paper's equations become clearer when you see DeepLIFT as defining a new type of "gradient" (the multipliers) that respects the network's discrete structure and nonlinearities better than vanilla gradients do.