@def title = "Unconventional Approaches to Interpretability in Flat Minima"
@def published = "5 November 2025"
@def tags = ["machine-learning"]

# Unconventional Approaches to Interpretability in Flat Minima

You're right that ensemble/averaging answers are obvious. Here are genuinely unconventional approaches that have shown promise but aren't mainstream knowledge:

## 1. Sharpness-Aware Minimization (SAM) and Variants

Instead of just finding flat minima, SAM explicitly seeks them during training by perturbing parameters in directions that increase loss, then minimizing at those perturbed points. The counterintuitive part: **intentionally making optimization harder improves generalization AND interpretability**. Flat minima found this way tend to have more coherent feature representations.

**Why unconventional**: You're adding computational cost to make training "worse" in the short term. Yet SAM-trained models often have more interpretable neurons and cleaner feature decompositions.

> **Sources and Common Approaches**: The original SAM paper is Foret et al. (2021) "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR). Common variants include: ASAM (Adaptive SAM), GSAM (Gradient SAM), LookSAM, and Fisher-SAM. The interpretability claim is more speculative - it comes from empirical observations in Jiang et al. (2022) on feature collapse and some mechanistic interpretability work, but isn't the main focus of SAM literature.
> 
> **On reproducibility concerns**: This is a sharp observation! SAM doesn't actually decrease reproducibility in practice because: (1) You're not randomly searching - you're doing adversarial perturbation in a principled direction (gradient of loss), (2) The perturbation radius is small and controlled, (3) Empirically, SAM training is quite stable and reaches similar minima across runs. The paradox: making each step "harder" makes the overall trajectory MORE reproducible because you're constrained to flatter regions from the start, which are larger basins. Sharp minima are actually less reproducible because small initialization/stochasticity differences matter more.

## 2. Linear Mode Connectivity as an Interpretability Probe

Two independently trained networks in flat minima can often be connected by a near-linear path of low-loss solutions. **The structure of this path reveals functional organization** - which features are fundamental vs. which are implementation details.

**Controversial application**: Train multiple models, identify the "invariant core" along the connecting path. This core is often far more interpretable than any single model. It's like crystallizing out the essential computation from implementation noise.

> **Making this concrete**: Yes, it's interpolation but not naive linear interpolation. The key papers are Frankle et al. (2020) "Linear Mode Connectivity and the Lottery Ticket Hypothesis" and Garipov et al. (2018) on loss surface connectivity. 
> 
> **How to interpolate**: For two networks with parameters θ₁ and θ₂, you compute θ(α) = (1-α)θ₁ + αθ₂ for α ∈ [0,1]. Sometimes you need to align neurons first (permutation invariance problem) using techniques like weight matching or optimal transport.
> 
> **How to define connectivity**: Two minima are "connected" if the interpolated path θ(α) maintains low loss (and ideally high accuracy) for all α. Formally: max_α L(θ(α)) - min{L(θ₁), L(θ₂)} < ε for small ε.
> 
> **Finding the invariant core**: The practical approach is to: (1) Sample points along the connected path, (2) For each neuron/feature, measure its activation patterns across all path points, (3) Features with low variance across the path are "invariant" - they're necessary for the computation. Features with high variance are degenerate solutions. This is still somewhat ad-hoc and an active research area.
> 
> **Reality check**: You're absolutely right - this is a ton of engineering. The neuron alignment problem alone can be computationally expensive and brittle. The activation pattern analysis requires careful design choices (which layers? which datasets? what counts as "low variance"?). This is probably why it hasn't taken off despite being theoretically interesting. It's more of a research curiosity than a practical tool right now.

## 3. Lottery Ticket Rewinding to Convergence Points

Take a pruned lottery ticket, but instead of rewinding to initialization, rewind to just before it enters the flat basin. Train multiple times from there. **The variance in final solutions within the same basin tells you which features are "real" vs. spurious**.

**Tail-end insight**: Features that appear consistently across basin re-trainings are functionally necessary. Those that vary are degeneracies of flatness.

> **Computational reality**: Finding lottery tickets is **very computationally intensive**. The standard algorithm (Frankle & Carbin 2019):
> 
> 1. Train a network to completion
> 2. Prune lowest-magnitude weights (say, remove 20%)
> 3. Rewind to initialization, train again with the pruned mask
> 4. Repeat steps 2-3 iteratively (iterative magnitude pruning)
> 
> This requires training the full network **multiple times** (often 10-20+ iterations to reach high sparsity). Each iteration requires full training from scratch.
> 
> **The rewinding variant is even worse**: You need to:
> 1. Find the lottery ticket (already expensive)
> 2. Identify when training enters the flat basin (requires loss landscape analysis during training)
> 3. Train multiple times from that rewind point
> 
> **Practical verdict**: This is probably the most computationally expensive suggestion on the list. It's really only feasible for small-scale experiments (MNIST, small CNNs on CIFAR). For modern large models, the cost is prohibitive. The original lottery ticket hypothesis itself is already considered too expensive for practical use, and this adds another layer of cost on top.

## 4. Hessian Eigenspace Clustering

Compute the Hessian's top eigenvectors at your flat minimum. **Features that consistently align with low-eigenvalue directions are the ones "protected" by flatness** - they're robust to perturbation. Features orthogonal to these directions are typically more interpretable.

**Why it works**: Flatness isn't uniform. The basin has structure - some directions are flatter than others. The flat directions encode the semantically meaningful aspects of the solution.

> **Feasibility concern**: You're absolutely right - computing the full Hessian for neural networks is completely infeasible (O(p²) memory for p parameters). Practical approaches use approximations:
> 
> **Power iteration with Hessian-vector products**: You can compute the top-k eigenvectors using only Hessian-vector products Hv, which can be computed efficiently via automatic differentiation without forming the full matrix. This is O(p) memory. PyTorch's `torch.autograd.functional.hvp` does this.
> 
> **Lanczos algorithm**: Iteratively builds an approximation to the top eigenspace using only matrix-vector products. This is the standard approach in papers like Yao et al. (2020) "PYHESSIAN: Neural Networks Through the Lens of the Hessian."
> 
> **Hutchinson trace estimator + stochastic power iteration**: For even cheaper approximations, sample random directions and estimate eigenvalues statistically.
> 
> **The honest truth**: Even with these tricks, Hessian analysis for large models (millions of parameters) is still expensive and often limited to studying small models or subnetworks. Some recent work tries to use gradient covariance as a first-order proxy for Hessian structure, but it's much less principled.

> **How to actually use this**: This is primarily a **post-hoc analysis tool**, not something used during training (you're right that would be hopeless). The workflow is:
> 
> 1. Train your model normally
> 2. At the final minimum, compute top-k Hessian eigenvectors (say k=10-100)
> 3. For interpretability: Project individual neurons' parameter vectors onto these eigendirections. Neurons that align with low-eigenvalue directions are the "robust" features worth interpreting
> 4. Alternative use: Perturb the model along different eigendirections and see which neurons' activations change. Those that are stable under flat-direction perturbations are interpretable
>
> **Practical value**: Honestly limited. It's more useful for theoretical understanding of why some neurons are interpretable than as a practical interpretability method. The computation cost usually isn't worth it compared to just doing standard feature visualization or probing.

## 5. Noise Injection with Directional Constraints

Instead of standard dropout/noise, inject noise specifically in the flat directions of parameter space (identified via Hessian). **Force the model to maintain functionality despite maximum allowed perturbation in flat directions**. This "squeezes out" degeneracy.

**Counterintuitive result**: Models become more interpretable because they can't rely on arbitrary linear combinations of features - they must use robust, separable features.

## 6. Temperature-Dependent Analysis

This is genuinely weird: Take your trained model and analyze it at different "temperatures" (scaling the logits). **Different temperatures in flat minima reveal different aspects of the learned function**. Low temperature exposes fine-grained features; high temperature reveals coarse structure.

**Why promising**: The temperature-trajectory through decision boundaries in flat basins is often more interpretable than any single temperature snapshot.

> **Making this concrete**: Temperature T scales the logits before softmax: p(y|x) = softmax(f(x)/T). 
> 
> **Low temperature (T→0)**: The model becomes increasingly confident, making sharp decisions. It reveals fine-grained distinctions the model has learned. Example: distinguishing between different dog breeds, or subtle texture differences.
> 
> **High temperature (T→∞)**: The model becomes uncertain, outputs approach uniform. It reveals what the model considers "fundamentally different" categories. Example: dog vs. cat vs. car - the coarsest semantic groupings.
> 
> **The interpretability claim**: In flat minima, as you vary T, you can trace how the model hierarchically organizes concepts. Features that only matter at low T are "refinement features" (breed-specific whisker patterns). Features that matter even at high T are "core semantic features" (presence of fur, basic shape).
> 
> **Honest assessment**: This is extremely hand-wavy. The connection to flatness specifically is tenuous - temperature analysis works for any model. The "hierarchical organization" interpretation requires a lot of human judgment about what counts as fine vs. coarse. I included it because it's unconventional, but it's probably the weakest suggestion on this list in terms of rigorous methodology.

## 7. Algorithmic Alignment Probing

Train a tiny, constrained network (say, a shallow decision tree or sparse linear model) to mimic your overparameterized model's behavior. **The disagreements reveal where flatness enables degeneracy**. The agreements reveal the "true" algorithm being computed.

**Controversial claim**: The interpretable model's decision boundaries often bisect the flat basin in meaningful ways, suggesting the flat region contains many "functionally similar but mechanistically different" solutions.

## Most Promising for Experiments

I'd rank **SAM variants + Hessian eigenspace analysis** as highest success probability because:

- SAM is already proven to find qualitatively different minima
- Hessian eigenspace is computationally feasible and theoretically grounded
- Combined, they let you both "aim for" interpretable solutions and "diagnose" why they're interpretable

The **lottery ticket rewinding variant (#3)** is probably most underexplored and could yield surprising empirical insights with relatively simple experiments.