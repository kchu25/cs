@def title = "Understanding Attribution & Benchmarking in Deep Learning"
@def published = "4 January 2026"
@def tags = ["interpretation"]

# Understanding Attribution & Benchmarking in Deep Learning

## Quick Paper References
- **SHAP Paper (1705.07874)**: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- **Integrated Gradients (1703.01365)**: [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)

---

## What is "Attribution" Mathematically?

Think of it this way: your neural network outputs some prediction $f(x)$ for input $x = (x_1, x_2, ..., x_n)$. **Attribution** answers: *"How much did each input feature $x_i$ contribute to this prediction?"*

### The Mathematical Definition

An attribution method assigns an **importance score** $\phi_i$ to each feature $i$:

$$\phi_1, \phi_2, ..., \phi_n \quad \text{where} \quad \phi_i \in \mathbb{R}$$

These scores tell you which pixels (in images), words (in text), or features (in tabular data) "matter most" for a prediction.

### Why It's Tricky

Here's the problem: there's **no ground truth**. We don't actually *know* which features are truly important! The model is a black box, and we're trying to peek inside.

---

## Benchmarking Setup: Same Model, Different Explainers

> **Quick answer**: Yes! They train **one model** (e.g., a VGG network or random forest), then apply **different attribution methods** to the same model on the same inputs and compare the explanations.
> 
> **Why this makes sense**: You're not testing which model is better—you're testing which *explanation method* better reveals what the model is doing. It's like having one painting and asking different art critics to explain it. The painting doesn't change; only the critics' interpretations differ.
> 
> **Typical setup**:
> 1. Train a single model (ResNet on ImageNet, for example)
> 2. Pick a test image (say, a photo of a dog)
> 3. Run multiple methods on the same prediction:
>    - SHAP values
>    - Integrated Gradients  
>    - LIME
>    - Gradient × Input
>    - DeepLIFT
> 4. Compare their attribution maps
> 
> **What they compare**:
> - Do attributions highlight sensible features (dog's face vs. background)?
> - Do they satisfy theoretical axioms?
> - How fast do they compute?
> - Do humans agree with the explanations?
> 
> **Edge case**: Sometimes they'll also test across *different* models (ResNet vs. VGG vs. Inception) to check if an attribution method is **robust**—does it work well regardless of architecture? But within a single comparison, it's always the same model.
> 
> The goal: isolate the explanation method as the variable, keep everything else fixed. Otherwise you can't tell if differences come from the explainer or the model! 🔬

---

## How SHAP Benchmarked Attribution Methods

Since evaluating interpretability is ridiculously hard (you're right to call this out!), the SHAP paper took a **multi-pronged approach**:

### 1. **Theoretical Axioms** (The Main Innovation)

Instead of just testing empirically, they defined what a "good" attribution method *should* satisfy:

**Three Key Properties:**

1. **Local Accuracy**: The sum of attributions should equal the model's output:
   $$f(x) - f(\text{baseline}) = \sum_{i=1}^n \phi_i$$

2. **Missingness**: If a feature doesn't appear in the model, its attribution should be zero

3. **Consistency**: If changing a model makes a feature more important, its attribution shouldn't *decrease*

The authors proved SHAP values are the **unique** attribution method satisfying these axioms. This is powerful because it sidesteps empirical benchmarking entirely—it's a mathematical guarantee!

### 2. **Computational Efficiency**

They compared **speed** vs. accuracy:

- **Kernel SHAP** vs. LIME and Shapley sampling
- Measured: How many model evaluations needed to converge to accurate feature importance?
- Result: SHAP converged faster with fewer samples (see their Figure 3)

This isn't about correctness—it's about **sample efficiency**: getting good estimates with minimal computational cost.

### 3. **Human Intuition Studies** (The Bold Part)

Here's where it gets interesting. They ran **user studies on Amazon Mechanical Turk**:

- Showed humans simple models they could understand (like small decision trees)
- Asked: "Which features do you think are important?"
- Compared human judgments to SHAP, LIME, and DeepLIFT

**The assumption**: If an attribution method disagrees with humans on *interpretable* models, it's probably not trustworthy on complex models either.

**Results**: SHAP aligned better with human intuition than competing methods.

### 4. **Class Discrimination on MNIST**

They tested whether attributions help distinguish between classes:

- Trained on MNIST digits
- Checked if attributions for "3" highlight different pixels than "8"
- Compared SHAP vs. DeepLIFT vs. LIME

**Metric**: Can you tell which class a prediction came from by looking at the attribution map alone?

---

## The Fundamental Problem (Why This is ALL So Hard)

The Integrated Gradients paper (1703.01365) nails the core issue:

> "We found that every empirical evaluation technique we could think of could not differentiate between artifacts that stem from **perturbing the data**, a **misbehaving model**, and a **misbehaving attribution method**."

When you evaluate an attribution method empirically, you can't tell:
- Is the method wrong?
- Or is the model actually using weird spurious correlations?
- Or did you mess up the dataset somehow?

**This is why both papers went axiomatic**: define mathematical properties you want, then prove your method satisfies them.

### Why Empirical Evaluation Breaks Down

Let's make this concrete with examples of each failure mode:

**Problem 1: The Data Perturbation Trap**

Say you want to test if an attribution method works. Natural idea: remove the "important" features and see if the prediction changes!

- Attribution says: "This pixel matters most for predicting 'dog'"
- You test it: black out that pixel → prediction changes to 'cat'
- Success, right? 🎉

**Wrong!** Here's what went wrong:

You just created an **out-of-distribution input**. Real images don't have random black squares! The model never saw this during training, so now it's freaking out and predicting garbage. The prediction changed not because that pixel was truly important, but because you broke the data distribution.

*Real example*: In ImageNet, if you black out a "suspicious" region, models often predict random things—not because those pixels were critical, but because black squares are weird artifacts the model interprets as different objects entirely.

**Problem 2: The Misbehaving Model Trap**

Imagine your attribution method highlights a dog's collar as most important for the "dog" prediction. Is the method wrong?

**Maybe not!** The model might genuinely be using the collar as a shortcut. If all training images of dogs had collars and cats didn't, the model learned a spurious correlation. The attribution is *correctly* revealing that the model is *incorrectly* focusing on collars.

Now you're stuck: 
- If attributions highlight the collar → is the attribution method bad, or is it honestly reporting the model's bad behavior?
- If attributions highlight the face → is it correct, or is it just showing what *you expect* while missing the real (collar-based) logic?

You can't tell without knowing ground truth, which you don't have!

**Problem 3: The Multiple Explanations Trap**

Here's a mindbender: sometimes there are *multiple valid explanations* for the same prediction.

- Attribution Method A says: "The dog's ears are most important"  
- Attribution Method B says: "The dog's nose is most important"

**Both could be right!** If you cover the ears, the prediction might still be "dog" (nose is enough). If you cover the nose, still "dog" (ears are enough). They're redundant features.

Which attribution is "correct"? This is philosophically unclear! Different methods might emphasize different sufficient features, and there's no objective way to say one is better.

### Why Axioms Save Us (Sort Of)

Instead of asking "does this match reality?" (which we can't answer), axioms ask:

**"Does this method behave consistently with mathematical properties we care about?"**

For example:
- **Sensitivity (IG)**: If changing a feature changes the output, that feature should get non-zero attribution
  - This sidesteps needing ground truth! We can check this property mathematically.

- **Completeness (SHAP)**: Attribution scores should sum to the prediction's difference from baseline
  - Again, purely mathematical—no need to know what the model "should" care about

**The catch**: Axioms don't guarantee the method is explaining what the model *actually* does—they just guarantee logical consistency. It's the difference between:
- ❌ "This is the true explanation" (unknowable)
- ✅ "This explanation is mathematically coherent" (provable)

Still better than empirical eval where you can't tell what's breaking! But it means we're not really solving interpretability—we're just making rigorous statements about what properties our explanations satisfy.

**Humbling conclusion**: We don't have ground truth for interpretability, so we can't definitively say any method is "correct." We can only say "this method satisfies properties X, Y, Z that seem desirable." It's the best we've got! 🤷

---

## Integrated Gradients' Approach (1703.01365)

Since empirical eval is a nightmare, they defined **two core axioms**:

### **Axiom 1: Sensitivity**
If changing an input feature changes the output, the attribution for that feature should be non-zero.

*Example*: If adding a single pixel changes "dog" → "cat", that pixel better have non-zero attribution!

### **Axiom 2: Implementation Invariance**
Two functionally identical networks (same input-output mapping) should give identical attributions.

*Why this matters*: If you just refactor your code without changing behavior, attributions shouldn't change!

**The kicker**: Most existing methods (gradients, DeepLIFT) **fail** these axioms! 

### Their Solution: Path Integrals

Integrated Gradients computes attributions as:

$$\text{IG}_i(x) = (x_i - x_i') \int_{\alpha=0}^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

Where $x'$ is a "baseline" (e.g., black image). This is basically:
- Walk along a straight path from baseline to input
- Accumulate gradients along the way
- Multiply by the feature difference

This satisfies both axioms provably!

---

## Why Speed Wasn't the Main Focus

You asked about speed—here's the deal:

1. **SHAP** *did* compare speed (Kernel SHAP vs LIME), but only for computational efficiency, not as a quality metric
2. **Integrated Gradients** is actually super fast—just a few gradient calls!
3. Speed is secondary because **correctness** is the hard part

The real battle is: "Does this method give meaningful explanations?" not "Can it run in 10ms?"

---

## The Bigger Picture

Both papers essentially said:

> "We can't trust empirical evaluation alone, so we'll define what 'correct' means mathematically, then design methods that provably satisfy those definitions."

**SHAP**: Game theory → unique solution  
**Integrated Gradients**: Axioms → path methods

This is why these papers are influential: they moved interpretability from "let's try stuff and see" to "let's prove what works."

---

## Key Takeaway

**Attribution** = assigning importance scores to input features

**Benchmarking** = nearly impossible empirically, so:
- Define axioms/properties you want
- Prove your method satisfies them
- Do *some* empirical validation (human studies, speed) as sanity checks

The hard truth: we still don't have perfect ground truth for interpretability. The best we can do is define desirable properties and hope they capture what we actually care about! 🤷‍♂️

---

## Wait, Are These Just Rebranded Game Theory Axioms?

> **Short answer**: YES, mostly! You caught them red-handed. 🎯
> 
> These papers took **Shapley's axioms from cooperative game theory** (1953!) and repackaged them for ML with new names. Let's expose the heist:
> 
> ### The Original Shapley Axioms (1953)
> 
> Shapley defined four properties for "fair" payoff distribution among players:
> 
> 1. **Efficiency**: Total payoff equals the sum of individual contributions  
>    → *ML renamed this*: **"Completeness"** (attributions sum to prediction)
> 
> 2. **Symmetry**: If two players contribute equally, they get equal payoffs  
>    → *ML version*: **"Missingness"** (unused features get zero attribution)
> 
> 3. **Dummy Player**: If a player contributes nothing, they get zero payoff  
>    → *Also becomes*: **"Missingness"** (basically the same idea)
> 
> 4. **Additivity**: If you play two games, your total payoff is the sum of payoffs from each game  
>    → *ML version*: **"Linearity"** (rarely emphasized in these papers)
> 
> **"Sensitivity"** and **"Implementation Invariance"** are NEW additions specific to ML, but they're inspired by similar concepts in game theory about consistency.
> 
> ### Why the Rebrand?
> 
> **Cynical take**: New names make it sound novel! "We propose a method satisfying Efficiency" sounds boring. "We prove Completeness" sounds like a fresh contribution.
> 
> **Charitable take**: The ML context is genuinely different. In game theory, you're distributing a fixed payoff among players. In ML, you're decomposing a prediction into feature contributions. Same math, but different story—so different names help clarify the interpretation.
> 
> ### Can You Test It Using Game Theory Directly?
> 
> **Hell yeah!** Here's how:
> 
> **Setup**: Treat your ML prediction as a cooperative game:
> - **Players** = input features (pixels, words, etc.)
> - **Coalition value** $v(S)$ = model's prediction when only features in set $S$ are present
> - **Grand coalition payoff** = final prediction $f(x)$
> 
> Now compute Shapley values the classical way:
> 
> $\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [v(S \cup \{i\}) - v(S)]$
> 
> If SHAP truly satisfies Shapley's axioms, these values should match SHAP's output!
> 
> ### The Gotcha
> 
> Here's where it gets spicy: **computing $v(S)$ for arbitrary subsets $S$ is ambiguous in ML**.
> 
> In game theory, $v(S)$ is given (e.g., "these 3 players together produce \$100 value"). In ML:
> - How do you evaluate a model with only pixels {1, 5, 27} present?
> - Do you set other pixels to zero? Gray? Blur them? Sample from the training distribution?
> 
> **This is the "baseline" problem**: different choices of baseline (how you represent "absent" features) give different $v(S)$ values, thus different Shapley values!
> 
> SHAP addresses this with different variants:
> - **Kernel SHAP**: treats absence as "expected value over training data"
> - **Deep SHAP**: uses reference distribution
> - **Gradient SHAP**: uses expected gradients
> 
> So yes, it's game theory—but the translation from "absent players" to "absent features" introduces choices that Shapley never had to make!
> 
> ### Testing Strategy
> 
> Want to validate SHAP against pure game theory?
> 
> 1. **Use a simple, discrete game**: 
>    - Binary features (present/absent is unambiguous)
>    - Small number of features (so you can compute all $2^n$ coalition values)
>    - Example: a decision tree with 5 binary inputs
> 
> 2. **Compute classical Shapley values** manually using the formula above
> 
> 3. **Compute SHAP** on the same model and inputs
> 
> 4. **They should match exactly** (if SHAP is doing what it claims)
> 
> **Fun experiment**: Try this on a simple OR function: 
> 
> $$f(x_1, x_2, x_3) = x_1 \vee x_2 \vee x_3$$
> 
> You can hand-compute Shapley values and compare to SHAP!
> 
> ### The Verdict
> 
> You're absolutely right: these are mostly **rebranded game theory axioms** with some ML-specific tweaks. The real contribution isn't inventing new axioms—it's:
> 
> 1. **Recognizing** Shapley values apply to ML interpretability
> 2. **Adapting** them to handle continuous features and the baseline problem
> 3. **Deriving efficient approximations** (Kernel SHAP, Deep SHAP) since exact computation is intractable
> 
> The math isn't new. The application and computational methods are. Classic academic move: take a 70-year-old idea, apply it to a new domain, give it a fresh coat of paint! 🎨
> 
> **Your testing idea is solid**: if you implement the classical game theory version on simple, discrete models, you can verify SHAP isn't lying about its theoretical foundation. And if there are discrepancies, you've found where the ML adaptation diverges from pure game theory—which could be interesting!

---

## How to Sell This to Reviewers (The Pragmatic Guide)

> **The Problem**: Reviewer 2 says "you need quantitative benchmarks against baselines!"
> 
> **Your dilemma**: You literally just spent 10 pages explaining why traditional benchmarks are broken for interpretability! Now what?
> 
> **The SHAP/IG Strategy** (how they actually did it):
> 
> ### 1. **Lead with Theory, Hard**
> 
> Frame your contribution as **solving a fundamental problem with existing work**:
> 
> - "Prior methods lack theoretical justification and can produce inconsistent results"
> - "We prove our method is the *unique* solution satisfying [axioms X, Y, Z]"
> - "Unlike heuristic approaches, we provide mathematical guarantees"
> 
> This shifts the burden: now *other* methods need to justify why they don't satisfy your axioms! You're not just proposing "another method"—you're setting the standard everyone else should meet.
> 
> **Key move**: Make axioms sound inevitable. Don't say "we chose these properties"—say "any reasonable attribution method *must* satisfy these." Make reviewers feel like your axioms are obvious once stated (even if they weren't obvious before).
> 
> ### 2. **Do SOME Empirics, But Frame Them Carefully**
> 
> Here's the trick: don't claim empirical results prove you're correct. Instead, frame them as **sanity checks**:
> 
> - ✅ "Our method aligns with human intuition on interpretable models"  
>   *(not claiming ground truth, just "seems reasonable")*
> 
> - ✅ "Our attributions are more stable under perturbations"  
>   *(consistency is measurable without ground truth)*
> 
> - ✅ "We achieve 10x speedup over naive Shapley sampling"  
>   *(computational efficiency is objectively measurable)*
> 
> - ✅ "When tested on models with known behavior, our method correctly identifies the relevant features"  
>   *(use synthetic/simple cases where you DO have ground truth)*
> 
> **What NOT to say**:
> - ❌ "Our method achieves 95% accuracy" (accuracy at what? You don't have labels!)
> - ❌ "We outperform baselines on ImageNet" (outperform according to whom?)
> 
> ### 3. **Preempt the Reviewer with a "Limitations of Empirical Evaluation" Section**
> 
> Both papers did this brilliantly—they explicitly wrote about why traditional benchmarking fails:
> 
> - "We explored several empirical evaluation strategies but found they confound attribution quality with model behavior and data artifacts"
> - "Without ground truth, quantitative metrics can be misleading"
> 
> By addressing this *yourself*, you look thoughtful rather than defensive. You're saying: "we thought about this deeply and here's why we made these choices."
> 
> **Bonus**: Cite prior failed attempts at empirical evaluation. Show you're not just being lazy—you're learning from others' mistakes.
> 
> ### 4. **The Comparison Table Trick**
> 
> Make a table showing which axioms/properties different methods satisfy:
> 
> | Method | Sensitivity | Completeness | Implementation Invariance |
> |--------|-------------|--------------|---------------------------|
> | Gradients | ❌ | ❌ | ❌ |
> | LIME | ❌ | ❌ | ✓ |
> | DeepLIFT | ✓ | ✓ | ❌ |
> | **Ours (IG/SHAP)** | ✓ | ✓ | ✓ |
> 
> This looks like a quantitative comparison (reviewers love tables!) but it's actually *theoretical*. You're not claiming "we're better" subjectively—you're stating mathematical facts about which properties hold.
> 
> ### 5. **Show Examples Where Other Methods Fail**
> 
> Nothing sells a paper like showing competitors breaking in obvious ways:
> 
> - "When we apply [baseline method] to two functionally identical networks, it produces different attributions" (violates Implementation Invariance)
> - "Method X assigns zero attribution to a feature that, when removed, changes the prediction" (violates Sensitivity)
> 
> These are *objective* failures—you don't need ground truth to show a method contradicts its own logic!
> 
> ### 6. **The "Future Work" Escape Hatch**
> 
> If a reviewer demands something unreasonable:
> 
> - "While large-scale quantitative evaluation remains an open challenge for the field, our theoretical framework provides a foundation for future empirical work"
> - "Developing robust benchmarks is critical future work; we provide initial validation through [human studies/synthetic data]"
> 
> Translation: "you're asking for something nobody can do; we've done what's possible; someone else can solve the impossible part later."
> 
> ### 7. **Appeal to Authority**
> 
> If your method comes from established theory (game theory, functional analysis), lean into it:
> 
> - "Shapley values are the unique solution in cooperative game theory, studied for 70+ years"
> - "Path integrals are a fundamental concept in physics and mathematics"
> 
> Reviewers are less likely to dismiss something with decades of theoretical backing. You're not inventing ad-hoc heuristics—you're applying principled mathematics!
> 
> ---
> 
> ### The Bottom Line
> 
> **What they actually did**:
> - 70% of the paper: Theory, axioms, proofs
> - 20% of the paper: Sanity check empirics (human studies, speed, simple cases)
> - 10% of the paper: Comparisons showing other methods violate axioms
> 
> **The sales pitch**:
> - "We solve a foundational problem (lack of theoretical grounding)"
> - "We prove our method is unique/optimal under reasonable assumptions"
> - "We validate it doesn't produce obviously wrong results"
> - "We show competitors violate basic desirable properties"
> 
> **How to handle pushback**:
> - "We agree robust evaluation is critical—that's why we identify the core problem with existing approaches and propose axioms as a solution"
> - "Traditional quantitative metrics are misleading without ground truth; our theoretical guarantees are stronger"
> - Point to your "limitations of empirical eval" section: "we address this in §X"
> 
> The meta-strategy: **make the lack of ground truth someone else's problem, not yours**. You provided a rigorous theoretical solution; it's the field's job to figure out how to empirically validate interpretability (good luck with that! 😅).

---

## Terminology Confusion: Attribution vs. Feature Contribution

> **TL;DR**: ML people say "attribution," everyone else says "feature importance" or "contribution."
> 
> **Why the mess?** The SHAP paper borrowed Shapley values from game theory (1950s economics!), where they measure how much each player "contributes" to a coalition's payoff. In that world, you'd say "player i's contribution" or "marginal contribution."
> 
> But ML researchers rebranded it as **"attribution"**—basically asking "which features do we *attribute* the prediction to?" It's the same math, just different framing:
> 
> - **Game theory**: "How much did player *i* contribute to the team's winnings?"
> - **ML**: "How much do we attribute to feature *i* for this prediction?"
> 
> **What's actually used?**
> - Academic ML papers: **"attribution"** dominates (thanks to influential papers like these)
> - Explainable AI practitioners: **"feature importance"** is common
> - Classical statistics/econometrics: **"marginal effects"** or **"contribution"**
> - Random ML engineers: all of the above interchangeably 😅
> 
> The terminology shift happened because ML focuses on *explaining individual predictions* ("why did the model call *this* image a cat?"), while game theory focuses on *fair allocation* ("how do we split the prize money?"). Same math (phi values), different story you're telling.
> 
> **Hot take**: "Attribution" stuck because it sounds fancier and makes papers easier to cite. "Feature contribution" is probably clearer, but we're stuck with "attribution" now because that's what the influential papers called it. Such is life in rapidly evolving fields! 🤷