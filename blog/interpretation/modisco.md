
@def title = "Understanding Seqlets in TF-MoDISco"
@def published = "29 December 2025"
@def tags = ["interpretation"]

# Understanding Seqlets in TF-MoDISco

## The Big Picture: What is TF-MoDISco?

TF-MoDISco (Transcription Factor Motif Discovery from Importance Scores) is a computational tool that takes per-nucleotide importance scores (from methods like DeepLIFT, SHAP, or Integrated Gradients) and discovers recurring patterns—motifs—that the model has learned to recognize.

The key insight: instead of looking at raw sequence patterns, TF-MoDISco clusters based on *contribution patterns*. A traditional motif finder might say "GATAAG appears often." TF-MoDISco says "there's a pattern where positions 1, 4, and 5 consistently have high positive contributions."

## What's a seqlet?

A seqlet is basically a short stretch of sequence (typically 10-30 base pairs) that TF-MoDISco identifies as potentially important. Think of it as a "snippet of interest" extracted from your larger input sequences.

More precisely, a seqlet is defined by:
- **Start and end coordinates** within a parent sequence
- **The underlying nucleotide sequence** (e.g., "GATAAGCC")
- **Per-nucleotide contribution scores** from your interpretation method
- **A strand orientation** (forward or reverse complement)

## The TF-MoDISco Pipeline in Detail

Let's trace through exactly what happens:

### Step 1: Contribution Score Computation

Before TF-MoDISco even runs, you need per-nucleotide contribution scores. These come from interpreting a trained model:

```
Input sequence:  A  T  G  C  A  T  A  G  ...
Contribution:   0.1 0.0 0.8 0.9 0.1 0.0 0.7 0.6 ...
```

Common methods:
- **DeepLIFT**: Compares activations to a reference
- **Integrated Gradients**: Integrates gradients along path from baseline
- **SHAP**: Shapley values (DeepSHAP is popular)

Each method has different theoretical properties, but TF-MoDISco treats them all the same way.

### Step 2: Seqlet Identification (The Heuristic Part)

This is where the thresholding happens. TF-MoDISco needs to find "regions of interest":

**The sliding window approach:**
1. Slide a window (e.g., 21bp) across each sequence
2. Compute the sum of absolute contribution scores within the window
3. If the sum exceeds a threshold, mark this as a candidate seqlet

**The threshold determination:**
- By default, TF-MoDISco uses an adaptive threshold based on the distribution of window scores
- It typically looks for the "elbow" in the distribution, or uses a percentile (e.g., top 5%)
- The parameter `target_seqlet_fdr` controls how aggressive the detection is

**Concrete example:**
```
Position:        1    2    3    4    5    6    7    8    9   10
Sequence:        A    T    G    C    A    T    A    G    C    T
Contribution:   0.1  0.0  0.8  0.9  0.7  0.6  0.5  0.3  0.1  0.0

Window [1-5] sum: |0.1| + |0.0| + |0.8| + |0.9| + |0.7| = 2.5  ✓ Above threshold
Window [6-10] sum: |0.6| + |0.5| + |0.3| + |0.1| + |0.0| = 1.5  ✗ Below threshold
```

### Step 3: Seqlet Filtering and Expansion

Once candidates are identified:
- **Overlapping seqlets** are merged or the stronger one is kept
- **Boundaries are refined** by extending until contribution scores drop
- **Too-short seqlets** may be filtered out

### Step 4: The Clustering Magic (This is the interesting part!)

Now TF-MoDISco clusters seqlets that have similar contribution patterns. This is NOT just sequence similarity—two seqlets with the same sequence can be in different clusters if their contribution patterns differ!

**The similarity metric:**
TF-MoDISco uses a correlation-based similarity between seqlet contribution profiles:

$$\text{sim}(s_1, s_2) = \max_{\text{offset}} \text{corr}(\text{contrib}_{s_1}, \text{contrib}_{s_2}[\text{offset}])$$

It also considers the reverse complement, so "GATAAG" and "CTTATC" can cluster together.

**The clustering algorithm:**
- Uses a greedy clustering approach (not k-means or hierarchical)
- Iteratively assigns seqlets to the cluster with highest similarity
- Creates new clusters when similarity to existing clusters is too low

### Step 5: Building the CWM (Contribution Weight Matrix)

For each cluster, TF-MoDISco aligns all seqlets and averages their contribution scores:

```
Seqlet 1 contrib:  0.1  0.8  0.9  0.7  0.1
Seqlet 2 contrib:  0.2  0.7  0.8  0.6  0.2
Seqlet 3 contrib:  0.0  0.9  0.9  0.8  0.0
                   ─────────────────────────
CWM (average):     0.1  0.8  0.87 0.7  0.1
```

The CWM tells you: "At position 2 and 3, contributions are consistently high across instances."

You also get a **PWM (Position Weight Matrix)** by looking at the nucleotide frequencies:
```
Position:    1    2    3    4    5
A:          0.8  0.0  0.0  1.0  0.0
T:          0.2  0.0  0.0  0.0  1.0
G:          0.0  1.0  0.0  0.0  0.0
C:          0.0  0.0  1.0  0.0  0.0
```

## How are seqlets quantified?

You're absolutely right that there's a heuristic involved! Here's how it works:

### The threshold approach

Since TF-MoDISco starts with per-nucleotide contribution scores (from SHAP, DeepLIFT, etc.), it needs a way to decide which regions are "important enough" to be seqlets. The typical approach:

1. **Set a contribution threshold** - Only nucleotides with contribution scores above a certain threshold are considered
2. **Find contiguous regions** - Connected stretches of high-contribution nucleotides become candidate seqlets
3. **Apply filtering** - Very short regions (like 1-2 bp) might get filtered out as noise

### Common heuristics

The exact threshold can be:
- A fixed percentile (e.g., top 5% of contribution scores)
- A multiple of the standard deviation above the mean
- Set empirically based on your data

### Key parameters that affect seqlet detection

| Parameter | What it controls | Trade-off |
|-----------|-----------------|-----------|
| `sliding_window_size` | Width of detection window | Larger = smoother, may miss short motifs |
| `flank_size` | How much to extend seqlets | Larger = more context, more noise |
| `target_seqlet_fdr` | How many seqlets to find | Lower = more seqlets, more noise |
| `min_seqlets_per_task` | Minimum seqlets needed | Higher = more robust, may miss rare motifs |

## Why this matters

This thresholding step is crucial because it determines what patterns TF-MoDISco will try to cluster and discover. Set it too high and you miss subtle motifs; too low and you get swamped with noise.

The beauty is that once you have seqlets defined, TF-MoDISco can cluster them based on their contribution patterns to discover recurring motifs - no additional quantification needed at that point!

---

> **Does the stretch need uniformly high contributions?**
>
> Nope! The seqlet just needs *some* nucleotides above threshold within a window. Think of it like finding "peaks" in the contribution landscape - you're looking for regions with elevated signal, not necessarily flat plateaus. Typically, a sliding window approach identifies regions where enough nucleotides (or the aggregate score) exceeds the threshold.
>
> **Is this heuristic-y?**
>
> Oh absolutely, 100% yes. You're making arbitrary choices about:
> - What threshold to use
> - How big the sliding window should be
> - How many high-contribution bases constitute a "region"
> - Whether to use max, mean, or sum aggregation
>
> Different settings can give you different seqlets, which means different discovered motifs. It's practical and works well, but definitely not theoretically rigorous.
>
> **The aggregation opacity issue**
>
> You've hit on something important here. Yes, they aggregate! To compare seqlets, TF-MoDISco needs to summarize each one somehow - often taking the sum or mean of contributions across nucleotides in the seqlet. This adds another layer of abstraction: you're now clustering aggregated summaries of already-thresholded data. It works empirically because motifs have characteristic patterns, but yeah... it's opaque. You're multiple steps removed from your original contribution scores.
>
> **The global interpretation problem**
>
> Here's the kicker: at the end, you get a discovered motif pattern (like "GATAAG"), but what's the *contribution* of that pattern globally across all your sequences? TF-MoDISco doesn't really give you a clean answer to this.
>
> You know the pattern exists and appears in multiple seqlets, but:
> - You don't get a single "importance score" for the motif across your dataset
> - The contribution scores were local to individual sequences
> - The clustering/aggregation process has mixed contributions from different sequences and contexts
>
> So you end up with interpretable *patterns* (what motifs exist) but not necessarily interpretable *global contributions* (how much does this motif matter overall). It's more of a pattern discovery tool than a global feature importance tool. You'd need to do additional analysis - like testing the motif's effect across your dataset - to quantify its global importance.

---

## A Deeper Look: What Does the CWM Actually Represent?

Let's be precise about what a CWM (Contribution Weight Matrix) is and isn't.

### Mathematically

For a motif cluster with $N$ seqlets, each of length $L$, the CWM at position $j$ for nucleotide $b$ is:

$$\text{CWM}_{j,b} = \frac{1}{N_b} \sum_{i: s_i[j] = b} c_i[j]$$

where:
- $s_i[j]$ is the nucleotide at position $j$ of seqlet $i$
- $c_i[j]$ is the contribution score at position $j$ of seqlet $i$
- $N_b$ is the count of seqlets with nucleotide $b$ at position $j$

**The key issue**: You're averaging contributions from different sequence contexts. A "G" at position 3 in one seqlet might have contribution 0.9 because of its neighbors, while another "G" at position 3 has contribution 0.5 due to different context.

### What CWM captures vs. what it obscures

| CWM Captures | CWM Obscures |
|--------------|--------------|
| Average positional importance | Variance in importance |
| Nucleotide preferences | Context-dependent effects |
| Core motif structure | Flanking sequence effects |
| Consistent patterns | Rare but important variants |

### An analogy

Imagine you're studying how people cross intersections:
- **Seqlets** = individual crossing observations
- **Clustering** = grouping crossings by intersection type (4-way, T-junction, roundabout)
- **CWM** = "average walking path" for each intersection type

The CWM tells you the typical path, but obscures individual variations (some people jaywalk, some wait for lights, some run). If those variations matter, you've lost that information.

## Comparison to Other Motif Discovery Methods

| Method | Input | What it finds | Strengths | Weaknesses |
|--------|-------|---------------|-----------|------------|
| **TF-MoDISco** | Contribution scores | Patterns model uses | Model-specific, captures importance | Heuristic clustering, no global score |
| **MEME** | Raw sequences | Overrepresented patterns | Statistically principled | May miss model-relevant patterns |
| **HOMER** | Foreground/background | Differentially enriched | Fast, handles background | Assumes differential enrichment |
| **DeepBind** | Sequences + labels | Predictive patterns | End-to-end learning | Black box, needs training |

**The key distinction**: Traditional methods ask "what patterns exist in the data?" TF-MoDISco asks "what patterns does the model care about?" These can be different!

---

## Wait, so why is this used in Nature papers?

**My answer above is partially correct, but let me clarify**: TF-MoDISco *does* give you something interpretable at the global level, just not in the traditional sense you might expect.

What you get:
- **Contribution Weight Matrix (CWM)**: This is the averaged contribution scores across all seqlets in a cluster. It tells you which positions in the motif are consistently important.
- **Per-instance contributions**: You can see how much each specific motif instance contributed in its local context.
- **Pattern frequency**: You know how often the pattern appears across your sequences.
- **Metacluster organization**: Related patterns are grouped together (e.g., all GATA-like motifs)

What you DON'T get directly:
- A single number like "this motif explains 15% of model predictions globally"
- Easy comparison between motifs (which is more important overall?)
- Causal importance (does the model *need* this motif, or just *use* it?)

**Why it's still valuable and widely used:**

1. **It works empirically** - The patterns it discovers are biologically meaningful and match known TF binding sites
2. **Context matters** - The CWM captures position-specific contributions, which is more nuanced than a PWM
3. **Better than alternatives** - Traditional motif discovery (MEME, HOMER) often miss patterns that deep learning models actually use
4. **Hypothesis generation** - You discover what patterns the model learned, then validate them experimentally
5. **Visual interpretability** - CWM logos are intuitive and publication-ready

---

## But wait... aren't CWMs theoretically messy?

**You're absolutely right to push back on this.** CWMs have some real theoretical issues:

**The averaging problem:**
- You're averaging contribution scores across seqlets from different sequences
- These contributions came from different contexts, different positions, different predictions
- What does the "average contribution" even mean when the underlying examples are heterogeneous?

**The alignment problem:**
- Seqlets get aligned before averaging, but this alignment is based on similarity, not on any theoretical principle
- You're forcing different instances into the same coordinate system
- Insertions/deletions are handled by extension, not by proper alignment

**The interpretation problem:**
- Is a CWM position with high average contribution "important" because:
  - That position matters a lot in a few instances?
  - That position matters moderately across many instances?
  - The contribution is just less noisy there?

**The additivity assumption:**
- Averaging assumes contributions are additive/decomposable
- But neural networks have complex interactions—position 2 might only matter *because* of what's at position 5
- These interactions are lost in the CWM

**So why does everyone use it anyway?**

Honestly? Because it's *pragmatically useful* despite being *theoretically sketchy*. The patterns look right, they match known biology, and they generate testable hypotheses. 

It's a bit like p-values in statistics - lots of theoretical problems, widely misinterpreted, but still used everywhere because they're practical. The computational biology field has essentially decided that discovering biologically meaningful patterns is worth the theoretical hand-waving.

The dirty secret: much of neural network interpretability is like this. We're trying to understand complex, non-linear systems with tools that make convenient assumptions. TF-MoDISco is just more honest about the heuristics than most methods.

---

## Practical Recommendations

If you're going to use TF-MoDISco, here are some tips:

### 1. Validate your discovered motifs

Don't just trust the CWM. Check that the pattern makes biological sense:
- Compare to known motif databases (JASPAR, HOCOMOCO)
- Look at the seqlet instances—do they actually look similar?
- Check if the motif appears in biologically relevant regions (promoters, enhancers, etc.)

### 2. Be aware of parameter sensitivity

Run TF-MoDISco with different parameters and see if the same motifs emerge:
- Try different `target_seqlet_fdr` values
- Vary the `sliding_window_size`
- If a motif disappears with small parameter changes, be skeptical

### 3. Look beyond the CWM

The CWM is a summary. Dig deeper:
- Examine the seqlet alignment—are they truly similar?
- Look at the contribution score distribution per position
- Check for bimodal distributions (might indicate sub-clusters)

### 4. Compute supplementary statistics

TF-MoDISco gives you seqlets. Use them to compute:
- **Total contribution**: Sum of contributions across all instances of the motif
- **Coverage**: What fraction of your sequences contain this motif?
- **Effect size**: Mean prediction with motif vs. without

### 5. Consider alternatives for specific questions

| If you want to know... | TF-MoDISco is... | Consider instead... |
|------------------------|------------------|---------------------|
| What patterns does the model use? | ✅ Good | - |
| How important is pattern X globally? | ⚠️ Indirect | In-silico mutagenesis |
| Does pattern X cause high predictions? | ❌ No | Synthetic sequence tests |
| What's the minimal motif? | ⚠️ Approximate | Motif truncation experiments |

---

## The Bottom Line

TF-MoDISco is a powerful tool for discovering *what patterns your model has learned*. It's not perfect—the seqlet detection is heuristic, the clustering is approximate, and the CWM is a lossy summary. But it provides interpretable, biologically meaningful results that have driven real discoveries.

Use it as a starting point for hypothesis generation, not as the final word on model interpretation. Validate what you find, be aware of the assumptions, and remember that "interpretability" in deep learning is still more art than science.

**Key takeaways:**
1. Seqlets are short sequences with high contribution scores, identified heuristically
2. Clustering is based on contribution *patterns*, not just sequence similarity
3. CWMs are averaged contributions—useful but theoretically messy
4. Global importance requires additional analysis beyond TF-MoDISco
5. Despite limitations, it works well in practice for motif discovery