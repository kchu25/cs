@def title = "Group Equivariance in Neural Networks"
@def published = "1 November 2025"
@def tags = ["machine-learning"]

# Group Equivariance in Neural Networks

## The Core Idea (No Jargon)

Think of it this way: you have some **transformation** you can do to your data (like sliding an image left, or rotating it, or shuffling a list). 

**Equivariance** just means: if I apply that transformation to my input, I can predict exactly what happens to my output. The output changes in the "same way" as the input did.

> **Why is it called "Equivariance"?**
> 
> The name comes from Latin: *equi-* (equal) + *variance* (change/variation). It literally means "equal change" or "varying equally."
> 
> When you transform the input, the output **varies** (changes). Equivariance means both the input and output **vary equally** - they change in the same corresponding way. The transformation "propagates through" the function in a predictable manner.
> 
> Compare this to **invariance** (*in-* = not), which means "not varying" - the output stays the same regardless of how you transform the input.

---

## Why "Groups"? What Are They Really?

A **group** is just a fancy name for "a collection of transformations that play nice together." That's it!

What does "play nice" mean? Four simple things:

1. **You can combine transformations**: Slide left 5 pixels, then slide left 3 pixels = slide left 8 pixels. Still in your collection.

2. **Order of combining doesn't matter in weird ways**: (slide left then slide up) then rotate = slide left then (slide up then rotate). It's associative.

3. **There's a "do nothing" transformation**: Don't move the image at all. That's your identity.

4. **Every transformation can be undone**: Slide left 5 pixels? There's a "slide right 5 pixels" that undoes it.

That's literally all a group is - transformations with these four properties.

---

## Concrete Examples of Groups

### Translation Group (What CNNs Use)

**The transformations**: All possible ways to slide/shift an image
- Slide left 3 pixels
- Slide right 10 pixels and down 5 pixels  
- Don't move it at all (identity)
- etc.

**Combining them**: Slide right 5 then slide left 3 = slide right 2 overall

**Why this is a group**: 
- ✓ Combining slides gives another slide
- ✓ You can undo any slide (slide the opposite direction)
- ✓ There's a "don't move" transformation
- ✓ Combining slides is associative

### Rotation Group

**The transformations**: All possible ways to rotate an image
- Rotate 30 degrees clockwise
- Rotate 90 degrees counterclockwise
- Don't rotate (identity)
- etc.

**Combining them**: Rotate 90° clockwise then 45° clockwise = rotate 135° clockwise

### Permutation Group $S_n$ (What Graph Networks Use)

**The transformations**: All ways to shuffle/reorder a list
- Swap item 1 and item 3
- Reverse the entire list
- Leave it alone (identity)
- etc.

**Combining them**: First swap positions 1↔2, then swap 2↔3

> **Do Transformers use group equivariance?**
> 
> **Yes!** Transformers (without positional encodings) are **permutation equivariant** - they use the permutation group $S_n$.
> 
> **What this means**: If you shuffle the input tokens, the output features shuffle in exactly the same way. Self-attention operations are:
> - `Attention(Q, K, V)` treats the input as a **set** of tokens
> - The operation doesn't depend on the order of tokens
> - Shuffling inputs → shuffled outputs in the same way
> 
> **Why add positional encodings then?** 
> - Pure permutation equivariance means the model is **blind to order**
> - For language, order matters! "Dog bites man" ≠ "Man bites dog"
> - Positional encodings **break** the permutation symmetry intentionally
> - This trades the inductive bias for expressiveness when order matters
> 
> **So Transformers vs CNNs:**
> - **CNNs**: Strong spatial inductive bias (translation equivariance)
> - **Transformers**: Weak/no spatial bias (permutation equivariance, broken by positional encoding)
> - **Vision Transformers**: Sacrifice spatial bias for flexibility, need more data but can be more expressive

---

## The "Action" Concept Made Simple

An **action** is just: "how does this group of transformations actually change my data?"

**For images and translations**:
- Transformation: "slide right 5 pixels"
- Action: Take every pixel and move it 5 spots to the right
- Mathematically: $T_5(\text{image})[x, y] = \text{image}[x-5, y]$

**For lists and permutations**:
- Transformation: "swap positions 1 and 3"  
- Action: Take the list and swap those elements
- Mathematically: $\pi(\text{list})[1] = \text{list}[3]$, $\pi(\text{list})[3] = \text{list}[1]$

---

## What Equivariance Really Means (Visually)

Let me trace through an example:

```
INPUT IMAGE:        🐱 at position (10, 10)
                    |
                    | Pass through CNN
                    ↓
OUTPUT:             "Cat detected" at position (10, 10)

Now slide input right 5 pixels:

INPUT IMAGE:        🐱 at position (15, 10)  
                    |
                    | Pass through SAME CNN
                    ↓
OUTPUT:             "Cat detected" at position (15, 10)
```

**Equivariance** = the "cat detected" signal **moved exactly like the cat moved**.

Compare to **invariance**:
```
INPUT:  🐱 anywhere  →  OUTPUT: "Cat: YES"
```
You lose the position info entirely.

---

## What Does the Group Formalization Give You?

Here's the magic: once you say "my transformations form a group," you get a **systematic way to build neural networks** that respect those transformations.

### Example: Building a CNN

**Problem**: I want a network where if I slide the input, the output slides the same way.

**Old way** (before group theory): 
- Try a bunch of architectures
- Test if they have this property
- Hope for the best

**Group theory way**:
1. Say "I care about the translation group"
2. The math **tells you exactly** what operations preserve this: **convolution**!
3. The proof shows convolution is basically the **only** linear operation that works

The group formalization doesn't just describe what CNNs do - it **derives** why convolution is the right operation, from first principles.

---

## Why Groups Make This Powerful

Once you know your transformations form a group, you can:

1. **Build architectures systematically**: Group theory tells you what operations to use (convolution for translations, special filters for rotations, etc.)

2. **Guarantee the property**: If each layer is equivariant and you stack them, the whole network is equivariant (because equivariant functions compose)

3. **Quantify the benefit**: If your group has size $|G|$, you need roughly $|G|$ times less data, because seeing one example is like seeing $|G|$ examples (one for each transformation)

4. **Generalize beyond images**: Want permutation equivariance for graphs? The same group theory framework tells you how to build it (that's how Graph Neural Networks were derived!)

---

## The Bottom Line

**Without group theory**: "CNNs work well on images, we figured out convolution is good, not sure why"

**With group theory**: "Images have translation symmetry (they form a group under sliding). Group theory proves convolution is the correct operation for translation equivariance. We can quantify the sample efficiency gains and extend this principle to other symmetries."

It's not just abstract math - it's a **design principle** that tells you how to build networks for different kinds of data based on what symmetries that data has.

---

# Mathematical Foundations

## Groups: Formal Definition

A **group** $G$ is a set with a binary operation $\cdot$ satisfying:

- **Closure**: $g_1 \cdot g_2 \in G$ for all $g_1, g_2 \in G$
- **Associativity**: $(g_1 \cdot g_2) \cdot g_3 = g_1 \cdot (g_2 \cdot g_3)$
- **Identity**: $\exists e \in G$ such that $g \cdot e = e \cdot g = g$ for all $g \in G$
- **Inverse**: For all $g \in G$, $\exists g^{-1} \in G$ such that $g \cdot g^{-1} = g^{-1} \cdot g = e$

## Group Actions

A group $G$ **acts** on a set $X$ if there exists a mapping:

$$\rho: G \times X \rightarrow X$$

written as $g \cdot x$, satisfying:

1. **Identity acts trivially**: $e \cdot x = x$ for all $x \in X$
2. **Compatibility**: $g_1 \cdot (g_2 \cdot x) = (g_1 \cdot g_2) \cdot x$ for all $g_1, g_2 \in G, x \in X$

**Example: Translation Group**

The translation group $G = (\mathbb{R}^2, +)$ acts on images $X = \mathbb{R}^{H \times W \times C}$ by:

$$T_v(x)[i,j] = x[i-v_1, j-v_2]$$

where $v = (v_1, v_2)$ is a translation vector.

---

## Equivariance: Formal Definition

A function $f: X \rightarrow Y$ is **equivariant** with respect to group actions $\rho_X$ on $X$ and $\rho_Y$ on $Y$ if:

$$f(\rho_X(g, x)) = \rho_Y(g, f(x)) \quad \forall g \in G, \forall x \in X$$

Or more compactly:

$$f \circ \rho_X(g) = \rho_Y(g) \circ f$$

This says: **"transforming then processing" equals "processing then transforming"**.

> **Intuitions for the math:**
>
> **Reading the equation** $f(\rho_X(g, x)) = \rho_Y(g, f(x))$:
> 
> **Left side**: $f(\rho_X(g, x))$
> - First, apply transformation $g$ to input $x$ (using action $\rho_X$)
> - Then, apply function $f$ to the transformed input
> - Path: $x \xrightarrow{\text{transform}} \rho_X(g,x) \xrightarrow{f} f(\rho_X(g,x))$
>
> **Right side**: $\rho_Y(g, f(x))$
> - First, apply function $f$ to the original input $x$
> - Then, apply the same transformation $g$ to the output (using action $\rho_Y$)
> - Path: $x \xrightarrow{f} f(x) \xrightarrow{\text{transform}} \rho_Y(g, f(x))$
>
> **Equivariance means**: Both paths give the same result! The order doesn't matter.
>
> **Concrete example with images:**
> - $g$ = "shift right 5 pixels"
> - $f$ = "detect edges" (convolution)
> - $x$ = original image
>
> Left side: Shift image right 5 pixels, then detect edges
> 
> Right side: Detect edges first, then shift the edge map right 5 pixels
>
> Equivariance says: **you get the same edge map either way!**
>
> **Why the compact form** $f \circ \rho_X(g) = \rho_Y(g) \circ f$:
> - This is function composition notation
> - $\circ$ means "compose" (do one operation after another)
> - Reading right to left: $(f \circ \rho_X(g))(x) = f(\rho_X(g, x))$
> - It emphasizes that **$f$ commutes with the group action**
> - The transformation $g$ "passes through" $f$ unchanged

> **Important clarification: Is the operation itself a group?**
>
> **No!** This is a common point of confusion. Let's be clear:
>
> - **The GROUP** ($G$): The transformations of your data (e.g., all possible translations/shifts)
> - **The OPERATION** ($f$): The function that respects those transformations (e.g., convolution)
> - **These are different things!**
>
> **Equivariance is a PROPERTY of a function**, not a requirement that the function forms a group.
>
> **Example:**
> - **Group $G$**: All translations $\{T_v : v \in \mathbb{R}^2\}$ - this IS a group ✓
> - **Operation $f$**: Convolution $f(x) = w * x$ - this is NOT a group ✗
> - **But $f$ IS equivariant to $G$** ✓
>
> **Why convolution isn't a group:**
> - Most convolutions are **not invertible** (they lose information)
> - You can't "undo" a convolution in general
> - Lack of inverses means no group
>
> **Why convolution is equivariant:**
> - It **commutes with translations**: $f(T_v(x)) = T_v(f(x))$
> - Shifting then convolving = convolving then shifting
> - This is all we need for equivariance!
>
> **Analogy**: Think of rotations of a sphere (a group) and projection onto a plane (equivariant operation). The projection isn't a rotation itself, but it respects rotations: rotate-then-project = project-then-rotate.

### Commutative Diagram

```
    X  ----f---->  Y
    |              |
 ρ_X(g)         ρ_Y(g)
    |              |
    ↓              ↓
    X  ----f---->  Y
```

The diagram commutes: both paths from top-left to bottom-right give the same result.

---

## CNN Example: Translation Equivariance

For 2D convolution with kernel $w$:

$$(w * x)[i,j] = \sum_k \sum_l w[k,l] \cdot x[i-k, j-l]$$

**Claim**: Convolution is translation-equivariant.

**Proof:**

Let $T_v$ denote translation by vector $v = (v_1, v_2)$.

$$\begin{align}
(w * T_v(x))[i,j] &= \sum_{k,l} w[k,l] \cdot (T_v(x))[i-k, j-l] \\
&= \sum_{k,l} w[k,l] \cdot x[i-k-v_1, j-l-v_2] \quad \text{[by def of } T_v\text{]} \\
&= (w * x)[i-v_1, j-v_2] \quad \text{[shift indices]} \\
&= T_v(w * x)[i,j] \quad \text{[by def of } T_v\text{]}
\end{align}$$

Therefore: $w * T_v(x) = T_v(w * x)$ ✓

---

## Invariance vs Equivariance

**Invariance** means the output doesn't change under transformations:

$$f(\rho_X(g, x)) = f(x) \quad \forall g \in G$$

Invariance is a special case of equivariance where $\rho_Y$ is the **trivial action**: $\rho_Y(g, y) = y$ for all $g, y$.

**Example**: Global average pooling creates translation invariance from translation equivariance.

---

## Composability of Equivariant Functions

**Theorem**: If $f$ and $g$ are both $G$-equivariant, then $f \circ g$ is $G$-equivariant.

**Proof**: 
$$(f \circ g)(\rho_X(g, x)) = f(g(\rho_X(g, x))) = f(\rho_Y(g, g(x))) = \rho_Z(g, f(g(x))) = \rho_Z(g, (f \circ g)(x))$$

This is why stacking convolutional layers maintains translation equivariance throughout the network.

---

## Sample Efficiency Through Symmetry

If your function is equivariant and your data has that symmetry:

$$\text{Training examples needed} \approx \frac{\text{Examples without symmetry}}{|G|}$$

where $|G|$ is the group size (infinite for continuous groups like translations and rotations!).

> **What does "your data has that symmetry" mean?**
> 
> This means your data is **unchanged (or predictably changed) under certain transformations**. More precisely, it means there exist **symmetry-preserving transformations** where the meaningful content or label doesn't change.
> 
> **Examples of symmetric transformations:**
> - **Translation/shifting for images**: A cat shifted 40 pixels right is still the same cat. Translation is a symmetry of image classification.
> - **Rotation**: A rotated photo of a dog is still a dog. Rotation is a symmetry for many vision tasks.
> - **Permutation for sets**: A set {apple, banana, orange} is the same set regardless of order. Permutation is a symmetry of set operations.
> 
> **Counter-example (transformation breaks symmetry):**
> - **Translation for chess**: Moving all pieces 2 squares right creates a completely different position. Translation is NOT a symmetry of chess - position matters absolutely. "Pawn on e4" ≠ "pawn on g4" strategically.
> 
> **Better terminology**: We could say "the data has translation symmetry" or "translation is a symmetry-preserving transformation for this data." The symmetry isn't in the data itself, but in how certain transformations preserve the data's meaningful properties.
> 
> When your task has these symmetry-preserving transformations, equivariant architectures can exploit them for huge efficiency gains!

**Intuition**: Learning a cat detector with translation equivariance means that one image of a cat teaches you about cats at *all* positions simultaneously.

---

## The Fundamental Trade-off

Strong inductive bias = strong assumptions:

- **When the bias matches the data**: Massive efficiency gains, better generalization with less data
- **When the bias doesn't match**: Fundamental limitation, reduced expressiveness

**Example**: CNNs assume translation symmetry.
- **Natural images**: Excellent! A cat is a cat regardless of position.
- **Chess boards**: Terrible! Position matters absolutely (a1 vs h8 have different strategic meanings).

This is one reason Vision Transformers work—they sacrifice the global translation bias for more flexibility, which helps when you have enough data that you don't need that strong inductive bias anymore.