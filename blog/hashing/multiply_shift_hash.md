@def title = "Multiply-Shift Hashing: Fast and Elegant"
@def published = "6 October 2025"
@def tags = ["hashing"]


# Multiply-Shift Hashing: Fast and Elegant

## What is Multiply-Shift Hashing?

Multiply-shift hashing is a beautifully simple and *blazingly fast* hash function construction that's widely used in practice. Unlike the prime-based polynomial method, it works with **powers of 2**, making it perfect for modern computers that love bit operations.

**Historical Note:** This method was rigorously analyzed and popularized by **Martin Dietzfelbinger and colleagues** in their seminal work on universal hashing in the 1990s. Dietzfelbinger et al. proved the theoretical properties and showed how multiply-shift could achieve near-universal performance with remarkable efficiency.

### The Basic Idea

Say you want to hash $w$-bit keys into a table of size $M = 2^b$ (so you need $b$-bit hash values).

**The construction:**
1. Pick a random **odd** $w$-bit integer $a$
2. For any key $x$, compute:
$$h_a(x) = (a \cdot x) \gg (w - b)$$

That's it! Multiply $x$ by $a$, then take the top $b$ bits by right-shifting.

### A Concrete Example

Let's say we're hashing 32-bit keys into a table of size $M = 256 = 2^8$.

- We need $b = 8$ bits for our hash output
- Pick a random odd 32-bit number, say $a = 2654435769$ (this is actually a famous constant!)
- To hash $x = 12345$:
  - Compute $a \cdot x = 2654435769 \times 12345 = 32768347242905$ (in 64-bit arithmetic, we'd take this mod $2^{32}$)
  - Shift right by $32 - 8 = 24$ bits
  - Get the top 8 bits as your hash value

## Why Does This Work?

### The Intuition

When you multiply by a random odd number $a$, you're essentially "scrambling" the bits of $x$ in a pseudorandom way. The multiplication mixes the bits together, and by taking the **top bits** (not bottom!), you're getting the most "mixed" portion of the result.

**Why odd?** Because odd numbers are coprime to $2^w$, which ensures the multiplication permutes the space nicely. If $a$ were even, you'd lose information in the lower bits.

**Why top bits?** The lower bits of $a \cdot x$ depend mainly on the lower bits of $x$ and $a$. The upper bits get contributions from all positions due to carry propagation - they're the most "mixed" and randomized.

## Is It Universal?

Here's where it gets interesting - multiply-shift hashing is **not quite universal** in the strict sense we defined earlier! 

For strict universality, you'd need: $\Pr[h(x) = h(y)] \leq 1/M$ for all $x \neq y$.

Multiply-shift gives you something slightly weaker: it's **almost universal** or **approximately universal**. The collision probability is close to $1/M$ but not guaranteed to be exactly at most $1/M$ for all pairs.

**BUT** - and this is crucial - **it works amazingly well in practice!** The theoretical gap between "universal" and "almost universal" rarely matters for real-world applications.

## Why People Love Multiply-Shift

**1. Speed**
- Just one multiplication and one shift
- No expensive modulo operations
- Works perfectly with power-of-2 table sizes
- CPUs are optimized for these operations

**2. Simplicity**
- Minimal code: literally `(a * x) >> shift`
- Easy to implement correctly
- No need to find prime numbers

**3. Hardware-Friendly**
- Modern CPUs have fast integer multiplication
- Bit shifts are essentially free
- Cache-friendly (no division, no branching)

**4. Good Enough Guarantees**
- Collision probability is very close to $1/M$ in practice
- Works great for hash tables, bloom filters, etc.
- The theoretical distinction from "true" universality rarely matters

## The Famous Constant: Knuth's Magic Number

You'll often see multiply-shift hashing with $a = 2654435769$ for 32-bit keys. This isn't random - it's:

$$a = \lfloor 2^{32} / \phi \rfloor$$

where $\phi = \frac{1 + \sqrt{5}}{2}$ is the golden ratio!

**Why the golden ratio?** Donald Knuth suggested this because $\phi$ has nice properties related to distributing points uniformly. When you multiply by a number related to $\phi$, you get good distribution properties (related to Fibonacci hashing and the low-discrepancy sequences).

For 64-bit keys, the analogous constant is $a = 11400714819323198485$.

## Comparison with Prime-Based Hashing

| Feature | Prime-Based Polynomial | Multiply-Shift |
|---------|----------------------|----------------|
| **Theoretical Guarantee** | Strictly universal | Almost universal |
| **Table Size** | Any prime $M$ | Power of 2: $M = 2^b$ |
| **Speed** | Slower (modulo by prime) | Faster (shift operation) |
| **Implementation** | Need to find primes | Just pick odd $a$ |
| **Practical Performance** | Excellent | Excellent |

## When to Use What?

**Use Multiply-Shift when:**
- You care about speed (most of the time!)
- You're okay with power-of-2 table sizes
- You're building practical systems (hash tables, caches, etc.)
- "Almost universal" is good enough for your use case

**Use Prime-Based when:**
- You need strict theoretical guarantees
- You're writing a paper and need to prove universality
- Table size flexibility matters (can use any prime)
- You're implementing something that critically depends on exact universality

## The Bottom Line

Multiply-shift hashing is the "engineer's choice" - fast, simple, and effective. It sacrifices a tiny bit of theoretical purity for a huge gain in practical performance. 

The prime-based polynomial method is the "theorist's choice" - perfect for proofs and guarantees.

In practice? Most high-performance systems use multiply-shift or similar methods. The theoretical distinction rarely matters outside academia, and the speed difference definitely matters in production!

**The takeaway:** Universal hashing theory teaches you *what properties to aim for*, but there are many ways to get "close enough" that work brilliantly in practice. Multiply-shift is one of the best examples of this pragmatic approach. 🚀