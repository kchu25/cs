@def title = "Why Hashing is Great: An Intuitive Guide"
@def published = "6 October 2025"
@def tags = ["hashing"]



# Why Hashing is Great: An Intuitive Guide

## The Big Picture: Why We Love Hashing

Imagine you're organizing a massive library with millions of books, but you only need quick access to a few thousand of them. You *could* sort them all and use binary search, but hashing offers something magical: **essentially constant-time lookups** with remarkably simple code.

### What Makes Hashing So Good?

**1. Blazingly Fast Operations**
- **Lookup**: $O(1)$ expected time - just compute where the item should be and check that spot
- **Insert**: $O(1)$ - even simpler! Just plop it in the right bucket
- **Delete**: $O(1)$ expected time - find it and remove it from its list

**2. Ridiculously Simple to Implement**
Unlike balanced trees with their rotations and rebalancing rituals, hashing is straightforward:
- Have an array $A$ of size $M$
- Pick a hash function $h: U \to \Set{0, 1, \ldots, M-1}$
- To store key $x$: put it in bucket $A[h(x)]$
- Handle collisions with linked lists (or other methods)

That's it! No complex tree invariants to maintain.

**3. Space Efficient with the Right Approach**
Using **universal hashing**, you can achieve great performance with table size $M = O(N)$ where $N$ is the number of elements you're actually storing.

---

## The Universal Hashing Magic Trick

Here's the catch with hashing: for *any* fixed hash function, an adversary could craft a terrible set of keys that all collide. It's like the pigeonhole principle coming back to haunt you.

**The Solution: Randomize Your Hash Function!**

This is the key insight of **universal hashing**:

$$\text{For any } x \neq y \text{ in the universe } U: \quad \Pr_{h \sim \mathcal{H}}[h(x) = h(y)] \leq \frac{1}{M}$$

Think of it like using randomization in algorithms: even if the data is adversarial, your algorithm's random choices prevent the adversary from consistently forcing bad outcomes!

### What This Guarantees

If you pick your hash function at random from a universal family, then for any element $x$, the expected number of other elements in your set $S$ that collide with $x$ is at most $\frac{N}{M}$.

*Here, $N$ is the number of elements you’re storing (as above), and $M$ is the number of buckets in your table. This comes from the universal property: for each of the $N-1$ other elements, the chance it collides with $x$ is at most $1/M$, so the expected number of collisions is at most $(N-1)/M \approx N/M$.*

In practical terms: if you set $M = N$, then on average, each bucket will only have a constant number of elements. This is what makes hash tables fast and efficient. In other words: if you make the number of buckets $M$ about equal to the number of items $N$, then most buckets will have just one or two elements, and operations like lookup or insert will be fast—on average, you only need to check a couple of items per operation.

Mathematically, if you have $N$ items and $M$ buckets, the expected number of items in any bucket is $N/M$. If you set $M = N$, then $N/M = 1$, so on average, each bucket has just one item. This is why we say the average bucket size is “constant” (doesn’t grow with $N$).

---

## Where Prime Numbers Enter the Scene

Now for the punchline about primes! The lecture describes a clever construction method that relies on prime numbers:

### The Prime-Based Construction

View your key $x$ as a vector of integers: $x = [x_1, x_2, \ldots, x_k]$ where each $x_i \in \{0, 1, \ldots, M-1\}$.

**Require: $M$ is prime!**

To create a hash function:
1. Pick random numbers $r_1, r_2, \ldots, r_k$ from $\Set{0, 1, \ldots, M-1}$
2. Define: $$h(x) = (r_1 x_1 + r_2 x_2 + \cdots + r_k x_k) \bmod M$$

### Why Do We Need $M$ to be Prime?

Here's the beautiful part! To prove this is universal, consider two different keys $x \neq y$. They must differ in some position $i$ where $x_i \neq y_i$.

After choosing all the other random coefficients $r_j$ (for $j \neq i$), we have a collision exactly when:

$$r_i(x_i - y_i) = h'(y) - h'(x) \pmod{M}$$

where $h'$ represents the partial sum without the $i$-th term.

**Here's where primality is crucial:** Since $M$ is prime and $x_i \neq y_i$, the quantity $(x_i - y_i)$ is non-zero mod $M$. In modular arithmetic with a prime modulus, **every non-zero element has a multiplicative inverse!**

This means we can "divide" both sides by $(x_i - y_i)$ to get:

$$r_i = \frac{h'(y) - h'(x)}{x_i - y_i} \pmod{M}$$

**The payoff**: There is *exactly one* value of $r_i$ (out of $M$ possibilities) that causes a collision. Since we chose $r_i$ uniformly at random, the collision probability is exactly $\frac{1}{M}$ - which is precisely what we need for universality!

### What If $M$ Wasn't Prime?

If $M$ were composite, division might not be well-defined. For example, if $M = 6$ and we needed to divide by 2, we'd have a problem because $2 \times 3 \equiv 0 \pmod{6}$. Multiple values might cause collisions, breaking our careful $\frac{1}{M}$ probability bound.

---

**🤔 SIDE NOTE: "But wait... if $M$ is huge and my numbers are small, does the prime thing even matter?"**

You know what, I had the exact same thought! Like, if $M = 10007$ and all my data is under 1000, then $(x_i - y_i)$ is way smaller than $M$. Why would I care if $M$ is prime?

So here's the thing - it's not really about size, it's about factors. Check this out: say $M = 1000$ (nice and big, not prime). All my data is under 100. I'm thinking "perfect, plenty of room!" Then I get $x_i = 60$ and $y_i = 10$, so the difference is 50. That's tiny compared to 1000, right?

Except 50 and 1000 share a factor (actually $\gcd(50, 1000) = 50$), which means 50 has no multiplicative inverse mod 1000. The proof we just did? Completely breaks. Not because the numbers were too big, but because they happened to share factors with $M$.

And here's the kicker - you'd need to guarantee this never happens for *any* pair of inputs you might see. Unless you know your data really well, that's a nightmare to verify. Prime $M$ just... handles it. Every non-zero difference works, period. No surprises three months into production when some weird input combination breaks everything.

So yeah, "big enough $M$" isn't quite the same as "prime $M$" - primality is what gives you the ironclad guarantee regardless of what data shows up.

---

## Connection to Count-Min Sketch

Speaking of universal hashing in the wild - **Count-Min Sketch** is a perfect example of why this theory matters in practice!

Quick refresher: Count-Min Sketch is a probabilistic data structure for frequency estimation in streams. You have a $d \times w$ table (depth $d$, width $w$) and $d$ independent hash functions $h_1, \ldots, h_d$, each mapping items to $\{0, 1, \ldots, w-1\}$.

### Why Universal Hashing is Perfect Here

When you use the prime-based universal hash construction for Count-Min Sketch:

**1. Independent Hash Functions**
You pick $d$ different random coefficient vectors $(r_1^{(j)}, r_2^{(j)}, \ldots, r_k^{(j)})$ for $j = 1, \ldots, d$. Each gives you an independent hash function from the universal family. This independence is *crucial* for Count-Min's error guarantees!

**2. The Error Bound Relies on Universality**
The beautiful error guarantee of Count-Min Sketch is:
$\text{Estimated frequency} \leq \text{True frequency} + \frac{2}{w} \times \text{Total count}$

with probability at least $1 - e^{-d}$.

This bound **directly depends** on the collision probability being $\leq 1/w$! If your hash function isn't universal, all bets are off. You could have way more collisions than expected, and your estimates would be garbage.

**3. Prime Width is Standard**
In practice, many Count-Min Sketch implementations choose $w$ to be prime (or close to prime) precisely because they're using this universal hash construction! It's not just theoretical pedantry - it's ensuring your frequency estimates are actually reliable.

**The Practical Upshot**: When you see Count-Min Sketch in a paper or library using $h_i(x) = (a_i \cdot x + b_i) \bmod p$ where $p$ is prime, now you know *exactly* why - it's leveraging this universal hashing result to get provable guarantees on estimation error!

Without universality, Count-Min Sketch would just be "some heuristic that sometimes works" instead of a principled data structure with solid probabilistic guarantees. That's the power of this theory! 🎯

---

Q: can CMS work well even without taking prime as a consideration, as we can use arbitrarily large prime number?

Great question! You've hit on something important - the difference between theory and practice.
Short answer: Yes, Count-Min Sketch works well in practice even without worrying about primes!
Here's the thing - you could absolutely pick a prime close to whatever ww
w you want (like 1009 instead of 1000), and that would work perfectly. But honestly, most people just use powers of 2 (like 1024) with hash families designed for powers of 2, and it works great.

The prime requirement is really about **proving** the theoretical guarantee, not about making Count-Min actually function. If you use $w=1000$
 with polynomial hashing, it'll probably work fine in practice - you just can't *prove* the collision probability is exactly  $\leq 1/w$
 for all possible inputs.

The practical reality: Count-Min is pretty forgiving. Engineers typically optimize for speed (powers of 2) rather than theoretical purity (primes), and it works well. The universal hashing theory tells you how to get ironclad guarantees, but "pretty good" hashing is often good enough for real applications.
So yeah, you're right - since you can pick arbitrarily large primes, the prime thing isn't really a limitation. It's more about whether you care about having a mathematical proof versus just knowing "this works well in practice."


---

## The Perfect Hashing Bonus Round

The lecture culminates in **perfect hashing**: achieving $O(1)$ worst-case lookups (not just expected) with only $O(N)$ space!

The two-level scheme is clever:
1. **First level**: Hash $N$ items into table of size $N$ using universal hashing
2. **Second level**: For each bucket with $n_i$ collisions, create a secondary table of size $n_i^2$ (ensuring no collisions)

The magic: with proper choice of hash function, $\sum_i n_i^2 = O(N)$ in expectation!

---

## The Bottom Line

Hashing is great because it's:
- **Fast**: constant expected time for all operations
- **Simple**: minimal code complexity
- **Flexible**: works for any crazy universe of keys
- **Theoretically sound**: universal hashing gives strong guarantees

And primes? They're the secret sauce that makes the multiplication method work, giving us a clean, efficient way to construct universal hash families through the magic of modular arithmetic!