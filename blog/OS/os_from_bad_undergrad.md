@def title = "Core Mental Models for Operating Systems"
@def published = "29 December 2025"
@def tags = ["machine-learning", "sparsity"]

# Core Mental Models for Operating Systems

Hey, I totally get your frustration. OS courses often feel like a grab bag of random topics, but there really is a unifying logic underneath. Let me break down the core mental models—including the mathematical ones—that make everything click.

## The Big Picture

An OS exists to solve one fundamental problem: You have limited physical resources (CPU, memory, disk, network) and many programs that want to use them simultaneously.

Everything boils down to three challenges:
1. **Virtualization** - Making limited resources appear unlimited
2. **Concurrency** - Managing things happening "at the same time" 
3. **Persistence** - Keeping data safe despite failures

## The Mathematical Toolkit

Here are the formal structures that let you *reason* about systems:

| Math Structure | Formal Definition | OS Applications | Key Property |
|---------------|-------------------|-----------------|--------------|
| **State Machine** | $(S, s_0, E, \delta)$ where $\delta: S \times E \to S$ | Process states, lock states, page states | Reachability analysis: "Can we reach deadlock state?" |
| **Partial Order** | Relation $\to$ that's transitive, irreflexive, antisymmetric | Happens-before, event ordering | If $a \not\to b$ and $b \not\to a$, then $a \parallel b$ (concurrent) |
| **Invariant** | Predicate $P(s)$ true for all reachable states $s$ | Mutual exclusion, memory safety | $P(s) = \text{true} \implies$ system is correct |
| **Graph** | $G = (V, E)$ | Resource allocation, wait-for graphs | Cycle detection = deadlock detection |
| **Function/Mapping** | $f: A \to B$ | Page tables, file descriptors | Composition: $(f \circ g)(x) = f(g(x))$ |
| **Probability** | $P(X), E[X], \text{Var}(X)$ | Queueing models, performance analysis | Little's Law: $L = \lambda W$ |

### State Machines in Detail

**Components:**
- $S$ = set of states (e.g., $\{\text{Ready}, \text{Running}, \text{Waiting}, \text{Terminated}\}$)
- $s_0$ = initial state
- $E$ = events (e.g., schedule, block, interrupt)
- $\delta$ = transition function

**Why it matters:** You can prove properties by checking all reachable states. Safety = "bad states unreachable."

### Partial Orders (Happens-Before)

**Definition:** $a \to b$ means "$a$ happens before $b$"

**Properties:**
- Transitive: $a \to b \land b \to c \implies a \to c$
- Irreflexive: $a \not\to a$
- Antisymmetric: $a \to b \implies b \not\to a$

**Concurrency:** $a \parallel b$ iff $a \not\to b$ and $b \not\to a$

**Critical insight:** Race conditions occur when concurrent events access shared state.

**Example:**
```
Thread 1: write(x); read(y);     a₁ → b₁
Thread 2: write(y); read(x);     a₂ → b₂
```
Are $a_1$ and $a_2$ ordered? No! So 4 possible execution orders exist.

### Invariants

**Definition:** Predicate that's always true: $\forall s \in \text{ReachableStates}, P(s) = \text{true}$

| System Component | Invariant Example |
|-----------------|-------------------|
| Mutex | $\|\{\text{threads in critical section}\}| \leq 1$ |
| Memory allocator | $\sum \text{allocated} + \sum \text{free} = \text{total}$ |
| Balanced tree | $\|\text{height}(\text{left}) - \text{height}(\text{right})\| \leq 1$ |
| File system | $\text{blocks\_used} = \sum_{i} \text{file\_sizes}[i]$ |

**Bugs = Invariant violations.** If you can state your invariants precisely, you can reason about correctness.

### Graph Theory for Resources

**Resource Allocation Graph:**
- Nodes: $V = P \cup R$ (processes and resources)
- Edges: $P \to R$ (request), $R \to P$ (allocated)
- **Cycle $\implies$ potential deadlock**

**Wait-For Graph (simplified):**
- Nodes: $V = P$ (processes only)
- Edge: $P_i \to P_j$ means "$P_i$ waits for $P_j$"
- **Cycle $\iff$ deadlock**

**Deadlock conditions (Coffman):** All must hold simultaneously:
1. Mutual exclusion
2. Hold and wait
3. No preemption
4. Circular wait (cycle in graph)

Break any one $\implies$ no deadlock.

### Queueing Theory

**Little's Law:** 
$$L = \lambda W$$

Where:
- $L$ = average number in system
- $\lambda$ = arrival rate
- $W$ = average time in system

**Utilization:** $U = \lambda \times S$ (arrival rate × service time)
- If $U < 1$: stable
- If $U \geq 1$: queue grows unbounded

**Application:** If processes arrive at rate $\lambda = 10/\text{sec}$ and each takes $S = 0.08$ sec, then $U = 0.8$ (80% CPU utilization), and on average $L = 10 \times W$ processes are in the system.

## Connecting Math to Core Concepts

### Virtualization = Function Composition

Virtual memory is layers of mappings:
$$\text{Virtual Addr} \xrightarrow{\text{TLB}} \text{Page Table} \xrightarrow{f} \text{Physical Addr} \xrightarrow{g} \text{Cache} \xrightarrow{h} \text{DRAM}$$

Result: $(h \circ g \circ f)(\text{addr})$ — multiple levels of indirection.

### Concurrency = Product of State Machines

Two threads $T_1, T_2$ with state spaces $S_1, S_2$:
$$\text{Combined system state space} = S_1 \times S_2$$

If $|S_1| = n$ and $|S_2| = m$, combined system has $n \times m$ states.

**State explosion:** With $k$ threads, state space grows as $\prod_{i=1}^k |S_i|$ — exponential! This is why concurrency is hard.

### Persistence = Idempotence + Ordering

**Idempotence:** $f(f(x)) = f(x)$

| Operation | Idempotent? | Why? |
|-----------|-------------|------|
| `SET x = 5` | ✓ | Applying twice = once |
| `x = x + 1` | ✗ | Effect accumulates |
| `DELETE file` | ✓ | Second delete is no-op |
| `APPEND log` | ✗ | Creates duplicate entries |

**Write-Ahead Logging:** Log operation $O$ before applying it. On crash, replay log. Works because logged operations are designed to be idempotent.

## The Core Design Patterns

### 1. Time-Multiplexing vs Space-Multiplexing

| Strategy | How it Works | Examples |
|----------|--------------|----------|
| **Time-multiplexing** | Share by taking turns | CPU scheduling, network links |
| **Space-multiplexing** | Partition and allocate | Memory allocation, disk blocks |

### 2. Mechanism vs Policy

| Component | Mechanism (How) | Policy (When/What) |
|-----------|----------------|-------------------|
| **Scheduling** | Context switch | Which process to run (RR, Priority) |
| **Memory** | Page allocation | Which page to evict (LRU, FIFO) |
| **File system** | Block I/O | Which blocks to cache |

### 3. The Tradeoff Space

Every OS decision is optimization under constraints:

$$\text{Maximize } f(\text{throughput, fairness, latency}) \text{ subject to } g(\text{resources}) \leq C$$

Examples:
- Scheduling: Maximize throughput subject to fairness constraints
- Caching: Maximize hit rate subject to memory limit
- Page replacement: Minimize page faults subject to available frames

## How This Connects to Distributed Systems

Distributed systems = OS concepts + Networks + Partial failures

| OS Concept | Distributed Systems Extension |
|------------|-------------------------------|
| Happens-before $\to$ | Vector clocks, causal consistency |
| State machines | Replicated state machines, Raft/Paxos |
| Invariants | Safety properties, consistency models |
| Partial orders | CRDTs, eventual consistency |
| Queueing theory | Load balancing, request routing |
| Graphs | Cluster topology, gossip protocols |

The math stays the same, but now:
- No shared memory (only message passing)
- No global clock (only partial orders)
- Failures are common (need probability models)

## How to Build Competence

1. **Draw state machines:** For any mechanism, sketch states and transitions
2. **State invariants:** What must always be true? Can you prove it holds?
3. **Find the partial order:** What operations can happen concurrently?
4. **Model as optimization:** What are you maximizing? What are constraints?
5. **Code something:** Implement a lock, allocator, or scheduler

The goal isn't memorizing algorithms — it's recognizing patterns: "This is a partial order problem" or "We need to maintain this invariant."

## Why Optimization Problems Dominate OS Research

> **Yes, exactly!** OS research is fundamentally optimization research because every resource management problem is a constrained optimization problem.
> 
> Think about it: you have **limited** resources (CPU cycles, memory, disk bandwidth, network) and **competing** demands from multiple processes. The OS can't satisfy everyone perfectly, so it has to make tradeoffs.
> 
> **Why optimization shows up everywhere:**
> 
> - **Scheduling:** You want low latency for interactive apps, high throughput for batch jobs, and fairness so no process starves. Can't maximize all three simultaneously—so you optimize a weighted combination.
> 
> - **Page replacement:** You want to minimize page faults, but you only have a fixed number of physical frames. Which pages do you evict? This is literally the "online caching problem" in algorithms research.
> 
> - **Disk scheduling:** Minimize seek time (throughput) vs. prevent starvation (fairness). The "elevator algorithm" is solving an optimization problem in real-time.
> 
> - **Power management:** Maximize performance while minimizing energy consumption. Modern CPUs constantly solve this optimization (dynamic voltage/frequency scaling).
> 
> **The research angle:** Classic algorithms assume you know the future (optimal page replacement needs to know future accesses). OS research asks: "What's the best we can do **online** with **limited information**?" This leads to competitive analysis, machine learning for prediction, adaptive algorithms, etc.
> 
> When you see papers with titles like "Optimal X under Y constraints" or "Learning-based Z scheduling"—they're all variations of this core optimization framework. The math might get fancy (Markov decision processes, linear programming, game theory), but the structure is always: maximize something good, subject to constraints on resources.
> 
> **Distributed systems takes this further:** Now you're optimizing across multiple machines with network delays and failures. Same math, harder constraints.

## The Key Insight

OS design is **reasoning under constraints**:
- Finite resources → optimization theory
- Concurrent execution → partial orders, state machines  
- Failures → probability, invariants

When you see these mathematical structures, you can *prove* correctness rather than just hope for it.

You got a B not because you weren't capable, but because the course taught mechanisms without these unifying models. Now you have the framework to make it all coherent.