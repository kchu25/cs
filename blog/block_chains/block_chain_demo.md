@def title = "Building a Simple Blockchain in Julia"
@def published = "27 November 2025"
@def tags = ["one-or-few-shots-learning"]

# Building a Simple Blockchain in Julia

Hey! So you want to build a blockchain in Julia? Absolutely possible, and I'll show you a minimal example that captures the core concepts.

## What Makes a Blockchain?

At its heart, a blockchain is just a linked list of blocks, where each block contains:
- Some data
- A timestamp
- A hash of the previous block (this creates the "chain")
- Its own hash

## The Code

Let's build it step by step:

```julia
using SHA
using Dates

# Define what a Block looks like
mutable struct Block
    index::Int
    timestamp::DateTime
    data::String
    previous_hash::String
    hash::String
end

# Function to calculate a block's hash
function calculate_hash(block::Block)
    content = string(block.index, block.timestamp, block.data, block.previous_hash)
    return bytes2hex(sha256(content))
end

# Create the first block (the "genesis block")
function create_genesis_block()
    block = Block(0, now(), "Genesis Block", "0", "")
    block.hash = calculate_hash(block)
    return block
end

# Add a new block to the chain
function add_block(blockchain::Vector{Block}, data::String)
    previous_block = blockchain[end]
    new_index = previous_block.index + 1
    new_timestamp = now()
    new_previous_hash = previous_block.hash
    
    new_block = Block(new_index, new_timestamp, data, new_previous_hash, "")
    new_block.hash = calculate_hash(new_block)
    
    push!(blockchain, new_block)
    return new_block
end

# Verify the blockchain is valid
function is_chain_valid(blockchain::Vector{Block})
    for i in 2:length(blockchain)
        current = blockchain[i]
        previous = blockchain[i-1]
        
        # Check if the hash is correct
        if current.hash != calculate_hash(current)
            return false
        end
        
        # Check if blocks are properly linked
        if current.previous_hash != previous.hash
            return false
        end
    end
    return true
end
```

## Let's Use It!

```julia
# Create our blockchain
blockchain = [create_genesis_block()]

# Add some blocks
add_block(blockchain, "First transaction: Alice sends 10 coins to Bob")
add_block(blockchain, "Second transaction: Bob sends 5 coins to Charlie")
add_block(blockchain, "Third transaction: Charlie sends 2 coins to Alice")

# Print the blockchain
println("Our Blockchain:")
println("=" ^ 50)
for block in blockchain
    println("Block #$(block.index)")
    println("  Timestamp: $(block.timestamp)")
    println("  Data: $(block.data)")
    println("  Previous Hash: $(block.previous_hash[1:min(10, length(block.previous_hash))])...")
    println("  Hash: $(block.hash[1:10])...")
    println()
end

# Verify it's valid
println("Is blockchain valid? $(is_chain_valid(blockchain))")
```

## What's Happening Here?

The magic is in the **hashing**. Each block contains a hash of the previous block, creating an unbreakable chain. If someone tries to tamper with a block's data, its hash changes, which breaks the link to the next block. This is what makes blockchains tamper-evident!

The `SHA256` algorithm takes any input and produces a unique fingerprint. Change even one character in the data, and you get a completely different hash.

## Understanding the Hash Fields

You might notice each block stores **two** hash values:

1. **`hash`** - The block's own fingerprint/ID, calculated from its index, timestamp, data, and previous_hash
2. **`previous_hash`** - A copy of the parent block's hash, creating the chain link

Why both? Think of it like a linked list:
```
Block 0               Block 1                  Block 2
hash: "abc123"        hash: "def456"           hash: "ghi789"
previous_hash: "0"    previous_hash: "abc123"  previous_hash: "def456"
                      └─ points to Block 0     └─ points to Block 1
```

When you calculate a block's hash, you're hashing these specific fields (not the entire block object):
- `index`, `timestamp`, `data`, `previous_hash`

Notice the `hash` field itself isn't included - that would be circular!

## Does Order in the Vector Matter?

Here's something interesting: **the order in the vector doesn't technically matter for the blockchain's integrity!** The chain is defined by the cryptographic links (`previous_hash` → `hash`), not by array position.

You could shuffle the blocks randomly in the vector, and as long as you can follow the hash pointers, you could reconstruct the correct order. The `index` field and hash links preserve the true sequence.

However, in practice, we keep them ordered in the vector because:
- It's way more convenient for humans to read
- Validation is simpler when you can just iterate forward
- Adding new blocks is easier (just append to the end)

But the security? That comes purely from the cryptographic chain of hashes, not the array order!

## Try It Out

Copy this code into a Julia REPL and run it! You'll see your own little blockchain being created. It's missing features like proof-of-work (mining) and a distributed network, but it has the fundamental structure that makes blockchains work.

Pretty cool, right? This is the same basic idea behind Bitcoin and other cryptocurrencies, just without all the complexity of consensus algorithms and peer-to-peer networking!