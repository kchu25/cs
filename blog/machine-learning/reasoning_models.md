@def title = "Getting Started with Reasoning Models"
@def published = "6 February 2026"
@def tags = ["machine-learning"]

# Getting Started with Reasoning Models

## What Is "Reasoning" in the Context of LLMs?

When we say a model "reasons," we mean it generates an explicit **chain of thought** (CoT) before producing a final answer. Instead of jumping straight to an output, the model works through intermediate steps --- much like showing your work on a math exam.

A standard LLM prompt-response looks like this:

```
User: What is 27 * 34?
Assistant: 918
```

A reasoning model instead produces:

```
User: What is 27 * 34?
Assistant: <think>
I need to multiply 27 by 34.
27 * 34 = 27 * 30 + 27 * 4 = 810 + 108 = 918
</think>
918
```

The key structural innovation is the `<think>...</think>` block: a **scratchpad** the model uses to work through the problem before committing to an answer. The user sees only the final answer, but the reasoning trace can be inspected for debugging, auditing, or trust.

### Formal View

Given an input $x$, a standard model produces output $y = f(x)$. A reasoning model produces a trace $t$ and then an answer conditioned on it:

$$t \sim p(t \mid x), \quad y \sim p(y \mid x, t)$$

The trace $t$ is a variable-length sequence of tokens that captures the model's intermediate reasoning. Training the model to produce useful traces is the core challenge.

---

## Why Does This Matter? The Scaling Axis Shift

Traditional LLM scaling focused on **train-time compute**: bigger models, more data, more FLOPs. Reasoning models introduce a complementary axis --- **test-time compute**. Instead of making the model larger, you let it *think longer* at inference.

This is a profound shift:

- **Train-time scaling**: Double your parameters and training data to get better answers
- **Test-time scaling**: Keep the same model but let it generate more tokens before answering

The practical implication: **smaller models that think longer can match or beat larger models that answer immediately.** This is exactly why reasoning models can be surprisingly effective even at 1.5B--7B parameter scales.

---

## How Are Reasoning Models Trained?

There are two main approaches, and they are often combined.

### Approach 1: Distillation (Supervised Fine-Tuning on Reasoning Traces)

The simplest recipe:

1. Take a powerful reasoning model (e.g., DeepSeek-R1, a 671B MoE model)
2. Generate chain-of-thought traces on a dataset of problems
3. Fine-tune a smaller model on these (problem, trace, answer) triples

This is how the DeepSeek-R1-Distill family was created. The paper showed that distilling R1's traces into Qwen and Llama base models at 1.5B, 7B, 8B, 14B, 32B, and 70B scales produced strong reasoning performance. The 14B distilled model even outperformed OpenAI's o1-mini on several benchmarks.

**Key insight from Open-R1 distillation experiments:**

- **Don't use sample packing** --- it significantly hurts performance for long reasoning traces because questions and answers can get split across chunks
- **Use larger learning rates** (4e-5 vs. the usual 2e-5) --- each doubling gained ~10 points on LiveCodeBench
- **Prefill with `<think>`** --- distilled models sometimes revert to non-reasoning behavior; forcing the thinking token in the prompt template fixes this

### Approach 2: Reinforcement Learning (the DeepSeek-R1 Recipe)

This is the more exciting (and harder) path. DeepSeek-R1 showed you can **incentivize reasoning from scratch** using RL, without human-annotated reasoning traces.

**The pipeline:**

1. **Start with a base model** that has been pretrained and instruction-tuned
2. **Apply RL with verifiable rewards** --- for math, check if the answer is correct; for code, run test cases
3. **The model discovers reasoning patterns on its own** --- self-reflection, verification, backtracking

The specific RL algorithm used is **GRPO (Group Relative Policy Optimization)**, a variant of PPO that is more memory-efficient because it doesn't need a separate value model. Instead, GRPO:

- Generates a **group** of $G$ candidate outputs for each prompt
- Scores them with a reward function
- Computes advantages **relative to the group** (normalized within-group rewards)
- Updates the policy to increase the probability of higher-reward outputs

$$\mathcal{L}(\theta) = -\E\left[\frac{1}{G}\sum_{i=1}^{G} \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i, \; \text{clip}(\cdot) A_i \right) - \beta \, D_{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

where $A_i$ is the advantage computed from the group's rewards and the KL term prevents the model from drifting too far from the reference policy.

**The remarkable finding (R1-Zero):** When you apply GRPO directly to a base model without any supervised fine-tuning, the model spontaneously develops:

- **Self-verification**: "Let me check this answer..."
- **Reflection**: "Wait, I made an error, let me reconsider..."
- **Exploration of multiple strategies**: trying different approaches before settling on one

These behaviors **emerge** from the RL process without being explicitly taught. This was one of the most surprising results of the DeepSeek-R1 paper.

---

## The Connection to Agentic AI

Yes, reasoning and agentic AI are deeply connected. Here's how:

### What Makes an "Agent"?

An AI agent is a system that can:

1. **Plan**: Break a complex task into steps
2. **Act**: Execute actions (tool calls, code execution, API requests)
3. **Observe**: Process the results of actions
4. **Reflect**: Evaluate progress and adjust the plan

### Reasoning Is the Engine of Agency

Every one of those capabilities requires **reasoning**:

- **Planning** = reasoning about what steps are needed (chain of thought about task decomposition)
- **Tool use** = reasoning about which tool to call and what arguments to pass
- **Reflection** = reasoning about whether the current approach is working

Without reasoning, an agent is just a model making one-shot decisions. With reasoning, it can maintain a coherent plan, recover from mistakes, and adapt its strategy.

### The Practical Connection

Modern agentic frameworks (LangChain, CrewAI, AutoGen, smolagents) work better with reasoning models because:

- **Longer context reasoning** allows the model to keep track of multi-step plans
- **Self-correction** means the agent can recover when a tool call fails
- **Explicit reasoning traces** make it possible to debug agent behavior

Think of it this way: **reasoning models are to agents what a prefrontal cortex is to a human.** The agent framework provides the body (tools, memory, environment), and the reasoning model provides the deliberative thinking.

---

## Small Reasoning Models: Surprisingly Capable

One of the most exciting developments in 2025--2026 is that **reasoning ability scales better with test-time compute than with parameter count**. This means small models can punch far above their weight.

### Evidence

| Model | Parameters | Notable Results |
|-------|-----------|----------------|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | Competitive on GSM8K, useful for simple reasoning tasks |
| DeepSeek-R1-Distill-Qwen-7B | 7B | Outperforms non-reasoning 70B models on math benchmarks |
| DeepSeek-R1-Distill-Qwen-14B | 14B | Surpasses OpenAI o1-mini on many benchmarks |
| DeepSeek-R1-Distill-Qwen-32B | 32B | Close to DeepSeek-R1 (671B MoE) on many tasks |
| QwQ-32B | 32B | Competitive with much larger reasoning models |
| OlympicCoder-7B | 7B | Matches DeepSeek's distilled model on competitive programming |

### Why Small Models Can Reason

The intuition: reasoning traces **offload computation from parameters to tokens**. A small model with a 2000-token reasoning trace is effectively doing more "compute" per problem than a large model giving a one-shot answer. The trace acts as external working memory.

Also, for verifiable domains (math, code), GRPO-style RL can be applied even to 1.5B models. Unsloth showed GRPO training on a 1.5B model requires as little as ~5GB of VRAM with LoRA.

---

## Your DGX Spark: What Can You Do?

The NVIDIA DGX Spark is powered by the GB10 Grace Blackwell Superchip with **128GB of unified memory** and up to **1 petaFLOP of FP4 AI performance**. This is a serious local AI workstation. Here's what's realistic:

### Inference: Run Reasoning Models Locally

With 128GB of unified memory, you can run large models that wouldn't fit on consumer GPUs:

| Model | Size (approx) | Fits on DGX Spark? | Notes |
|-------|------|-------------------|-------|
| DeepSeek-R1-Distill-Qwen-1.5B | ~3GB | Yes, easily | Great for experimentation |
| DeepSeek-R1-Distill-Qwen-7B | ~14GB (FP16) | Yes | Sweet spot for local reasoning |
| DeepSeek-R1-Distill-Qwen-32B | ~64GB (FP16) | Yes | Strong reasoning, fits in memory |
| QwQ-32B | ~64GB (FP16) | Yes | Excellent reasoning model |
| DeepSeek-R1-Distill-Llama-70B | ~140GB (FP16), ~70GB (INT8) | Tight at INT8/FP8 | Possible with quantization |
| DeepSeek-R1 (671B MoE) | ~130GB (4-bit) | Possible with aggressive quantization | The full model, quantized |

**How to get started:**

```bash
# Option 1: Use Ollama (simplest)
curl -fsSL https://ollama.com/install.sh | sh
ollama run deepseek-r1:7b
ollama run deepseek-r1:32b

# Option 2: Use vLLM (better performance, OpenAI-compatible API)
pip install vllm
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# Option 3: Use NVIDIA NIM containers (pre-optimized for DGX Spark)
# NVIDIA ships NIM containers with the DGX software stack
# Check build.nvidia.com/spark for playbooks
```

### Fine-Tuning: Train Your Own Reasoning Model

DGX Spark supports fine-tuning models up to 70B parameters. For reasoning:

**Distillation (easiest):**

```bash
# Using Hugging Face TRL
pip install trl transformers datasets

# Fine-tune Qwen-2.5-1.5B on reasoning traces
# See: github.com/huggingface/open-r1
```

**GRPO Training (more advanced):**

```python
from trl import GRPOConfig, GRPOTrainer

# Define a reward function (e.g., math correctness)
def math_reward(completions, **kwargs):
    # Check if the answer is correct
    rewards = []
    for completion in completions:
        # Extract answer, verify against ground truth
        rewards.append(1.0 if is_correct(completion) else 0.0)
    return rewards

config = GRPOConfig(
    num_iterations=2,       # reuse samples (mu parameter)
    per_device_train_batch_size=4,
    num_generations=8,      # group size G
    max_completion_length=2048,
    learning_rate=5e-7,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=[math_reward],
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

Unsloth has demonstrated that GRPO + LoRA on a 1.5B model fits in ~5GB VRAM, so DGX Spark can comfortably handle this with room to spare for much larger models.

### Concrete Project Ideas

1. **Math reasoning tutor**: Fine-tune a 7B model with GRPO on math datasets (GSM8K, MATH) to build a local reasoning assistant that shows its work step by step

2. **Code reasoning agent**: Use OlympicCoder-7B or distill from R1 on your own code problems. Run it locally for private, offline code assistance

3. **Domain-specific reasoner**: Have a niche domain? Generate reasoning traces with a large cloud model, then distill into a small local model that runs on your Spark

4. **Agentic workflows**: Run a 32B reasoning model as the "brain" of a local agent that uses tools (file system, databases, APIs) --- all on-device, fully private

---

## A Practical Roadmap

If you're new to reasoning models, here is a suggested path:

### Week 1: Explore

- Install Ollama on your DGX Spark
- Run `deepseek-r1:7b` and `deepseek-r1:32b`
- Try math problems, logic puzzles, and code challenges
- Compare reasoning traces between model sizes

### Week 2: Understand the Theory

- Read the [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) (published in Nature, 2025)
- Work through the [HuggingFace reasoning course](https://huggingface.co/learn/nlp-course/chapter12/1)
- Understand GRPO vs PPO vs DPO

### Week 3: Distillation

- Pick a small base model (Qwen2.5-1.5B-Instruct or 7B)
- Generate reasoning traces using the 32B model on your Spark
- Fine-tune the small model on those traces using TRL
- Evaluate on held-out problems

### Week 4: RL Training

- Set up GRPO training with TRL or Open-R1
- Start with a simple verifiable domain (math, simple code)
- Train a 1.5B model to develop reasoning from scratch
- Compare RL-trained vs distilled models

---

## Key Resources

- **DeepSeek-R1 Paper**: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) --- the foundational paper
- **Open-R1 Project**: [github.com/huggingface/open-r1](https://github.com/huggingface/open-r1) --- open-source reproduction effort
- **TRL Library**: [github.com/huggingface/trl](https://github.com/huggingface/trl) --- GRPO implementation
- **HuggingFace Reasoning Course**: [Chapter 12](https://huggingface.co/learn/nlp-course/chapter12/1)
- **Unsloth**: [unsloth.ai](https://unsloth.ai) --- memory-efficient training, GRPO with LoRA
- **DGX Spark Playbooks**: [build.nvidia.com/spark](https://build.nvidia.com/spark) --- curated recipes for your hardware
- **Ollama**: [ollama.com](https://ollama.com) --- easiest way to run models locally

---

## The Bottom Line

Reasoning models represent a genuine paradigm shift: **test-time compute as a scaling axis**. The fact that a 7B model thinking for 30 seconds can outperform a 70B model answering instantly reshapes the economics of AI deployment.

With a DGX Spark's 128GB of unified memory, you're in a uniquely good position. You can run 32B reasoning models at full precision, fine-tune up to 70B, and train GRPO on smaller models --- all locally, all private, all without cloud costs.

The field is moving fast. DeepSeek-R1 was published in January 2025 and already published in Nature. Open-R1 is actively reproducing and improving the recipe. New distilled models drop weekly. The barrier to entry has never been lower.

Start with Ollama and a 7B model. See what reasoning looks like. Then go deeper.