@def title = "DeepLIFT Notation: Understanding the Subscripts"
@def published = "20 December 2025"
@def tags = ["interpretation"]

# DeepLIFT Notation: Understanding the Subscripts

In DeepLIFT, the notation $C_{\Delta x_i \Delta y}$ uses subscripts that represent **differences** (deltas). Here's what each component means:

## The Notation: $C_{\Delta x_i \Delta y}$

### $\Delta x_i$ - "Difference in feature $i$"
- $\Delta x_i = x_i - x_{i,\text{ref}}$
- This represents the **change in the input feature $i$** from its reference value
- It's the difference between the actual input value and the baseline/reference value for that specific feature

### $\Delta y$ - "Difference in output"
- $\Delta y = y - y_{\text{ref}}$
- This represents the **change in the model's output** from its reference prediction
- It's the difference between the actual prediction and the baseline prediction

## What $C_{\Delta x_i \Delta y}$ Represents

The full notation $C_{\Delta x_i \Delta y}$ is the **contribution score** that answers:

> "How much did the change in feature $i$ (i.e., $\Delta x_i$) contribute to the change in output (i.e., $\Delta y$)?"

### Alternative Notation: $C_{\Delta x_i \Delta f}$

You'll also commonly see $C_{\Delta x_i \Delta f}$ instead of $C_{\Delta x_i \Delta y}$, where:
- $f$ represents the model/function itself
- $\Delta f = f(x) - f(x_{\text{ref}})$ is the change in the function's output

Both notations mean the same thing:
- $C_{\Delta x_i \Delta y}$ - emphasizes output values
- $C_{\Delta x_i \Delta f}$ - emphasizes the function being explained

The $\Delta f$ version is arguably more consistent since it makes explicit that we're explaining a *function's* behavior.

## The Decomposition

DeepLIFT ensures that these contributions sum to the total output difference:

$$\Delta y = \sum_{i=1}^{M} C_{\Delta x_i \Delta y}$$

Or equivalently:

$$y - y_{\text{ref}} = \sum_{i=1}^{M} C_{\Delta x_i \Delta y}$$

This means: the total change in prediction equals the sum of all individual feature contributions.

## Intuitive Understanding

Think of it as a budget allocation problem:
- You have a **total change** in the output: $\Delta y$
- Each feature's change $\Delta x_i$ is "responsible" for part of this
- $C_{\Delta x_i \Delta y}$ tells you **how much of $\Delta y$ to attribute to $\Delta x_i$**

The subscripts emphasize that we're measuring **how differences in inputs relate to differences in outputs**, which is the core principle of DeepLIFT.