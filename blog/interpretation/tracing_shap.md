@def title = "Tracing DeepSHAP to Lundberg's Original Paper"
@def published = "19 December 2025"
@def tags = ["interpretation"]

# Tracing DeepSHAP to Lundberg's Original Paper

## What Lundberg Actually Said (2017 NIPS Paper)

Let me trace the "double expectation" idea back to what Lundberg's original paper actually contains.

### The Key Quote from Section 4.2

From the paper (found in the search results):

> "If we interpret the reference value in Equation 3 as representing E[x] in Equation 12, then **DeepLIFT approximates SHAP values assuming that the input features are independent of one another and the deep model is linear.**"

That's basically it. This is the only explanation Lundberg gives for the connection.

### What is "Equation 12" in Lundberg's Paper?

From the search results, Equation 12 defines SHAP values using **conditional expectations**:

$$f_x(z') = f(h_x(z')) = \mathbb{E}[f(z) | z_S]$$

where $S$ is the set of non-zero indices in $z'$ (the features that are "present").

This is saying: the value function for coalition $S$ should be the **conditional expectation** of the model output given those features.

### What is "Equation 3" (DeepLIFT)?

DeepLIFT computes contributions $C_{\Delta x_i \Delta y}$ that satisfy:

$$f(x) - f(r) = \sum_i C_{\Delta x_i \Delta y}$$

where $r$ is a **reference value** (single baseline).

### Lundberg's Argument (Such As It Is)

Lundberg is saying:

1. DeepLIFT uses a single reference $r$
2. If you interpret this reference as $\mathbb{E}[x]$ (the expected value)
3. And if features are independent
4. And if the model is locally linear
5. Then DeepLIFT contributions approximate SHAP values

**That's the entire justification.** No derivation of double expectations, no formal proof, just this claim.

---

## Where Does the "Double Expectation" Come From?

Based on the search results, here's what I can trace:

### From the SHAP Documentation

The official SHAP library documentation states:

> "By integrating over many background samples, Deep[SHAP] estimates approximate SHAP values such that they sum up to the difference between the expected model output on the passed background samples and the current model output (f(x) - E[f(x)])."

This suggests:
- DeepSHAP averages over **multiple background samples** (not just one reference)
- This averaging approximates the expectation $\mathbb{E}[f(x)]$

### The Implementation Insight

From the GitHub SHAP repository description:

> "The implementation here differs from the original DeepLIFT by using **a distribution of background samples instead of a single reference value**, and using Shapley equations to linearize components such as max, softmax, products, divisions, etc."

So DeepSHAP = DeepLIFT + averaging over multiple references.

### The Logic (Reconstructed)

Here's how the "double expectation" view arises, though **Lundberg never writes it this way explicitly**:

**Step 1:** True SHAP values require computing:
$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \text{weight}(S) \cdot [v(S \cup \{i\}) - v(S)]$$

where $v(S) = \mathbb{E}[f(z) | z_S]$ is the conditional expectation.

**Step 2:** Under the independence assumption, replace conditional with marginal:
$$v(S) \approx \mathbb{E}_{z_{\bar{S}}}[f(z_S, z_{\bar{S}})]$$

This is the first expectation (over missing features).

**Step 3:** Approximate this expectation by sampling references:
$$v(S) \approx \frac{1}{|R|} \sum_{r \in R} f(z_S, r_{\bar{S}})$$

**Step 4:** DeepLIFT with reference $r$ gives you contributions for that specific $r$.

**Step 5:** Average over multiple references:
$$\phi_i \approx \frac{1}{|R|} \sum_{r \in R} C_{\Delta x_i \Delta y}(x, r)$$

This is the second expectation (over references).

**Combined:** You get $\mathbb{E}_S \mathbb{E}_r[\ldots]$ where:
- First expectation: over coalitions (implicit in the Shapley formula)
- Second expectation: over reference samples (explicit in DeepSHAP implementation)

---

## Is the "Double Expectation" View Correct?

### What we can verify:

**✓ Lundberg says:** DeepLIFT approximates SHAP under independence assumption

**✓ SHAP library implements:** Averaging DeepLIFT over multiple background samples

**✓ Mathematical logic:** Shapley values involve summing over coalitions (first expectation), and DeepSHAP averages over references (second expectation)

### What we CANNOT verify from Lundberg's paper:

**✗ Explicit double expectation formula:** Never appears in the 2017 paper

**✗ Formal proof:** That averaging DeepLIFT over references equals $\mathbb{E}_S[...]$

**✗ Derivation:** Showing how DeepLIFT's multipliers relate to coalition sampling

### The honest assessment:

The "double expectation" framework is a **pedagogical reconstruction** that makes mathematical sense of what DeepSHAP does, but it's not how Lundberg originally presented it. The original paper just asserts the connection with minimal justification.

The framework from your original document is **logically sound** as an explanation of why DeepSHAP works, but it's not directly traceable to Lundberg's paper. It appears to be a post-hoc rationalization by the interpretation community.

---

## What Does the 2022 Nature Communications Paper Say?

From the search results, Chen, Lundberg, and Lee (2022) state:

> "DeepSHAP was originally introduced as an adaptation of DeepLIFT in the original SHAP paper **but is briefly and informally introduced**, making it difficult to know exactly what the method entails"

Even the authors themselves admit the original presentation was informal!

The 2022 paper provides a more rigorous treatment but focuses on extending DeepSHAP to series of models, not on deriving the basic double expectation formula.

---

## Bottom Line

**Is the double expectation correct?** 

Probably yes, as a way to understand what DeepSHAP computes.

**Does it come from Lundberg's 2017 paper?** 

No. The paper just asserts that DeepLIFT approximates SHAP under independence, without this level of mathematical detail.

**Where does it come from?**

It's a **reasonable reconstruction** by people trying to understand:
1. What SHAP values actually require (coalition sampling)
2. What DeepLIFT computes (contributions for one reference)
3. What DeepSHAP does (averages over multiple references)
4. How these connect (double expectation interpretation)

The framework is useful for understanding, but it's not authoritative source material from Lundberg's original work.