---
name: "model-escalation"
description: "Formalize the local-first to OpenRouter cloud-review workflow."
---

# Model Escalation Protocol

The default development model is the local model. 

1. **Local Model (Default)**:
   Use the local model for standard coding, unit tests, simple debugging, documentation, standard PyTorch/NumPy work, and test execution loops.

2. **Cloud Escalation (OpenRouter)**:
   Use the stronger cloud model ONLY when the task requires significantly stronger reasoning or broader synthesis. Examples:
   - Ambiguous scientific reasoning
   - Complex mathematical derivations
   - Advanced control design
   - Difficult multi-module debugging
   - Second-opinion review of an important design decision

3. **Escalation Process**:
   Escalation must be deliberate. When escalating, clearly state the specific question or ambiguity that requires cloud reasoning, gather the cloud model's conclusion, and then apply the approved solution back in the local workflow.
