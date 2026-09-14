---
name: "wec-context"
description: "Ensure the agent understands the repository and current state before proceeding."
---

# WEC Context Protocol

When instructed to load context or familiarize yourself with the repository, you MUST follow these steps in order:

1. **Read Core Files First**:
   Read `AGENTS.md` and `PROJECT_STATE.md` to understand your role and the current state of the project.

2. **Selective Review**:
   Based on the task at hand, selectively review the following architecture documents if necessary:
   - `ARCHITECTURE.md`: For structural understanding.
   - `MODULE_MAP.md`: To understand module boundaries.
   - `SCIENCE_RULES.md`: To enforce scientific invariants.
   - `AI_WORKFLOW.md`: To understand the AI-assisted engineering loop.
   - `MODEL_ROUTING.md`: To understand when to escalate to cloud reasoning.

3. **Avoid Guessing**:
   Never assume the state of the project. Always rely on these documents as the ground truth.
