---
name: "run-tests"
description: "Run WEC digital twin tests using the tiered testing workflow."
---

# WEC Testing Protocol

Execute testing using the appropriate tier:

1. **Targeted Test**
   Run when modifying a specific function or class.
   `python -m pytest tests/path_to_specific_test.py -k "test_name" -v`

2. **Module Test**
   Run when modifying a module to ensure it didn't break internally.
   `python -m pytest tests/path_to_module/ -v`

3. **Integration Test**
   Run to ensure interfaces between modified modules are still intact.
   `python -m pytest tests/integration/ -v`

4. **Full Test (Sanity Check)**
   Run to ensure the entire Tier-1 closed loop is intact.
   `python -m pytest tests/ -v`

**Rules:**
- Never skip tests without a documented scientific reason.
- Never weaken assertions to hide a bug.
- If a test fails, diagnose whether the implementation, the test, or the scientific assumption is wrong.
