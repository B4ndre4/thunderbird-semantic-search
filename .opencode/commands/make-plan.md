---
description: Plan the implementation just discussed
agent: plan
---

# Task
Generate a sufficiently detailed implementation plan that can be executed by an LLM model less capable than the one generating the plan.

## Process

1. Review the changes to be made to code/documentation that were just discussed
2. Clearly detail the points where code needs to be modified or inserted and the changes or insertions to be made
3. Detail the comments to be inserted in the code according to AGENTS.md guidelines
4. Highlight any particularly complex steps so that the implementing agent pays maximum attention
5. If possible, also plan tests that the LLM agent can perform for verification

## Rules

- Plan the implementation of only the discussed changes and nothing else
- If during planning you encounter gaps or need clarifications, ask the user without assuming anything
- Do not add dependencies unless they are part of the discussed changes
- Include small code examples for the implementing agent following any rules present in AGENTS.md


## After Completion

Report:
- What has been planned
- Files to modify or create
- Any planned tests
