# Result-change records

A PR that changes the baseline results (`qa/baselines/`) by even one bit appends a numbered record to this directory ([QA-04](../quality-plan.md)). `tools/qa/results.py accept --record NNNN` verifies that the record file exists and mentions the case before it permits the update.

File names are `NNNN-<slug>.md` (numbered consecutively from `0001`). Once a record is created, its contents are not overwritten; a correction is added as a record with a new number.

## Format

```markdown
# NNNN: What changed (one sentence)

- Date: YYYY-MM-DD
- Issue: #N
- Pull request: #N
- Cause category: bug fix | algorithm change | precision handling | dependency update | compiler, flags, instruction set
- Specification IDs: DES-XX, ...
- Approval: (owner's name and date)

## Cause

What was changed, and why the result changes.

## Impact

| Case | Platform | Quantity | Before | After | Relative change |

## Tolerance comparison and validity against the numerical budget

## Impact on shipped versions and existing models

## What was verified
```

## Records

No result-change records are registered yet. The first platform qualification (initial registration of `numeric-canary`) is treated as creation of a new baseline.
