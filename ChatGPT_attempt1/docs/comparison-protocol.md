# Two-model comparison protocol

## Preserve the evidence

The supplied revised specification is in `docs/specification.md`. Preserve the original plan and the prompt that requested its critique from the student's conversation history. The supplied critique is copied unchanged to `docs/critique-original.txt`; its reported label is **4 Argon High Effort**, with provider/interface pending confirmation.

Keep each model's first attempt and correction attempt separately, for example:

```text
model-submissions/
  model-a/first/
  model-a/corrected/
  model-b/first/
  model-b/corrected/
```

Record actual displayed model labels, generation dates, interface, reasoning setting, access to browsing/execution, prompts, and complete outputs. Preserve the raw first output before modifying it. The current candidate was generated in a continuing Codex conversation with planning context and parallel agent assistance; document that condition rather than claiming a blind fresh-session experiment. Give the other candidate the same frozen specification and record any differences in access or context.

## Apply one unchanged external suite

Use the same external evaluator for both candidates. Import each submitted `physics.py` without rewriting its numerical solution or analytical reference. The test inputs and independent reference code must remain unchanged between candidates. Record the evaluator's version or file hash.

Evaluate: initial and boundary conditions; steady behavior; error at the four required dimensionless times; grid convergence; equal-kinematic-viscosity histories; density, viscosity, and gap time scaling; invalid-input and workload behavior; and the same manual interface checklist. The external reference must not import either candidate's analytical function. Keep candidate self-tests distinguishable from the external results.

Physics correctness is required before selecting the public demo. Compare explanatory clarity and usability after checking correctness. Do not combine missing evidence with zero scores; mark it Pending.

## One correction round per candidate

Send each candidate one consolidated prompt containing:

1. Observed test failure outputs.
2. Interface problems observed using the same checklist.
3. The specification requirements that were violated.

Do not supply code patches or give one model substantially more diagnostic help. Preserve both feedback prompts and correction outputs. Reevaluate with the original external suite. Any further changes for production must be labeled separately from the scored correction round. If an essential failure remains, record it honestly before repairing the published app.

## Critique decisions

The student explicitly rejected stress readouts to avoid extra complexity and chose conservative teaching cases. The revised specification also includes several critique suggestions. For those, the student should add their own accept/reject reasoning rather than adopt an assistant-written explanation as their own.

| Suggestion | Revised specification treatment | Decision evidence |
|---|---|---|
| E1: flow assumptions and exploration limits | Laminar unidirectional assumptions; bounded teaching cases; no universal transition threshold | Conservative cases confirmed; detailed rationale pending |
| E2: stress readouts | Rejected; stress remains explanatory text | Student: avoid extra complexity for transient and steady readouts |
| E3: forces before PDE; one visualization | Included | Student rationale pending |
| E4: analytical series and resolved test window | Included: 200 terms and specified positive times | Student rationale pending |
| E5: computational bounds | Included: 51 nodes, Fo=0.4, scaled duration, workload cap | Student rationale pending |
| E6: solver signature and correction protocol | Included | Student chose one correction round; further rationale pending |
| E7: transcript display and processing time | Included | Student rationale pending |
| O1–O4: sign, time distinction, simple cylinder, visible evidence | Included | Student rationale pending |

## Interpretation

Compare these documented attempts and their response to feedback. Do not conclude that one model is universally better. State differences in tool access, continued planning context, or agent assistance that affect fairness. Only the student's genuine critique decisions, observed results, and selection reasons should appear as completed evidence.
