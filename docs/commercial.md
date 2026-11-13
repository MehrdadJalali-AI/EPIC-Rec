# Why this matters for a recommendation product

This note explains, for product, compliance and business readers, what EPIC-Rec offers a company that runs a
recommender system. It describes capabilities of the method; it does not claim measured business outcomes, which
would require an online study with the company's own system.

## 1. Remove one recommendation without destabilising the rest

Users increasingly expect to say "not this one" — a film they will not watch, a product they already own, a
sensitive topic. A blunt fix (hiding the item, retraining, or deleting large parts of the history) changes many
other recommendations as a side effect. EPIC-Rec computes the **smallest** change to the user's own input that
removes exactly the unwanted item **while keeping every other item of the list** (C1 + C2). The rest of the page
the user already liked stays where it was.

## 2. Keep the reasons for the remaining recommendations

Removing an item can silently remove the reason behind another item ("because you watched X"). With C3, every
recommendation that stays keeps at least one faithful evidence path, so the explanations shown next to the list
remain true after the change. This matters wherever explanations are displayed or audited.

## 3. Transparent, checkable explanations

Each answer is a concrete set of the user's own interactions, the new list, and the surviving evidence paths, and
it is re-verified by an independent recomputation before it is reported. Support staff and users can read it as:
"these past interactions are why the item appeared; without them it would not, and nothing else would change."

## 4. Safer user controls

Because the mask is restricted to the user's own interactions, a control built on EPIC-Rec cannot touch other
users' data or the catalogue. The guarantees (C1, C2, C3) are explicit and are checked for every answer, so
"remove this" controls behave predictably, and their effect can be previewed before it is applied.

## 5. Auditable changes

Every explanation records the request, the constraints, the solver status (optimal, feasible with a proven bound,
infeasible, or timed out), the mask, the new list and the verification report. This gives an audit trail for
questions such as "why was this shown?" or "what changed after the user's request?", which supports transparency
obligations for recommender systems (for example, the EU Digital Services Act's requirements on explaining the
main parameters of recommender systems). Whether a specific deployment meets a specific legal obligation is a
legal question outside the scope of this software.

## 6. Lower collateral impact on engagement and conversion

Collateral changes to a list are a cost: items the user would have clicked disappear. EPIC-Rec minimises the change
and measures its collateral impact directly (list overlap, rank correlation of the retained items, evidence
survival). These quantities make it possible to compare intervention policies offline before an A/B test; the
size of any effect on engagement or conversion has to be measured online.

## 7. Know when a recommendation cannot be changed in isolation

Sometimes no change to the user's own history can remove one item without also removing others or their evidence.
EPIC-Rec proves this (infeasible status) and reports why — for example, the unwanted item and a liked item rely on
the same past interactions. For a product team this is actionable information: such items need a different
control (an explicit "hide" rule, a catalogue-level fix, or a model change) rather than a history-based one.

## Limits

The guarantees hold for the path-additive model used here, which decomposes scores exactly into paths. Other
models need either a decomposition of the same kind or approximate attributions, and then only the verification
step — not the guarantees — carries over. Exact solving can time out on very long histories; the outcome is then
reported as such, and the approximate explainers provide verified, non-minimal alternatives.
