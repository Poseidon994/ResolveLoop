# Orchestrator Routing & Learning System — Design Spec

This document specifies how the Orchestrator decides which tier (L1–L4) handles a
case, and how the system learns from experience to improve that decision over time.

It is written to map directly onto the existing `ExperienceStore` / PostgreSQL
schema (`experiences`, `learned_policies`, `feedback`, `audit_events`) described in
`OPERATIONS.md` and `README.md`, so it can be implemented as `router.py` /
`learning.py` without redesigning the data model.

---

## 1. Overview — Two Systems, Not One

| System | Runs | Purpose |
| :--- | :--- | :--- |
| **Routing Decision** | On every incoming case | Decide which tier handles this case, right now |
| **Learning / Promotion** | After a case completes, and periodically | Turn accumulated experience into reusable rules that make future routing decisions faster and better |

The routing decision function *consumes* what the learning system produces
(`procedural_memory` rules, `experience_store` similarity matches). Learning never
happens synchronously inside the routing path — it happens after outcomes are known.

---

## 2. Routing Decision — Cascade Logic

Routing is a cascade of checks, cheapest and most certain first. Each step can
short-circuit the rest. This ordering is *why* the system gets faster as it learns:
warm cases resolve at Step 1 or 2 and never reach the expensive LLM classification
at Step 3.

```python
def route(case: Case) -> RoutingDecision:
    # Step 0 — Hard safety overrides. Never learned away, never demoted.
    if case.amount is not None and case.amount >= ESC_400_THRESHOLD:   # e.g. $50,000
        return RoutingDecision(tier=Tier.L4, confidence=1.0,
                                reason="ESC-400 hard threshold")
    if contains_fraud_signal(case) or contains_legal_keyword(case):
        return RoutingDecision(tier=Tier.L4, confidence=1.0,
                                reason="risk override")

    # Step 1 — Procedural memory: learned rules
    rule = procedural_memory.match(extract_features(case))
    if rule and rule.confidence >= RULE_TRUST_THRESHOLD:        # e.g. 0.75
        if random() < EXPLORATION_RATE:                         # e.g. 0.05–0.10
            pass  # deliberately skip the rule this time -> fall through to Step 3
        else:
            return RoutingDecision(tier=rule.tier, confidence=rule.confidence,
                                    reason=f"learned rule {rule.id}",
                                    tool_sequence=rule.tool_sequence)

    # Step 2 — Episodic memory: nearest-neighbor case retrieval
    similar = experience_store.retrieve_similar(case, k=3)
    if similar and similar[0].similarity >= SIMILARITY_THRESHOLD:  # e.g. 0.80
        tier, confidence = weighted_vote(similar)
        return RoutingDecision(tier=tier, confidence=confidence,
                                reason="similar past cases",
                                tool_sequence=most_common_tool_sequence(similar))

    # Step 3 — Cold path: LLM / heuristic classification
    tier, confidence = llm_classify(case)   # gpt-5-nano estimates tier + confidence
    return RoutingDecision(tier=tier, confidence=confidence, reason="cold classification")
```

### Confidence bands (used at Step 3, when there is no memory to lean on)

| Confidence | Action |
| :--- | :--- |
| > 90% | Resolve at the tier the classifier suggests |
| 60–90% | Route one tier up for investigation before resolving |
| < 60% | Escalate directly (don't guess on a low-confidence case) |

Once memory exists, confidence is increasingly derived from **how many past cases
agreed**, not from a single LLM guess — this is the actual mechanism behind
"escalation accuracy improves over time."

### Feature extraction

`extract_features(case)` should produce a stable signature used for both rule
matching and similarity search, e.g.:

```python
{
  "problem_type": "short_payment",       # canonicalized category
  "entities": ["invoice", "payment"],    # detected entity types
  "amount_bucket": "<10k" | "10k-50k" | ">50k",
  "policy_hint": "SHORT-PAY-01" | None,
  "keywords": ["discount", "remittance"]
}
```

---

## 3. Learning — Case-Based Reasoning, Not Model Training

The system does not fine-tune anything. It gets smarter by accumulating and
**curating** a case library. This is deliberately simple to build, explain to
judges, and audit. Four parts:

### 3.1 Experience Recording (after every case)

```python
experience = {
    "case_id": case.id,
    "features": extract_features(case),
    "route_taken": tier,
    "tools_used": [...],
    "outcome": resolution_text,
    "score": evaluator.score(case, resolution),          # 0–100
    "was_escalated": bool,
    "could_lower_tier_have_solved_it": reflection.assess(case, resolution),
    "lesson": reflection.summarize_lesson(case, resolution),  # short, human-readable
    "confidence": 0.6,     # new experiences start at moderate confidence, not high
    "created_at": now()
}
experience_store.save(experience)   # -> `experiences` table
```

### 3.2 Rule Promotion — the actual "learning" step

A single successful case must **not** immediately become a rule — that overfits to
one lucky outcome. Promote only on repeated, consistent evidence:

```python
def maybe_promote_rule(feature_signature):
    matches = experience_store.get_by_features(feature_signature)

    if len(matches) >= MIN_SUPPORT:                 # e.g. 3 cases
        avg_score = mean(m.score for m in matches)
        tier_agreement = mode(m.route_taken for m in matches)
        agreement_ratio = fraction_matching(matches, tier_agreement)

        if avg_score >= PROMOTION_SCORE_THRESHOLD and agreement_ratio >= 0.8:
            procedural_memory.upsert_rule(
                signature=feature_signature,
                tier=tier_agreement,
                confidence=avg_score / 100,
                tool_sequence=most_common_tool_sequence(matches)
            )   # -> `learned_policies` table
```

Example from PS.md: `payment + expired_card → L1` only becomes a rule after several
consistent, high-scoring cases of that shape — not after the first one.

### 3.3 Feedback-Driven Confidence Adjustment (👍 / 👎 loop)

```python
def apply_feedback(experience_id, positive: bool):
    exp = experience_store.get(experience_id)
    exp.confidence += 0.05 if positive else -0.15   # asymmetric: punish failure harder
    exp.confidence = clip(exp.confidence, 0.0, 0.99)

    if exp.confidence < DEMOTE_THRESHOLD:            # e.g. 0.3
        procedural_memory.demote_or_retire(exp.rule_id)
        failure_memory.log(exp, reason="confidence collapsed")

    audit_events.log(experience_id, positive, new_confidence=exp.confidence)
```

The asymmetric penalty (drop faster than it rises) prevents the system from
calcifying around a bad generalization — it unlearns bad rules faster than it
trusts new ones.

### 3.4 Exploration Rate — why the system doesn't get stuck

Without occasionally routing a rule-matched case through full reasoning anyway
(`EXPLORATION_RATE` in Step 1 of routing), the system would never detect that a
rule has gone stale — e.g., a policy change redefines what counts as a short
payment, but the old rule keeps firing. Sending ~5–10% of rule-matched cases
through the cold path is what lets it **detect drift** and revise rules rather
than just accumulate them.

---

## 4. Why This Produces the V0 → V3 Improvement Curve

| Stage | Procedural memory state | Typical routing path | Expected effect |
| :--- | :--- | :--- | :--- |
| **V0** (0 cases) | Empty | Every case falls to Step 3 (LLM classify) | Baseline: most tool calls, most latency |
| **V1** (10 cases) | Sparse — few signatures have hit `MIN_SUPPORT` | Mostly Step 2 (similarity), occasional Step 1 | Slight improvement |
| **V2** (50 cases) | Several signatures promoted | Step 1 hits increase for common problem types | Noticeable drop in tool calls / escalation |
| **V3** (100 cases) | Most common problem types have rules | Majority resolved at Step 1; LLM only for novel cases | Largest resolution-rate gain, lowest latency |

This is the causal mechanism behind the "before vs after" numbers in PS.md Section
13 — the model itself isn't getting smarter; **fewer cases need the expensive
path** as the case library matures.

---

## 5. Tunable Constants (suggested starting values)

| Constant | Suggested value | Effect if raised | Effect if lowered |
| :--- | :--- | :--- | :--- |
| `ESC_400_THRESHOLD` | $50,000 | Fewer forced L4 escalations | More forced L4 escalations |
| `RULE_TRUST_THRESHOLD` | 0.75 | Rules used less often, more cold-path calls | Rules used more eagerly, risk of premature trust |
| `EXPLORATION_RATE` | 0.05–0.10 | Slower drift detection, cheaper on average | Faster drift detection, more expensive |
| `SIMILARITY_THRESHOLD` | 0.80 | Fewer Step-2 hits, more cold-path calls | More Step-2 hits, risk of bad analogies |
| `MIN_SUPPORT` | 3 | Slower rule formation, more conservative | Faster rule formation, higher overfit risk |
| `PROMOTION_SCORE_THRESHOLD` | 80 / 100 | Fewer, higher-quality rules | More rules, some lower-quality |
| `DEMOTE_THRESHOLD` | 0.3 confidence | Rules survive longer under negative feedback | Rules retired faster under negative feedback |

---

## 6. Suggested Function/Module Split

```
router.py
  - route(case) -> RoutingDecision
  - extract_features(case) -> dict
  - llm_classify(case) -> (tier, confidence)
  - weighted_vote(similar_experiences) -> (tier, confidence)

learning.py
  - record_experience(case, resolution, score) -> None
  - maybe_promote_rule(feature_signature) -> None
  - apply_feedback(experience_id, positive: bool) -> None
  - most_common_tool_sequence(experiences) -> list[str]

memory/
  experience_store.py   # `experiences` table — episodic memory, similarity search
  procedural_memory.py  # `learned_policies` table — promoted rules
  failure_memory.py     # demoted rules / low-confidence retirements, for audit
```

This keeps routing (read path) and learning (write path) cleanly separated, which
also makes it straightforward to run the V0→V3 benchmark: snapshot
`procedural_memory` + `experience_store` at each checkpoint, freeze them, and
re-run the same held-out eval set through `route()` against each snapshot.
