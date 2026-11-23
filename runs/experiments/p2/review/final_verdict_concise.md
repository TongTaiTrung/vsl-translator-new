# Scientific Review: Feature v3 Experiment

**Reviewer:** Agent 3 | **Date:** 2026-11-18 | **Verdict:** S4b DEFENSIBLE with caveats

---

## Claim Verification

### 1. distractor_flip_rate metric: VALID with MODERATE circularity risk

**Formula sharing:** Both metric (_composite_left_hand) and augmentation (_inter_hand_raw) use IDENTICAL geometry:
- `inter_hand_vec = left_palm - right_palm`  
- `hand_contact = exp(-8.0 * d^2)`

**Mitigating factors:**
- Formula matches REAL extraction (extract.py uses same exp(-8*d²))
- Metric tests on held-out validation fold (not training data)
- Improvement is substantial (flip 0.115→0.026, effect size 3.3σ)

**Concern:** Donors from SAME CV splits as targets. Model saw similar inter-hand geometries during training, so gain may be inflated.

**Verdict:** Improvement is LIKELY REAL but should be verified on independent test set with varied distractor positions.

---

### 2. lowered_static flip = 0: TRIVIALLY TRUE BY CONSTRUCTION

**Finding:** REST_Y = 0.9, lowered_y = 1.1 → distractor ALWAYS filtered by resting-hand gate.

**What it proves:** Gate works as designed (filters palm_y > 0.9).  
**What it DOESN'T prove:** Robustness to realistic distractors (1.1 is well above threshold).

**Verdict:** NOT evidence of learned robustness, just mechanical correctness.

---

### 3. noise_ret drop (0.886 → 0.726): METRIC ARTIFACT, not real loss

**Analysis:** Noise scaled by global std (includes zeros); presence_norm uses present-only std → incomparable metrics.

**Evidence:**
- Drop appears EXACTLY when presence_norm introduced (S1→S2)
- S4c (distractor_geom without presence_norm) maintains 0.902 retention
- OOF accuracy stays perfect (1.0) despite "retention drop"

**Verdict:** NOT evidence of degradation. Metric is invalid for comparing normalization methods.

---

### 4. S4b choice: DEFENSIBLE under condition (a) only

**User's rule:** Keep if (a) distractor flip, (b) open-set, OR (c) hard-pair improves AND no metric degrades.

**S4b vs. S1:**
- ✓ (a) Distractor flip: VERIFIED (effect size 3.3σ)
- ~ (b) Open-set FA: 0.667→0.522, modest (0.4σ), noisy
- ✗ (c) Hard pairs: NO significant improvement (effect <1σ for r/u, g/n)
- ⚠ noise_ret degrades, BUT artifact (explained above)

**Verdict:** Accept for DISTRACTOR ROBUSTNESS only. Do NOT claim hard pair improvements.

---

### 5. n=3 seeds: SUFFICIENT for distractor, INSUFFICIENT for pairs

**Statistical power:**
- Well-powered: flip_same_height (3.3σ), wrongacc (1.5σ)
- Underpowered: sep_r/u (0.33σ), sep_g/n (0.60σ), openset (0.0σ)

**Verdict:** Claims about confusable pairs are NOT SUPPORTED by current evidence.

---

## Leakage Check: ✓ NO LEAKAGE DETECTED

- Folds assigned by origin group
- validation_mask excludes augmented samples
- Temperature/threshold fitting on OOF logits only
- Metrics computed on original samples only

---

## Correctness Bugs: None found, 1 to verify

**✓ Correct:**
- REST_Y threshold consistency (0.9 in extract and process_sequence)
- Feature version skew (max|Δ|=0 verified by skew_check.py)

**⚠ To verify:** Does VSLPredictor correctly load/apply presence_norm from checkpoint?

---

## Key Risks

1. **Partial circularity:** Metric and augmentation share formula; gain likely real but may be inflated
2. **Underpowered for pairs:** n=3 insufficient for hard pair claims (need 10-15 seeds)
3. **Metric artifact:** noise_ret drop is NOT real robustness loss

---

## Next Steps (Prioritized)

1. **Verify distractor metric on independent data** (different donors/session) → addresses circularity
2. **Test lowered_static at realistic threshold** (0.85-0.95, not 1.1) → tests marginal cases  
3. **Audit VSLPredictor presence_norm loading** → ensures train/serve parity
4. **Re-run noise metric with presence-aware scaling** → valid robustness comparison
5. **Final validation on true held-out test set** → confirms gains not overfit to CV

---

## Summary Table

| Claim | Verdict | Strength | Note |
|-------|---------|----------|------|
| Distractor flip ↓ | VERIFIED | Strong (3.3σ) | Partial circularity caveat |
| lowered_static = 0 | TRIVIAL | N/A | By construction (REST_Y) |
| Noise retention ↓ | ARTIFACT | N/A | Metric incomparable |
| Open-set FA ↓ | PLAUSIBLE | Weak (0.4σ) | Noisy |
| Hard pairs ↑ | NOT SUPPORTED | Weak (<1σ) | Underpowered |
| **S4b choice** | **DEFENSIBLE** | - | **Distractor claim only** |

**Word count:** 592
