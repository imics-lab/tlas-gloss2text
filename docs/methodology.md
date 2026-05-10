# TLAS: Temporal-Linguistic Adaptive Streaming — Methodology

## 1. Introduction and Motivation

### 1.1 Problem Statement

Sign language translation systems must convert continuous streams of sign language glosses into natural language text. In real-world deployment, glosses arrive one at a time from a vision module, each with a timestamp reflecting when the sign was recognized. The system must decide *when* to translate accumulated glosses — too early produces fragmented, low-quality output; too late increases latency and degrades the user experience.

This is the **streaming segmentation problem**: given a continuous gloss stream without explicit sentence boundaries, determine optimal translation points that balance quality against latency.

### 1.2 Key Insight: Temporal Rhythm as a Signal

Existing streaming translation policies — Wait-k (Ma et al., 2019), TransLLaMa (Agostinelli et al., 2023) — make segmentation decisions based solely on linguistic content. They treat the input as a sequence of tokens and ignore *when* each token arrives.

In real sign language, **temporal rhythm carries grammatical information**. Within a sentence, signs arrive at a steady cadence (approximately 300–650ms per sign). Between sentences, signers pause for 2–7 seconds to mark boundaries, topic shifts, and emphasis. These pauses are not arbitrary — they are part of the grammar of signed languages.

TLAS is the first streaming translation policy to exploit this temporal signal. By fusing temporal pause detection with linguistic readiness estimation, TLAS achieves near-oracle segmentation quality on continuous streams.

### 1.3 Relation to Prior Work

| Method | Signal Used | Segmentation Mechanism | Limitations |
|--------|------------|----------------------|-------------|
| Wait-k (Ma et al., 2019) | None (fixed) | Translate every *k* tokens | Rigid; fragments sentences at arbitrary boundaries |
| TransLLaMa (Agostinelli et al., 2023) | Linguistic | Model outputs `<WAIT>` or translation | One API/model call per step; no temporal signal |
| **TLAS (this work)** | Temporal + Linguistic | Adaptive fusion of pause detection and readiness estimation | Requires timestamped input |

---

## 2. Architecture Overview

TLAS consists of three components that operate on each incoming gloss:

```
Continuous Gloss Stream (with timestamps)
         |
         v
+-------------------------+
| Temporal Pause Detector |  monitors inter-gloss timing
| (TPD)                   |  outputs pause_score in [0,1]
+------------+------------+
             |
             v
+-------------------------+
| Adaptive Fusion Gate    |  combines TPD + LRE signals
| (AFG)                   |  outputs READ / WRITE decision
+------------+------------+
             |
             v
+-------------------------+
| Linguistic Readiness    |  neural head on T5 encoder hidden states
| Estimator (LRE)         |  outputs readiness_score in [0,1]
+------------+------------+
             |
         WRITE --> Translation Backend (T5 / LLM API)
                          |
                          v
                   English text output
```

The complete per-gloss decision pipeline:

1. A new gloss arrives at timestamp *t*. It is appended to the buffer.
2. **TPD** computes a `pause_score` based on the gap between *t* and the previous gloss arrival.
3. **LRE** computes a `readiness_score` based on the accumulated glosses in the buffer.
4. **AFG** fuses both scores and decides: READ (wait for more glosses) or WRITE (translate now).
5. On WRITE: the buffer is sent to the translation backend, the output is emitted, and the buffer is flushed.

---

## 3. Temporal Pause Detector (TPD)

### 3.1 Purpose

The TPD monitors the rhythm of incoming glosses and produces a score reflecting whether the current gap between signs is unusually long — indicating a sentence boundary.

### 3.2 Algorithm

The TPD maintains an Exponential Moving Average (EMA) of inter-gloss intervals and computes a normalized pause score.

**State variables:**
- `ema_delta`: running average of inter-gloss gaps (initialized to `ema_prior / 1000` seconds)
- `last_timestamp`: timestamp of the most recent gloss

**On each new gloss at timestamp *t*:**

```
if first gloss:
    last_timestamp ← t
    return pause_score = 0.0

Δt ← t − last_timestamp
last_timestamp ← t

ema_delta ← α · Δt + (1 − α) · ema_delta          (1)

ratio ← Δt / (ema_delta + ε)                        (2)

pause_score ← clamp((ratio − 1) / (M − 1), 0, 1)   (3)
```

where:
- α is the EMA smoothing factor (`tpd_alpha`, default 0.3)
- M is the pause multiplier (`tpd_pause_multiplier`, default 2.5)
- ε = 10⁻⁹ for numerical stability

### 3.3 Interpretation

Equation (3) maps the normalized gap ratio to [0, 1]:
- When Δt = ema_delta (ratio = 1): pause_score = 0 — normal within-sentence pace
- When Δt = M × ema_delta (ratio = M): pause_score = 1 — definite sentence boundary
- Linear interpolation between these extremes

With default parameters and a typical within-sentence EMA of ~450ms:
- pause_score = 0.5 occurs at Δt ≈ 788ms
- pause_score = 1.0 occurs at Δt ≈ 1125ms

### 3.4 Adaptive Properties

The EMA adapts to individual signer pace:
- Fast signer (200ms average): pause trigger at ~500ms
- Slow signer (600ms average): pause trigger at ~1500ms

This eliminates the need for per-signer calibration.

### 3.5 Cold-Start Handling

The EMA is initialized with a prior of 450ms (configurable via `tpd_ema_prior_ms`), representing the average within-sentence inter-gloss interval observed in ASL data. This prevents spurious pause detections during the first few glosses, when the EMA has insufficient history.

### 3.6 Proactive Timeout

In continuous stream evaluation, the TPD supports a **proactive timeout** mechanism. Between processing steps, the system checks whether the time elapsed since the last gloss exceeds the adaptive threshold:

```
if gap > ema_delta × M:
    translate buffer immediately (before next gloss arrives)
    flush buffer
```

This simulates real-time system behavior where idle time triggers translation, rather than waiting for the next gloss to arrive and then detecting the pause retroactively. The proactive timeout is what enables TLAS to achieve clean sentence segmentation — the buffer is translated and flushed *during* the inter-sentence pause, so the next sentence starts with an empty buffer.

### 3.7 Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `tpd_alpha` | α | 0.3 | EMA smoothing factor (0 = full history, 1 = current only) |
| `tpd_pause_multiplier` | M | 2.5 | Gap ratio for full pause signal |
| `tpd_ema_prior_ms` | — | 450.0 | EMA initialization in milliseconds |

---

## 4. Linguistic Readiness Estimator (LRE)

### 4.1 Purpose

The LRE estimates whether accumulated glosses form a semantically complete, translation-ready unit. It outputs a `readiness_score` in [0, 1], where 0 means "clearly incomplete" and 1 means "ready to translate."

### 4.2 Architecture

The LRE is a small MLP head attached to the frozen T5 encoder:

```
Input: T5 encoder hidden states  h ∈ ℝ^{L×768}
       Attention mask             m ∈ {0,1}^L

Step 1 — Mean pooling (attention-weighted):
    p = Σᵢ (hᵢ · mᵢ) / Σᵢ mᵢ         p ∈ ℝ^768

Step 2 — Hidden layer:
    z = ReLU(W₁ p + b₁)                W₁ ∈ ℝ^{256×768}
    z = Dropout(z, p=0.1)              z ∈ ℝ^256

Step 3 — Output:
    readiness = σ(w₂ᵀ z + b₂)         w₂ ∈ ℝ^256, readiness ∈ [0,1]
```

Total trainable parameters: 768 × 256 + 256 + 256 × 1 + 1 = 197,121 (~773 KB).

### 4.3 Training with Semantic Oracle Labels

The LRE head is trained to predict **actual translation quality** rather than surface heuristics like token counts or position ratios. The training procedure:

1. **Oracle label generation**: For each training sentence with *n* glosses and reference translation *r*:
   - Construct gloss prefixes: *g₁*, *g₁ g₂*, ..., *g₁ ... gₙ*
   - Translate each prefix using the frozen T5 model: *ŷᵢ* = T5(*g₁ ... gᵢ*)
   - Score each translation: *sᵢ* = ROUGE-L(*ŷᵢ*, *r*)
   - Enforce monotonicity: *sᵢ* ← max(*sᵢ*, *sᵢ₋₁*)

2. **Training**: The LRE head is trained with MSE loss to predict the oracle scores:
   - Loss = (1/N) Σᵢ (LRE(encoder(*g₁...gᵢ*)) − *sᵢ*)²
   - Optimizer: AdamW, lr = 10⁻³
   - The T5 encoder is frozen throughout; only the MLP head is updated

The semantic oracle approach teaches the LRE to recognize when accumulated glosses carry enough meaning for a high-quality translation, rather than relying on proxy signals.

### 4.4 Standalone Scorer for Non-T5 Backends

When using API-based translation backends (Gemini, Ollama, Groq), the LRE still operates locally:

1. The T5 encoder and LRE head are loaded as a standalone module (~5ms per call)
2. Glosses are tokenized and encoded through the T5 encoder
3. The LRE head produces the readiness score
4. No API calls are made for readiness estimation

This design avoids expensive per-gloss API calls while maintaining consistent readiness scoring across all backends. The standalone scorer is loaded as a module-level singleton, shared across all policy instances.

### 4.5 Fallback Behavior

If no trained LRE head is found, the system degrades gracefully:
- Returns 0.5 (neutral readiness) for all inputs
- The AFG still operates, relying on the TPD signal and the max_lag safety valve

### 4.6 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lre_hidden_dim` | 256 | Hidden layer dimension |
| `lre_dropout` | 0.1 | Dropout probability |
| `lre_learning_rate` | 10⁻³ | Training learning rate |
| `lre_epochs` | 5 | Training epochs |
| `lre_batch_size` | 64 | Training batch size |

---

## 5. Adaptive Fusion Gate (AFG)

### 5.1 Purpose

The AFG combines the TPD and LRE signals to produce a binary READ/WRITE decision.

### 5.2 Decision Logic

The AFG evaluates four conditions in priority order (first match wins):

**Condition 1 — Final gloss:**
If `is_final = True`, always WRITE (end of stream).

**Condition 2 — Safety valve (max lag):**
If `buffer_length ≥ max_lag` (default 6), force WRITE.
This prevents indefinite buffering regardless of signal values.

**Condition 3 — Joint threshold:**
Compute the combined score:
```
combined = w_t · pause_score + w_l · readiness_score     (4)
```
If `combined ≥ θ`, WRITE.

With default weights (w_t = 0.4, w_l = 0.6) and threshold (θ = 0.40), this triggers when:
- readiness ≥ 0.67 alone (no pause needed)
- pause = 1.0 and readiness ≥ 0.0 (strong pause suffices)
- pause = 0.5 and readiness ≥ 0.33 (balanced)

**Condition 4 — Strong pause override:**
If `pause_score ≥ 0.8` AND `readiness_score ≥ 0.3`, WRITE.
This trusts a strong temporal signal even when the linguistic signal is moderate. It ensures that clear sentence boundaries detected by the TPD are respected, provided the accumulated glosses have at least minimal translation potential.

**Default: READ.**

### 5.3 Design Rationale

The asymmetric weighting (w_l = 0.6 > w_t = 0.4) reflects that linguistic readiness is generally more informative for translation quality, while the temporal signal provides boundary information. The strong pause override (Condition 4) ensures that the temporal signal is not completely overwhelmed by a low readiness score at genuine sentence boundaries.

The four-condition priority structure avoids complex interactions: safety conditions (final, max_lag) take precedence, then the joint criterion, then the temporal override. The READ default ensures the system buffers when uncertain.

### 5.4 Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `afg_weight_temporal` | w_t | 0.4 | Weight for pause_score in Eq. (4) |
| `afg_weight_linguistic` | w_l | 0.6 | Weight for readiness_score in Eq. (4) |
| `afg_threshold` | θ | 0.40 | Combined score threshold for WRITE |
| `afg_strong_pause_threshold` | — | 0.8 | Pause value for override condition |
| `afg_min_readiness_for_pause` | — | 0.3 | Minimum readiness for strong pause override |
| `afg_max_lag` | — | 6 | Maximum glosses before forced WRITE |

---

## 6. TLAS Ablation Modes

TLAS supports three operational modes to isolate the contribution of each component:

| Mode | w_t | w_l | Active Components | Name |
|------|-----|-----|-------------------|------|
| **FULL** | 0.4 | 0.6 | TPD + LRE + AFG | TLAS |
| **TEMPORAL_ONLY** | 1.0 | 0.0 | TPD + AFG (readiness always 0) | TLAS-temporal |
| **LINGUISTIC_ONLY** | 0.0 | 1.0 | LRE + AFG (pause always 0) | TLAS-linguistic |

In TEMPORAL_ONLY mode, the LRE is never called; the AFG reduces to thresholding the pause_score alone. In LINGUISTIC_ONLY mode, the TPD runs but contributes zero weight; the AFG thresholds the readiness_score alone.

These ablations serve to quantify:
1. The individual contribution of temporal vs. linguistic signals
2. The benefit of fusion over single-signal policies
3. Whether the trained LRE improves over raw temporal detection

---

## 7. Translation Backends

All backends implement the same `TranslationBackend` protocol, enabling any policy to work with any backend:

```python
class TranslationBackend(Protocol):
    async def translate(glosses: str, context: str = "", ...) -> TranslationResult
    async def get_readiness_score(glosses: str) -> float
    async def wait_or_translate(glosses: str) -> str
```

### 7.1 T5 Backend

- **Model**: T5-base (250M parameters), fine-tuned on ASLG-PC12 + SIGNUM + synthetic discourse pairs
- **Task prefix**: `"translate ASL to English: "` prepended to all inputs
- **Special tokens**: `<WAIT>` token (ID 32100) added during fine-tuning for TransLLaMa-style training
- **Decoding**: Beam search (num_beams = 4, no_repeat_ngram_size = 2, early_stopping)
- **LRE integration**: Native — the LRE head directly accesses the T5 encoder's hidden states
- **Async execution**: CPU-bound inference runs in a `ThreadPoolExecutor` with `max_workers=1`

### 7.2 Gemini Backend

- **Model**: Configurable via `GEMINI_MODEL` environment variable (default: gemini-2.5-flash)
- **LRE integration**: Standalone T5 encoder + LRE head loaded locally (~5ms per call)
- **Rate limiting**: Configurable delay between API calls (`api_delay`), exponential backoff on failure
- **TransLLaMa support**: WAIT-or-translate prompt (one API call per step)

### 7.3 Ollama Backend

- **Model**: Configurable via `OLLAMA_MODEL` (default: gpt-oss-32k)
- **Endpoint**: Local HTTP API at `localhost:11434`
- **LRE integration**: Same standalone scorer as Gemini
- **Session management**: Persistent `aiohttp.ClientSession` across calls

### 7.4 Context Handling

All backends support discourse context via a sliding window of prior translations:

```
Source = "translate ASL to English: [Context: sent₁ ||| sent₂ ||| sent₃] CURRENT GLOSSES"
```

The context window size is configurable (`context_window_size`, default 3). In evaluation, only TLAS variants receive discourse context; baselines receive no context (`context_window = 0`) to isolate the segmentation effect from the context effect.

---

## 8. Streaming Pipeline

### 8.1 Architecture

The `StreamingPipeline` orchestrates the interaction between policy and backend:

1. Receives gloss events (tokens with timestamps)
2. Feeds each event to the policy's `step()` method
3. On WRITE decisions, sends the buffer to the backend for translation
4. Maintains a discourse context ring buffer of prior translations
5. Records per-segment metrics (scores, timestamps, glosses)

### 8.2 Operation Modes

**Sentence-list mode** (`run_sentence_list`): Processes pre-segmented sentences with synthetic timestamps drawn from N(450ms, 100ms). Used for E1 evaluation.

**Stream-file mode** (`run_stream_file`): Replays timestamped stream files in real time (or accelerated). Used for E3 qualitative demos.

**Discourse stream mode** (in `evaluate.py`): Feeds entire discourse groups as continuous streams with real per-gloss timestamps. Used for E2 evaluation. Includes proactive timeout simulation.

### 8.3 Translation Pipeline

On each WRITE decision:
1. Collect buffered glosses and timestamps
2. Build context string from prior translations (if context window > 0)
3. Call `backend.translate(glosses, context=context_str)`
4. Record the translation as a `SegmentResult` with all metadata
5. Update the context window
6. Flush the policy buffer

---

## 9. T5 Fine-Tuning

### 9.1 Training Data

The T5 model is fine-tuned from a fresh `t5-base` checkpoint on four types of training pairs:

**Type 1 — Standard bidirectional pairs:** For each (gloss, text) pair from ASLG-PC12 and SIGNUM, create both gloss→text and text→gloss examples with the appropriate task prefix.

**Type 2 — TransLLaMa-style streaming examples:** For sentences with ≥3 glosses, create partial-input examples:
- Early prefix (first 1/3 of glosses) → target `<WAIT>`
- Mid prefix (first 1/2 of glosses) → target is partial translation or `<WAIT>`
- Full sentence → target is complete translation (already covered by Type 1)

This teaches the model to output `<WAIT>` for incomplete inputs, enabling the TransLLaMa baseline to use the T5 backend directly.

**Type 3 — Discourse context pairs:** From synthetic discourse groups, create context-augmented training pairs using a sliding window:
```
Source: "translate ASL to English: [Context: prior₁ ||| prior₂] CURRENT_GLOSSES"
Target: "current translation"
```

Only deaf turns with glosses become targets; hearing turns and prior deaf translations contribute context only.

**Type 4 — SIGNUM augmentation:** 779 German Sign Language pairs from the SIGNUM corpus, providing cross-domain vocabulary exposure.

### 9.2 Training Configuration

| Parameter | Value | Effective |
|-----------|-------|-----------|
| Base model | t5-base | 250M parameters |
| Epochs | 5 | — |
| Per-device batch size | 8 | — |
| Gradient accumulation | 4 | Effective batch = 32 |
| Learning rate | 5 × 10⁻⁵ | With linear warmup (10% of steps) |
| Weight decay | 0.01 | L2 regularization |
| Label smoothing | 0.1 | Smoothed cross-entropy |
| Precision | FP16 | Mixed precision on CUDA |
| Best model selection | eval_loss | Restore best checkpoint at end |

### 9.3 Special Token Handling

The `<WAIT>` token is registered as a special token in the tokenizer (token ID 32100). The model's embedding matrix is resized to accommodate it. During TransLLaMa training, the model learns to output `<WAIT>` for incomplete gloss prefixes, creating a learned boundary between "wait" and "translate" behaviors.

---

## 10. Synthetic Discourse Data Generation

### 10.1 Motivation

Real multi-sentence ASL discourse data with parallel English translations does not exist at scale. We generate synthetic discourse groups using an LLM (Gemini) with careful constraints to ensure linguistic validity.

### 10.2 Discourse Group Types

| Type | Structure | Proportion |
|------|-----------|------------|
| Monologue | 5 connected sentences from one deaf signer | ~40% |
| Deaf-deaf dialog | 5 sentences alternating between two deaf signers | ~30% |
| Deaf-hearing dialog | 6 turns alternating between a deaf and hearing speaker | ~30% |

### 10.3 Vocabulary Constraints

Generated glosses are constrained to an attested ASL vocabulary extracted from ASLG-PC12 and SIGNUM:
- Gloss roots are extracted by decomposing compounds (SIGN+SIGN), directional verbs (i-TELL-you → TELL), and spatial references (COME-here → COME)
- The top 400 ASLG roots + all SIGNUM roots are provided to the LLM as the allowed vocabulary
- Generated groups are validated for vocabulary coverage: rejected if average gloss root coverage < 80%

### 10.4 ASL Morphology Rules

The generation prompt enforces ASL-specific notation conventions:
- All signs in UPPERCASE
- Topic-comment word order (not English SVO)
- Directional verbs with lowercase pronoun affixes: `i-TELL-you`, `he-COME-middle`
- Spatial references with lowercase direction suffixes: `COME-here`, `STAY-right`
- Compounds with `+`: `EXCELLENT+MARKET` (supermarket)
- Modifiers: `STRONG-much`, `WE-all`, `BIG-wide`, `A-LOT-OF`
- Questions with wh-word last: `NAME YOUR WHAT?`

### 10.5 Timestamp Annotation

Each discourse group is annotated with per-gloss timestamps reflecting realistic signing rhythm:
- **Within-sentence gaps**: 300–650ms (short signs: 300–400ms; complex/directional: 500–650ms)
- **Between-sentence gaps**: 1500–7000ms depending on discourse context:
  - Quick question→answer: 1500–2500ms
  - Statement → related follow-up: 2500–3500ms
  - Topic shift: 4500–7000ms
  - Speaker transition: 1500–3000ms

Timestamps are generated via Gemini with a detailed prompt including real stream data examples, then validated for monotonicity and plausible interval ranges.

### 10.6 Dataset Statistics

- **Total**: 1,400 discourse groups
- **Split**: Lines 1–200 = test set; lines 201–1400 = training set
- **Training pairs**: ~5,234 genuine discourse context pairs from the training split
- **Test sentences**: 888 deaf sentences across 200 groups (average 4.4 sentences per group)

---

## 11. Comparison with Baseline Methods

### 11.1 Batch (Oracle Upper Bound)

Translates each ground-truth sentence independently with full context. This is the ceiling that streaming methods approach. Not a streaming method — requires knowing sentence boundaries in advance.

### 11.2 Wait-k (Ma et al., 2019)

Accumulates exactly *k* glosses, then translates. Non-overlapping windows: after translating, the buffer is flushed and the next *k* glosses are accumulated.

**Limitations in continuous streams**: With k=3, a 7-gloss sentence produces 2–3 fragments. The translation model sees each fragment independently, producing incoherent partial translations. Fragments that cross sentence boundaries are particularly harmful.

### 11.3 TransLLaMa (Agostinelli et al., 2023)

At each step, asks the translation model: "Should I wait or translate?" The model outputs either `<WAIT>` or a translation.

**T5 implementation**: The fine-tuned T5 model has been trained to output the `<WAIT>` token for incomplete prefixes. At each step, we generate from the current buffer and check if the output contains `<WAIT>`.

**API implementation**: A WAIT-or-translate prompt is sent at each step (one API call per gloss, plus one for the actual translation on WRITE). This faithfully replicates the TransLLaMa mechanism.

**Limitations**: No temporal signal; requires one model/API call per step for the decision; the model may not learn reliable WAIT/translate boundaries from limited training data.

### 11.4 LSG — Local Scoring Gate (Ablation)

LSG is our own prior approach before developing the trained LRE head. It uses raw T5 encoder statistics for segmentation decisions:

**T5 path**: Computes KL divergence between the current next-token probability distribution and a baseline distribution (from the first *k* glosses). WRITE if KL > threshold or max probability > confidence threshold.

**API path**: Falls back to the standalone LRE scorer (no API calls for readiness).

LSG is included as an ablation to quantify the improvement from the learned LRE over naive encoder heuristics. It is **not** a published external baseline.

---

## 12. Limitations

### 12.1 Dependence on Timestamped Input

TLAS requires gloss timestamps, which are naturally produced by vision-based sign recognition but not available in text-only gloss corpora. In our evaluation, we use LLM-generated timestamps calibrated to real ASL timing patterns. The quality of the temporal signal depends on the vision module's ability to provide accurate recognition timestamps.

### 12.2 LRE Training Data

The LRE head is trained on oracle labels derived from the same T5 model that performs the final translation. This creates a potential circularity: the LRE learns what the T5 model can translate well, not necessarily what constitutes a linguistically complete unit. Using an independent quality estimator (e.g., a separate reference-free metric) could mitigate this.

### 12.3 Fixed AFG Weights

The fusion weights (w_t, w_l) and threshold (θ) are currently hand-tuned. Reinforcement learning could optimize these parameters jointly, using translation quality minus latency penalty as the reward signal. This is planned as future work (GRPO or PPO optimization).

### 12.4 Context Window Limitation

The discourse context is implemented as a fixed sliding window of prior translations. This approach cannot capture long-range dependencies beyond the window size. A persistent state mechanism (e.g., Mamba's recurrent state) could provide implicit context propagation without explicit windowing.

### 12.5 Uniform Timestamp Regime

In sentence-level evaluation (E1), uniform synthetic timestamps (~450ms) neutralize the TPD because all gaps are similar. The pause_score remains near 0, and TLAS degenerates to max_lag triggering (translate every 6 glosses). This is by design — E1 measures linguistic-only performance, while E2 (continuous stream with realistic timing) is the primary evaluation paradigm.

---

## 13. Complete Hyperparameter Reference

### TPD (Temporal Pause Detector)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `tpd_alpha` | 0.3 | [0, 1] | EMA smoothing factor |
| `tpd_pause_multiplier` | 2.5 | > 1 | Ratio threshold for full pause signal |
| `tpd_ema_prior_ms` | 450.0 | > 0 | EMA initialization (ms) |

### LRE (Linguistic Readiness Estimator)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lre_hidden_dim` | 256 | > 0 | MLP hidden layer dimension |
| `lre_dropout` | 0.1 | [0, 1] | Dropout probability |
| `lre_learning_rate` | 10⁻³ | (0, 1) | Adam learning rate |
| `lre_epochs` | 5 | > 0 | Training epochs |
| `lre_batch_size` | 64 | > 0 | Training batch size |

### AFG (Adaptive Fusion Gate)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `afg_weight_temporal` | 0.4 | [0, 1] | Weight for pause_score |
| `afg_weight_linguistic` | 0.6 | [0, 1] | Weight for readiness_score |
| `afg_threshold` | 0.40 | [0, 1] | Combined score threshold |
| `afg_strong_pause_threshold` | 0.8 | [0, 1] | Strong pause override threshold |
| `afg_min_readiness_for_pause` | 0.3 | [0, 1] | Minimum readiness for pause override |
| `afg_max_lag` | 6 | > 0 | Maximum glosses before forced WRITE |

### T5 Fine-Tuning

| Parameter | Default | Description |
|-----------|---------|-------------|
| Base model | t5-base | 250M parameters, d_model=768 |
| Epochs | 5 | — |
| Batch size (effective) | 32 | 8 per-device × 4 accumulation |
| Learning rate | 5 × 10⁻⁵ | Linear warmup (10% of steps) |
| Weight decay | 0.01 | L2 regularization |
| Label smoothing | 0.1 | Smoothed cross-entropy |
| Max source length | 256 | Tokenizer truncation |
| Max target length | 128 | Generation max length |
| Beam search width | 4 | — |
| No-repeat n-gram | 2 | Diversity constraint |

### Streaming Pipeline

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_window_size` | 3 | Prior translations for discourse context |
| `simulated_intergloss_ms` | 450 | Mean synthetic inter-gloss gap (E1) |
| `simulated_intergloss_std_ms` | 100 | Std dev of synthetic inter-gloss gap (E1) |
| Wait-k `k` | 3 | Fixed buffer size for Wait-k baseline |
