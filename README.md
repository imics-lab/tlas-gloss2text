# TLAS: Temporal-Linguistic Adaptive Streaming for Sign Language Translation

Real-time streaming translation from ASL gloss sequences to English text, using a novel policy that fuses **temporal pause detection** with **linguistic readiness estimation**.

## Key Idea

Existing streaming translation policies (Wait-k, TransLLaMa) make decisions based solely on linguistic content. In real sign language, **temporal rhythm carries grammatical information** — pauses between signs mark sentence boundaries, topic shifts, and emphasis. TLAS is the first streaming policy to exploit this signal.

### Architecture

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
| Linguistic Readiness    |  lightweight head on encoder hidden states
| Estimator (LRE)         |  outputs readiness_score in [0,1]
+------------+------------+
             |
         WRITE --> Translation Model (T5 / LLM API)
                          |
                          v
                   English text output
```

**TPD**: Monitors inter-gloss timing via an exponential moving average. Fires when the current gap significantly exceeds the running norm.

**LRE**: Small MLP head on the frozen T5 encoder that predicts whether accumulated glosses form a translatable unit. Trained with **semantic oracle labels**: each partial gloss prefix is translated by the frozen T5, scored against the reference via ROUGE-L, and the resulting score becomes the training target. This teaches the LRE actual translation quality rather than token counting. For non-T5 backends, the T5 encoder + LRE head are loaded locally as a standalone scorer (~5ms/call), avoiding any per-gloss API calls.

**AFG**: Weighted fusion of TPD and LRE scores with three WRITE triggers: (1) combined score exceeds threshold, (2) strong pause with moderate readiness, (3) safety valve after max lag.

## Quick Start

```bash
pip install -r requirements.txt

# Configure API keys (for LLM backends)
cp .env.example .env
# Edit .env with your Groq / Gemini / Ollama keys

# Fine-tune T5 with discourse context pairs (fresh from t5-base, saves models/t5_base_discourse/)
TLAS_CONFIG=config/discourse_fresh.yaml python -m src.training.train_t5 --discourse

# Train the LRE head with semantic oracle labels (~45 min on GPU)
TLAS_CONFIG=config/discourse_fresh.yaml python -m src.training.train_lre

# E2 — continuous stream discourse evaluation (primary experiment)
HF_HOME=/home/vangelis/Research/sign-language/.hf_cache \
TLAS_CONFIG=config/discourse_fresh.yaml \
  python -m src.evaluation.evaluate --dataset discourse --backend t5 --n 200 --sbert --chrf

# E2 with API backends (Gemini, Ollama)
HF_HOME=/home/vangelis/Research/sign-language/.hf_cache \
TLAS_CONFIG=config/discourse_fresh.yaml \
  python -m src.evaluation.evaluate --dataset discourse --backend gemini --n 200 --sbert --chrf

# Evaluate the discourse context contribution (with vs. without prior context)
python -m src.evaluation.evaluate_context --backend t5

# Live streaming demo with per-step score visualization
python -m src.evaluation.streaming_demo --policy compare

# Generate additional synthetic discourse data via LLM
python -m src.training.synthetic_data --mode discourse --n 500
```

## Setup on a New Machine

```bash
git clone <repo_url>
cd continuous-gloss2text
pip install -r requirements.txt

# Copy these files manually (not in git):
#   .env                                   — API keys
#   models/t5_base_discourse/final/        — discourse-fine-tuned T5 weights + LRE head (~900 MB)
#   data/synthetic_discourse.jsonl         — 1,400 generated discourse groups (~2 MB)

# The ASLG-PC12 dataset downloads automatically on first run.
# Oracle labels for LRE training are cached after first generation.
# Gloss vocabulary cache (data/aslg_vocab_cache.json) is rebuilt automatically if missing.
```

## Streaming Policies

**External baselines (published methods):**

| Policy | Description |
|--------|-------------|
| Batch | Oracle: translate complete sentences (upper bound) |
| Wait-k | Fixed k-token delay (Ma et al., 2019) |
| TransLLaMa | Model outputs `<WAIT>` or a translation at each step (Agostinelli et al., 2023) |

**This work:**

| Policy | Description |
|--------|-------------|
| **TLAS** | Temporal + Linguistic fusion (primary contribution) |
| TLAS-temporal | Ablation: TPD only, no LRE |
| TLAS-linguistic | Ablation: LRE only, no TPD |
| LSG | Ablation: our prior approach using raw KL divergence + confidence heuristics, before the trained LRE head |

## Translation Backends

All backends implement the same `TranslationBackend` protocol, so any policy works with any backend:

| Backend | Model | Readiness scoring |
|---------|-------|-----------|
| **T5** | Fine-tuned T5-base (`models/t5_base_discourse/final/`) | Native LRE head on encoder hidden states |
| **Gemini** | Gemini (configurable via `GEMINI_MODEL`) | Standalone T5 encoder + LRE head (local, ~5ms) |
| **Ollama** | Any local model (e.g. GPT-OSS) | Standalone T5 encoder + LRE head (local, ~5ms) |
| **Groq** | Llama 3.1 8B | Standalone T5 encoder + LRE head (local, ~5ms) |

## Evaluation Metrics

- **BLEU-4** (sacrebleu) — primary translation quality metric
- **ROUGE-L** — longest common subsequence overlap
- **SBERT** (all-mpnet-base-v2) — sentence-level semantic similarity; more discriminative than token-level BERTScore for streaming evaluation
- **chrF++** (sacrebleu, char_order=6, word_order=2) — character n-gram F-score, robust to morphological variation

## Project Structure

```
src/
  config.py                 # All hyperparameters in one place
  pipeline.py               # Async streaming pipeline with context management
  data/
    loader.py               # ASLG-PC12 + SIGNUM data loading
    stream_simulator.py     # Replay timestamped gloss streams
  backends/
    base.py                 # TranslationBackend protocol
    t5_backend.py           # Fine-tuned T5 + LRE head
    groq_backend.py         # Groq API (Llama 3.1)
    gemini_backend.py       # Google Gemini API
    ollama_backend.py       # Local Ollama API
  policy/
    tpd.py                  # Temporal Pause Detector
    lre.py                  # Linguistic Readiness Estimator
    afg.py                  # Adaptive Fusion Gate
    tlas.py                 # TLAS policy (TPD + LRE + AFG)
    baselines.py            # External baselines: Batch, Wait-k, TransLLaMa; Ablation: LSG
  training/
    train_t5.py             # T5 fine-tuning with TransLLaMa-style augmentation
    train_lre.py            # LRE head training with semantic oracle labels
    synthetic_data.py       # LLM-based synthetic data generation
  evaluation/
    metrics.py              # BLEU, ROUGE, BERTScore, AL, LAAL
    evaluate.py             # Full evaluation matrix
    evaluate_context.py     # Discourse context ablation (with vs. without prior context)
    streaming_demo.py       # Live console demo
```

## Data

- **ASLG-PC12**: ~87K ASL gloss-English pairs ([HuggingFace](https://huggingface.co/datasets/achrafothman/aslg_pc12))
- **SIGNUM**: 779 German Sign Language sentence pairs (included in `data/`)
- **Synthetic discourse** (`data/synthetic_discourse.jsonl`): 1,400 multi-sentence discourse groups (monologue, deaf-deaf dialog, deaf-hearing dialog) generated via Gemini with vocabulary constrained to attested ASL glosses. First 200 groups = held-out test set; remaining 1,200 = training data (~5,200 genuine discourse context pairs).
- **Timestamped streams**: Token-by-token gloss streams with millisecond timestamps for streaming evaluation (included in `data/`)

## Configuration

All hyperparameters are centralized in `src/config.py`. Key TLAS parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tpd_alpha` | 0.3 | EMA smoothing factor for inter-gloss timing |
| `tpd_pause_multiplier` | 2.5 | Ratio above EMA that triggers full pause signal |
| `tpd_ema_prior_ms` | 450.0 | EMA prior (ms) to avoid cold-start fluctuations |
| `afg_weight_temporal` | 0.4 | Weight for TPD in the fusion gate |
| `afg_weight_linguistic` | 0.6 | Weight for LRE in the fusion gate |
| `afg_threshold` | 0.40 | Combined score threshold for WRITE decision |
| `afg_max_lag` | 6 | Safety valve: force WRITE after N glosses |
| `context_window_size` | 3 | Previous translations prepended for discourse context |

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
