"""
Central configuration for the TLAS streaming translation system.

Loads defaults from config/default.yaml, then overlays:
  1. A custom YAML file (set TLAS_CONFIG env var)
  2. Environment variables for API keys (.env file via python-dotenv)

All downstream code does: ``from src.config import cfg``
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ── Locate project root and load .env ────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


# ── Load YAML ────────────────────────────────────────────────────────────────

def _load_yaml() -> dict:
    """Load default config, then overlay a user-specified config if present."""
    default_path = _ROOT / "config" / "default.yaml"
    with open(default_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Overlay custom config if TLAS_CONFIG is set
    custom = os.getenv("TLAS_CONFIG")
    if custom:
        custom_path = Path(custom) if Path(custom).is_absolute() else _ROOT / custom
        with open(custom_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        for section, values in override.items():
            if isinstance(values, dict) and section in data:
                data[section].update(values)
            else:
                data[section] = values

    return data


_Y = _load_yaml()


# ── Dataclasses (populated from YAML) ────────────────────────────────────────

def _resolve(path_str: str) -> Path:
    """Resolve a path string relative to project root."""
    p = Path(path_str)
    return p if p.is_absolute() else _ROOT / p


def _sec(name: str) -> dict:
    return _Y.get(name, {})


@dataclass
class Paths:
    root: Path = field(default_factory=lambda: _ROOT)
    data: Path = field(default_factory=lambda: _resolve(_sec("paths").get("data", "data")))
    checkpoints: Path = field(default_factory=lambda: _resolve(_sec("paths").get("checkpoints", "checkpoints")))
    results: Path = field(default_factory=lambda: _resolve(_sec("paths").get("results", "results")))

    signum_glosses: Path = field(default_factory=lambda: _resolve(_sec("paths").get("signum_glosses", "data/signum_sents_anno_eng.txt")))
    signum_translations: Path = field(default_factory=lambda: _resolve(_sec("paths").get("signum_translations", "data/signum_sents_trans_eng.txt")))
    signum_vocab: Path = field(default_factory=lambda: _resolve(_sec("paths").get("signum_vocab", "data/signum_signs_anno_eng.txt")))

    monologue1_stream: Path = field(default_factory=lambda: _resolve(_sec("paths").get("monologue1_stream", "data/monologue1-gloss_stream.txt")))
    monologue2_stream: Path = field(default_factory=lambda: _resolve(_sec("paths").get("monologue2_stream", "data/monologue2-gloss_stream.txt")))
    dialog3_stream: Path = field(default_factory=lambda: _resolve(_sec("paths").get("dialog3_stream", "data/dialog3-stream.txt")))

    monologue1: Path = field(default_factory=lambda: _resolve(_sec("paths").get("monologue1", "data/monologue1.txt")))
    monologue2: Path = field(default_factory=lambda: _resolve(_sec("paths").get("monologue2", "data/monologue2.txt")))
    dialog3: Path = field(default_factory=lambda: _resolve(_sec("paths").get("dialog3", "data/dialog3.txt")))

    def __post_init__(self):
        self.checkpoints.mkdir(exist_ok=True)
        self.results.mkdir(exist_ok=True)


def _api() -> dict:
    return _sec("api")

@dataclass
class APIConfig:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", _api().get("groq_model", "llama-3.1-8b-instant")))

    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", _api().get("gemini_model", "gemini-2.5-flash")))

    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", _api().get("ollama_url", "http://localhost:11434/api/generate")))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", _api().get("ollama_model", "gpt-oss-32k:latest")))

    api_timeout: int = field(default_factory=lambda: _api().get("api_timeout", 60))
    api_delay: float = field(default_factory=lambda: _api().get("api_delay", 0.3))
    max_retries: int = field(default_factory=lambda: _api().get("max_retries", 3))


def _t5() -> dict:
    return _sec("t5")

@dataclass
class T5Config:
    model_name: str = field(default_factory=lambda: _t5().get("model_name", "t5-base"))
    checkpoint_dir: str = field(default_factory=lambda: _t5().get("checkpoint_dir", "checkpoints/t5_tlas"))

    gloss_to_text_prefix: str = field(default_factory=lambda: _t5().get("gloss_to_text_prefix", "translate ASL to English: "))
    text_to_gloss_prefix: str = field(default_factory=lambda: _t5().get("text_to_gloss_prefix", "translate English to ASL: "))

    wait_token: str = field(default_factory=lambda: _t5().get("wait_token", "<WAIT>"))
    sentence_sep: str = field(default_factory=lambda: _t5().get("sentence_sep", "|||"))

    max_source_length: int = field(default_factory=lambda: _t5().get("max_source_length", 256))
    max_target_length: int = field(default_factory=lambda: _t5().get("max_target_length", 128))

    num_beams: int = field(default_factory=lambda: _t5().get("num_beams", 4))
    no_repeat_ngram_size: int = field(default_factory=lambda: _t5().get("no_repeat_ngram_size", 2))
    early_stopping: bool = field(default_factory=lambda: _t5().get("early_stopping", True))


def _tr() -> dict:
    return _sec("training")

@dataclass
class TrainingConfig:
    num_train_samples: int = field(default_factory=lambda: _tr().get("num_train_samples", 3000))
    num_val_samples: int = field(default_factory=lambda: _tr().get("num_val_samples", 300))
    num_test_samples: int = field(default_factory=lambda: _tr().get("num_test_samples", 100))
    random_seed: int = field(default_factory=lambda: _tr().get("random_seed", 42))

    num_epochs: int = field(default_factory=lambda: _tr().get("num_epochs", 10))
    batch_size: int = field(default_factory=lambda: _tr().get("batch_size", 16))
    gradient_accumulation_steps: int = field(default_factory=lambda: _tr().get("gradient_accumulation_steps", 2))
    learning_rate: float = field(default_factory=lambda: _tr().get("learning_rate", 3e-4))
    warmup_ratio: float = field(default_factory=lambda: _tr().get("warmup_ratio", 0.1))
    weight_decay: float = field(default_factory=lambda: _tr().get("weight_decay", 0.05))
    label_smoothing_factor: float = field(default_factory=lambda: _tr().get("label_smoothing_factor", 0.1))
    fp16: bool = field(default_factory=lambda: _tr().get("fp16", True))

    max_streaming_examples_per_sentence: int = field(default_factory=lambda: _tr().get("max_streaming_examples_per_sentence", 3))

    lre_epochs: int = field(default_factory=lambda: _tr().get("lre_epochs", 5))
    lre_learning_rate: float = field(default_factory=lambda: _tr().get("lre_learning_rate", 1e-3))
    lre_batch_size: int = field(default_factory=lambda: _tr().get("lre_batch_size", 64))

    save_steps: int = field(default_factory=lambda: _tr().get("save_steps", 500))
    eval_steps: int = field(default_factory=lambda: _tr().get("eval_steps", 500))
    save_total_limit: int = field(default_factory=lambda: _tr().get("save_total_limit", 2))
    load_best_model_at_end: bool = field(default_factory=lambda: _tr().get("load_best_model_at_end", True))


def _tlas() -> dict:
    return _sec("tlas")

@dataclass
class TLASConfig:
    tpd_alpha: float = field(default_factory=lambda: _tlas().get("tpd_alpha", 0.3))
    tpd_pause_multiplier: float = field(default_factory=lambda: _tlas().get("tpd_pause_multiplier", 2.5))
    tpd_min_buffer: int = field(default_factory=lambda: _tlas().get("tpd_min_buffer", 1))
    tpd_ema_prior_ms: float = field(default_factory=lambda: _tlas().get("tpd_ema_prior_ms", 450.0))

    lre_hidden_dim: int = field(default_factory=lambda: _tlas().get("lre_hidden_dim", 256))
    lre_dropout: float = field(default_factory=lambda: _tlas().get("lre_dropout", 0.1))

    afg_weight_temporal: float = field(default_factory=lambda: _tlas().get("afg_weight_temporal", 0.4))
    afg_weight_linguistic: float = field(default_factory=lambda: _tlas().get("afg_weight_linguistic", 0.6))
    afg_threshold: float = field(default_factory=lambda: _tlas().get("afg_threshold", 0.55))
    afg_strong_pause_threshold: float = field(default_factory=lambda: _tlas().get("afg_strong_pause_threshold", 0.8))
    afg_min_readiness_for_pause: float = field(default_factory=lambda: _tlas().get("afg_min_readiness_for_pause", 0.3))
    afg_max_lag: int = field(default_factory=lambda: _tlas().get("afg_max_lag", 6))

    context_window_size: int = field(default_factory=lambda: _tlas().get("context_window_size", 3))


def _bl() -> dict:
    return _sec("baselines")

@dataclass
class BaselineConfig:
    wait_k: int = field(default_factory=lambda: _bl().get("wait_k", 3))

    lsg_kl_threshold: float = field(default_factory=lambda: _bl().get("lsg_kl_threshold", 2.0))
    lsg_confidence_threshold: float = field(default_factory=lambda: _bl().get("lsg_confidence_threshold", 0.70))
    lsg_baseline_k: int = field(default_factory=lambda: _bl().get("lsg_baseline_k", 2))
    lsg_max_lag: int = field(default_factory=lambda: _bl().get("lsg_max_lag", 6))

    mma_enabled: bool = field(default_factory=lambda: _bl().get("mma_enabled", True))


def _ev() -> dict:
    return _sec("evaluation")

@dataclass
class EvalConfig:
    compute_bleu: bool = field(default_factory=lambda: _ev().get("compute_bleu", True))
    compute_rouge: bool = field(default_factory=lambda: _ev().get("compute_rouge", True))
    compute_bertscore: bool = field(default_factory=lambda: _ev().get("compute_bertscore", True))
    compute_latency: bool = field(default_factory=lambda: _ev().get("compute_latency", True))

    bertscore_model: str = field(default_factory=lambda: _ev().get("bertscore_model", "roberta-large"))

    simulated_intergloss_ms: float = field(default_factory=lambda: _ev().get("simulated_intergloss_ms", 450.0))
    simulated_intergloss_std_ms: float = field(default_factory=lambda: _ev().get("simulated_intergloss_std_ms", 100.0))

    results_dir: str = field(default_factory=lambda: _ev().get("results_dir", "results"))


# ── Master Config ────────────────────────────────────────────────────────────

@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    api: APIConfig = field(default_factory=APIConfig)
    t5: T5Config = field(default_factory=T5Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tlas: TLASConfig = field(default_factory=TLASConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


# Module-level default instance — import this everywhere
cfg = Config()
