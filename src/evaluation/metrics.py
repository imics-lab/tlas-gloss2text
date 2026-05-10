"""
Evaluation metrics for streaming and batch translation.

Metrics:
  - BLEU-4         (sacrebleu corpus-level)
  - ROUGE-1/2/L    (rouge_score)
  - BERTScore F1   (bert_score, roberta-large)
  - Average Lagging (AL)   — Ma et al. (2019)
  - LAAL            — Length-Adaptive Average Lagging
  - Retention rate  — streaming BLEU / batch BLEU (how much quality is retained)

All metric functions accept lists of strings for corpus-level scoring.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    bleu:          float          = 0.0
    rouge1:        float          = 0.0
    rouge2:        float          = 0.0
    rougeL:        float          = 0.0
    bertscore_f1:  Optional[float] = None   # None when not computed
    avg_lagging:   float          = 0.0   # Average Lagging (AL)
    laal:          float          = 0.0   # Length-Adaptive AL
    retention:     float          = 0.0   # streaming_bleu / batch_bleu (set externally)
    num_samples:   int            = 0
    extras:        Dict           = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "bleu":         round(self.bleu,         2),
            "rouge1":       round(self.rouge1,        4),
            "rouge2":       round(self.rouge2,        4),
            "rougeL":       round(self.rougeL,        4),
            "bertscore_f1": round(self.bertscore_f1,  4) if self.bertscore_f1 is not None else None,
            "avg_lagging":  round(self.avg_lagging,   3),
            "laal":         round(self.laal,          3),
            "retention":    round(self.retention,     4),
            "num_samples":  self.num_samples,
        }


# ── BLEU ─────────────────────────────────────────────────────────────────────

def compute_bleu(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Corpus-level BLEU-4 (sacrebleu)."""
    try:
        import sacrebleu
        result = sacrebleu.corpus_bleu(hypotheses, [references])
        return result.score  # 0–100
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        return 0.0


# ── ROUGE ─────────────────────────────────────────────────────────────────────

def compute_rouge(
    hypotheses: List[str],
    references: List[str],
) -> Tuple[float, float, float]:
    """
    Corpus-level ROUGE-1, ROUGE-2, ROUGE-L (average of sentence-level scores).
    Returns (rouge1, rouge2, rougeL) as F-measure in [0, 1].
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1_total, r2_total, rL_total = 0.0, 0.0, 0.0
        n = 0
        for hyp, ref in zip(hypotheses, references):
            if not ref.strip():
                continue
            scores = scorer.score(ref, hyp)
            r1_total += scores["rouge1"].fmeasure
            r2_total += scores["rouge2"].fmeasure
            rL_total += scores["rougeL"].fmeasure
            n += 1
        if n == 0:
            return 0.0, 0.0, 0.0
        return r1_total / n, r2_total / n, rL_total / n
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        return 0.0, 0.0, 0.0


# ── BERTScore ─────────────────────────────────────────────────────────────────

def compute_bertscore(
    hypotheses: List[str],
    references: List[str],
    model_type: str = None,
    batch_size: int = 16,
) -> float:
    """
    Corpus-level BERTScore F1 (mean over sentences).

    Uses the standard token-level BERTScore formula (Zhang et al., 2020):
      - For each hypothesis token, find the max cosine similarity to any
        reference token  → Precision
      - For each reference token, find the max cosine similarity to any
        hypothesis token → Recall
      - F1 = 2 * P * R / (P + R)

    Implemented directly via transformers (bypasses the bert_score library
    to avoid tokenizer incompatibilities).  Returns value in [0, 1].
    """
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel

        model_path = model_type or "models/roberta-large"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device).eval()

        def _get_token_embeddings(texts: List[str]):
            """
            Returns list of [seq_len, hidden] tensors (one per text),
            L2-normalised, with padding tokens excluded.
            """
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
                with torch.no_grad():
                    out = model(**enc)
                hidden = out.last_hidden_state  # [B, T, H]
                mask = enc["attention_mask"]    # [B, T]
                for j in range(hidden.size(0)):
                    # Keep only non-padding tokens
                    valid = mask[j].bool()
                    vecs = hidden[j][valid]      # [valid_len, H]
                    vecs = F.normalize(vecs, dim=-1).cpu()
                    all_embeddings.append(vecs)
            return all_embeddings

        hyp_embeds = _get_token_embeddings(hypotheses)
        ref_embeds = _get_token_embeddings(references)

        f1_scores = []
        for h_vecs, r_vecs in zip(hyp_embeds, ref_embeds):
            # Cosine similarity matrix [hyp_len, ref_len]
            sim = torch.mm(h_vecs, r_vecs.t())
            # Precision: for each hyp token, best match in ref
            precision = sim.max(dim=1).values.mean().item()
            # Recall: for each ref token, best match in hyp
            recall = sim.max(dim=0).values.mean().item()
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            f1_scores.append(f1)

        return float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0

    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        return 0.0


# ── Sentence-BERT similarity ──────────────────────────────────────────────────

_SBERT_MODEL_PATH = "models/all-mpnet-base-v2/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130"

def compute_sbert(
    hypotheses: List[str],
    references: List[str],
    model_path: str = None,
    batch_size: int = 64,
) -> float:
    """
    Corpus-level Sentence-BERT cosine similarity (mean over sentence pairs).

    Uses all-mpnet-base-v2, trained with contrastive objectives so that
    semantically different sentences have low cosine similarity (~0.0) while
    paraphrases score ~0.85–0.95.  Dynamic range is much wider than BERTScore.

    Returns value in [-1, 1], typically [0, 1] for related English text.
    """
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        path = model_path or _SBERT_MODEL_PATH
        model = SentenceTransformer(path)

        hyp_emb = model.encode(hypotheses, batch_size=batch_size,
                               convert_to_tensor=True, show_progress_bar=False)
        ref_emb = model.encode(references, batch_size=batch_size,
                               convert_to_tensor=True, show_progress_bar=False)

        # Sentence-level cosine similarities
        cos_scores = torch.nn.functional.cosine_similarity(hyp_emb, ref_emb, dim=-1)
        return float(cos_scores.mean().item())

    except Exception as e:
        logger.warning(f"Sentence-BERT computation failed: {e}")
        return 0.0


# ── chrF ──────────────────────────────────────────────────────────────────────

def compute_chrf(
    hypotheses: List[str],
    references: List[str],
    char_order: int = 6,
    word_order: int = 2,
    beta: float = 2.0,
) -> float:
    """
    Corpus-level chrF++ score (sacrebleu).

    Character n-gram F-score with word n-grams (chrF++, word_order=2).
    Established MT metric that shows better discrimination than BLEU for
    paraphrase-heavy outputs.  No external model required.

    Args:
        char_order:  Character n-gram order (default 6, standard for chrF).
        word_order:  Word n-gram order (2 = chrF++, 0 = original chrF).
        beta:        Recall weight in F-score (default 2, standard for chrF).

    Returns value in [0, 100] (sacrebleu convention).
    """
    try:
        import sacrebleu
        chrf = sacrebleu.corpus_chrf(
            hypotheses,
            [references],
            char_order=char_order,
            word_order=word_order,
            beta=beta,
        )
        return float(chrf.score)
    except Exception as e:
        logger.warning(f"chrF computation failed: {e}")
        return 0.0


# ── Average Lagging ───────────────────────────────────────────────────────────

def compute_average_lagging(
    sentence_results,
) -> Tuple[float, float]:
    """
    Compute Average Lagging (AL) and Length-Adaptive AL (LAAL) across a list
    of SentenceResult objects.

    AL  (Ma et al. 2019): measures how many extra source tokens are read
        before each output token is emitted, averaged over the whole corpus.

    LAAL: normalises AL by sentence length ratio |x|/|y| so that short
        source sentences are not unfairly penalised.

    Returns (al, laal).
    """
    al_sum, laal_sum, n = 0.0, 0.0, 0
    for sr in sentence_results:
        al = sr.average_lagging
        ref_words = len(sr.reference.split()) if sr.reference else 1
        src_words = len(sr.gloss_input.split()) if sr.gloss_input else 1
        ratio = src_words / max(ref_words, 1)
        laal_sum += al * ratio
        al_sum   += al
        n += 1
    if n == 0:
        return 0.0, 0.0
    return al_sum / n, laal_sum / n


# ── Master metric computation ─────────────────────────────────────────────────

def compute_all_metrics(
    sentence_results,
    batch_bleu: Optional[float] = None,
    compute_bertscore_flag: bool = True,
    bertscore_model: str = None,
) -> MetricResult:
    """
    Compute all metrics from a list of SentenceResult objects.

    Args:
        sentence_results:       List of SentenceResult.
        batch_bleu:             Batch BLEU for computing retention rate.
        compute_bertscore_flag: Whether to compute BERTScore (slow).
        bertscore_model:        Override BERTScore model.
    """
    hypotheses = [sr.full_translation for sr in sentence_results]
    references  = [sr.reference       for sr in sentence_results]

    # Filter pairs where reference is empty
    pairs = [(h, r) for h, r in zip(hypotheses, references) if r and r.strip()]
    if not pairs:
        return MetricResult(num_samples=0)

    hyps, refs = zip(*pairs)
    hyps = list(hyps)
    refs = list(refs)

    bleu   = compute_bleu(hyps, refs)
    r1, r2, rL = compute_rouge(hyps, refs)

    bs = None
    if compute_bertscore_flag:
        bs = compute_bertscore(hyps, refs, model_type=bertscore_model)

    al, laal = compute_average_lagging(sentence_results)

    retention = 0.0
    if batch_bleu and batch_bleu > 0:
        retention = bleu / batch_bleu

    return MetricResult(
        bleu=bleu,
        rouge1=r1,
        rouge2=r2,
        rougeL=rL,
        bertscore_f1=bs,
        avg_lagging=al,
        laal=laal,
        retention=retention,
        num_samples=len(pairs),
    )


# ── Convenience: compute from hypothesis/reference lists ──────────────────────

# Aliases to avoid shadowing by parameter names in compute_metrics()
compute_bertscore_fn = compute_bertscore
compute_sbert_fn = compute_sbert
compute_chrf_fn = compute_chrf


@dataclass
class SimpleMetrics:
    """Lightweight result for hypothesis/reference list evaluation."""
    bleu: float = 0.0
    rouge_l: float = 0.0
    bertscore_f1: Optional[float] = None
    sbert: Optional[float] = None
    chrf: Optional[float] = None

    def to_dict(self) -> Dict:
        d = {"bleu": round(self.bleu, 2), "rouge_l": round(self.rouge_l, 4)}
        if self.bertscore_f1 is not None:
            d["bertscore_f1"] = round(self.bertscore_f1, 4)
        if self.sbert is not None:
            d["sbert"] = round(self.sbert, 4)
        if self.chrf is not None:
            d["chrf"] = round(self.chrf, 2)
        return d


def compute_metrics(
    hypotheses: List[str],
    references: List[str],
    sources: List[str] = None,
    compute_bertscore: bool = False,
    compute_sbert: bool = False,
    compute_chrf: bool = False,
    bertscore_model: str = None,
) -> SimpleMetrics:
    """
    Compute BLEU, ROUGE-L, and optional semantic metrics from
    hypothesis/reference string lists.

    This is a convenience wrapper for evaluate_context.py and other scripts
    that don't use SentenceResult objects.
    """
    pairs = [(h, r) for h, r in zip(hypotheses, references) if r and r.strip()]
    if not pairs:
        return SimpleMetrics()
    hyps, refs = zip(*pairs)
    hyps, refs = list(hyps), list(refs)

    bleu_score = compute_bleu(hyps, refs)
    _, _, rL = compute_rouge(hyps, refs)

    bs = compute_bertscore_fn(hyps, refs, model_type=bertscore_model) if compute_bertscore else None
    sb = compute_sbert_fn(hyps, refs) if compute_sbert else None
    ch = compute_chrf_fn(hyps, refs) if compute_chrf else None

    return SimpleMetrics(bleu=bleu_score, rouge_l=rL, bertscore_f1=bs, sbert=sb, chrf=ch)


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def format_metrics_table(results: Dict[str, MetricResult]) -> str:
    """
    Format a {policy_name: MetricResult} dict as a Markdown table.
    Columns: Policy | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | AL | LAAL | Retention
    """
    header = (
        "| Policy | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | "
        "AL | LAAL | Retention |"
    )
    sep = "|--------|------|---------|---------|---------|--------------|----|----|-----------|"
    rows = [header, sep]
    for name, m in sorted(results.items()):
        bs = f"{m.bertscore_f1:.4f}" if m.bertscore_f1 is not None else "N/A"
        rows.append(
            f"| {name} | {m.bleu:.2f} | {m.rouge1:.4f} | {m.rouge2:.4f} | "
            f"{m.rougeL:.4f} | {bs} | {m.avg_lagging:.3f} | "
            f"{m.laal:.3f} | {m.retention:.4f} |"
        )
    return "\n".join(rows)
