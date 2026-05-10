"""
Synthetic ASL gloss–text pair generation via LLM.

Three generation modes:
  1. Single pairs  (--mode pairs)      → data/synthetic_pairs.jsonl
  2. Discourse     (--mode discourse)  → data/synthetic_discourse.jsonl
  3. Timestamps    (--mode timestamps) → adds gloss_timestamps_ms to existing discourse JSONL

Discourse mode generates connected multi-sentence monologues/dialogs with:
  - Vocabulary constrained to attested ASLG-PC12 + SIGNUM glosses
  - Real discourse examples as few-shot prompts
  - Post-hoc OOV validation (rejects groups with > OOV threshold)

Timestamp mode annotates existing discourse groups with realistic per-gloss
timestamps via LLM, simulating natural ASL signing rhythm (one group per API call).

Run:
  python -m src.training.synthetic_data --mode pairs      [--n 500] [--backend gemini]
  python -m src.training.synthetic_data --mode discourse  [--n 200] [--backend gemini]
  python -m src.training.synthetic_data --mode timestamps [--backend gemini]
  python -m src.training.synthetic_data --mode vocab                # extract + cache vocab only
"""

import argparse
import asyncio
import json
import logging
import re
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ── Gloss morphology / root extraction ────────────────────────────────────────

def _extract_roots(token: str) -> List[str]:
    """
    Normalize a gloss token to its base root(s) for vocabulary matching.

    Rules (applied in order):
      - Compounds (SIGN+SIGN)         → split and recurse on each part
      - Directional verbs             → extract uppercase ROOT from i-ROOT-you, he-ROOT-i
      - Lowercase spatial/modifier    → strip suffix from ROOT-right, WE-all, STRONG-much
      - Otherwise                     → return token as-is (e.g. NOT-HAVE, A-LOT-OF)
    """
    if not token:
        return []
    if '+' in token:
        roots = []
        for part in token.split('+'):
            roots.extend(_extract_roots(part))
        return [r for r in roots if r]
    # Directional verb: lowercase-UPPERCASE[-lowercase]  e.g. i-TELL-you, he-COME-middle
    m = re.match(r'^[a-z][a-z]*-([A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*)(?:-[a-z][a-z]*)?$', token)
    if m:
        return [m.group(1)]
    # Uppercase root with lowercase suffix: STRONG-much, WE-all, X-right, HOSPITAL-to
    m = re.match(r'^([A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*)-[a-z]', token)
    if m:
        return [m.group(1)]
    return [token]


# ── Gloss normalization ───────────────────────────────────────────────────────

# Grammatical markers that should always appear as lowercase suffixes after '-'.
# The model often generates e.g. COME-HERE, WE-ALL, STRONG-MUCH — normalize them.
_LOWERCASE_SUFFIXES = frozenset({
    # Spatial / directional
    'here', 'there', 'right', 'left', 'middle', 'to', 'from',
    # Pronoun suffixes (directional verbs)
    'i', 'you', 'he', 'she', 'we',
    # Quantifier / modifier suffixes
    'all', 'both', 'much', 'wide', 'sth', 'space',
})
_LS_UPPER = frozenset(s.upper() for s in _LOWERCASE_SUFFIXES)


def _normalize_gloss_token(tok: str) -> str:
    """
    Fix common generation errors in a single gloss token:
      - Strip trailing punctuation: LIKE? → LIKE, GO? → GO
      - Uppercase direction suffix → lowercase: COME-HERE → COME-here
      - Uppercase pronoun suffix  → lowercase: he-TELL-YOU → he-TELL-you
      - Uppercase modifier suffix → lowercase: WE-ALL → WE-all
    Compound tokens (using '+') are handled part by part.
    """
    # Strip trailing punctuation (?, !, ., ,)
    tok = tok.rstrip('?!.,')
    if not tok:
        return tok
    if '+' in tok:
        return '+'.join(_normalize_gloss_token(p) for p in tok.split('+'))
    parts = tok.split('-')
    if len(parts) < 2:
        return tok
    # Normalize each segment that is a known lowercase suffix
    normalized = []
    for i, part in enumerate(parts):
        if i > 0 and part in _LS_UPPER:
            normalized.append(part.lower())
        else:
            normalized.append(part)
    return '-'.join(normalized)


def _normalize_gloss(gloss: str) -> str:
    """Apply token-level normalization to a full gloss sequence."""
    return ' '.join(_normalize_gloss_token(tok) for tok in gloss.split())


# ── Vocabulary extraction and caching ─────────────────────────────────────────

_VOCAB_CACHE_PATH = cfg.paths.data / "aslg_vocab_cache.json"

_GLOSS_LINE_RE = re.compile(r'\(Gloss\):\s*(.*)')


def _collect_local_tokens() -> Tuple[Set[str], Counter]:
    """Collect all gloss tokens from SIGNUM and local discourse files."""
    full_forms: Set[str] = set()
    root_counts: Counter = Counter()

    def _add(token: str, weight: int = 1) -> None:
        tok = token.strip()
        if tok:
            full_forms.add(tok.upper())          # store uppercase for matching
            for root in _extract_roots(tok):     # extract roots from original case
                root_counts[root.upper()] += weight

    # SIGNUM sign vocabulary (well-curated: upweight)
    if cfg.paths.signum_vocab.exists():
        with open(cfg.paths.signum_vocab, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    for variant in parts[1].split('|'):
                        _add(variant, weight=10)

    # SIGNUM sentence glosses
    if cfg.paths.signum_glosses.exists():
        with open(cfg.paths.signum_glosses, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    for tok in parts[1].split():  # keep original case for root extraction
                        _add(tok)

    # Sentence-level monologue / dialog files
    discourse_files = [
        cfg.paths.data / "monologue1.txt",
        cfg.paths.data / "monologue2.txt",
        cfg.paths.data / "dialog1.txt",
        cfg.paths.data / "dialog2.txt",
        cfg.paths.data / "dialog3.txt",
    ]
    for path in discourse_files:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = _GLOSS_LINE_RE.search(line)
                if m:
                    for tok in m.group(1).split():
                        _add(tok)

    # Token-level stream files
    for stream_path in [cfg.paths.monologue1_stream, cfg.paths.monologue2_stream]:
        if not stream_path.exists():
            continue
        with open(stream_path, encoding="utf-8") as f:
            for line in f:
                m = _GLOSS_LINE_RE.search(line)
                if m:
                    _add(m.group(1).strip())

    return full_forms, root_counts


def build_gloss_vocabulary(
    include_aslg: bool = True,
    aslg_top_n: int = 400,
    cache_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Tuple[Set[str], Set[str], List[str]]:
    """
    Build gloss vocabulary from SIGNUM, local files, and optionally ASLG-PC12.

    Returns:
      full_forms  – every exact token attested in training corpora
      root_vocab  – all normalized roots (used for OOV validation)
      prompt_roots – top-N roots sorted alphabetically (for LLM prompt reference)
    """
    cache_path = cache_path or _VOCAB_CACHE_PATH

    if not force_rebuild and cache_path.exists():
        logger.info(f"Loading vocabulary cache from {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
        return set(data["full_forms"]), set(data["root_vocab"]), data["prompt_roots"]

    full_forms, root_counts = _collect_local_tokens()
    logger.info(f"Local vocab: {len(full_forms)} full forms, {len(root_counts)} roots")

    if include_aslg:
        logger.info("Extracting vocabulary from ASLG-PC12 (cached after first run)...")
        try:
            from datasets import load_dataset
            ds = load_dataset("achrafothman/aslg_pc12")
            for item in ds["train"]:
                for tok in item["gloss"].upper().split():
                    tok = tok.strip()
                    if tok:
                        full_forms.add(tok)
                        for root in _extract_roots(tok):
                            root_counts[root] += 1
            logger.info(f"After ASLG-PC12: {len(full_forms)} full forms, {len(root_counts)} roots")
        except Exception as e:
            logger.warning(f"Could not load ASLG-PC12 for vocabulary: {e}")

    root_vocab = set(root_counts.keys())
    prompt_roots = sorted(r for r, _ in root_counts.most_common(aslg_top_n))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({
            "full_forms": sorted(full_forms),
            "root_vocab": sorted(root_vocab),
            "prompt_roots": prompt_roots,
            "root_counts": dict(root_counts.most_common(3000)),
        }, f, indent=2)
    logger.info(f"Vocabulary cached to {cache_path}")

    return full_forms, root_vocab, prompt_roots


def validate_gloss_vocabulary(
    gloss: str,
    root_vocab: Set[str],
    full_forms: Set[str],
    threshold: float = 0.80,
) -> Tuple[bool, float]:
    """
    Check that a gloss sequence uses known vocabulary.

    A token is "in vocabulary" if:
      (a) its exact form appears in full_forms, OR
      (b) any of its extracted roots appears in root_vocab.

    Returns (passes: bool, coverage_fraction: float).
    """
    tokens = gloss.strip().split()
    if not tokens:
        return False, 0.0
    in_vocab = 0
    for tok in tokens:
        if tok in full_forms:
            in_vocab += 1
        elif any(r in root_vocab for r in _extract_roots(tok)):
            in_vocab += 1
    coverage = in_vocab / len(tokens)
    return coverage >= threshold, coverage


# ── Real discourse examples (few-shot) ────────────────────────────────────────

# Manually translated few-shot examples derived from monologue1.txt and dialog1.txt.
# These are the only authentic connected-discourse examples we have with both
# gloss sequences and English translations.
# Few-shot examples for monologue and deaf-deaf dialog (all turns have gloss+text)
_DISCOURSE_FEW_SHOT: List[Dict] = [
    {
        "type": "monologue",
        "topic": "trip to Holland",
        "sentences": [
            {
                "gloss": "NOW i-TELL-you YESTERDAY I HOLLAND DRIVE",
                "text": "Let me tell you about yesterday when I drove to Holland.",
            },
            {
                "gloss": "IN-THE-MORNING I 6 CLOCK EARLY GET-UP",
                "text": "I got up early in the morning at 6 o'clock.",
            },
            {
                "gloss": "WEATHER NICE SUN WITHOUT-sth CLOUDS",
                "text": "The weather was nice and sunny without any clouds.",
            },
            {
                "gloss": "BUT THEN WEATHER BAD FOG STRONG-much RAIN A-LOT-OF",
                "text": "But then the weather turned bad with heavy fog and a lot of rain.",
            },
            {
                "gloss": "MY CAR X-right ENGINE BROKEN WE-all ACCIDENT HAVE-BEEN",
                "text": "My car broke down and we were in an accident.",
            },
        ],
    },
    {
        "type": "dialog",
        "topic": "work and salary",
        "sentences": [
            {
                "speaker": "A",
                "gloss": "I WORK A-LOT-OF BUT SALARY LITTLE",
                "text": "I work a lot but my salary is very low.",
            },
            {
                "speaker": "B",
                "gloss": "YOUR BOSS X-right YOU WITH-what DISCUSS ABOUT-what SALARY",
                "text": "Have you talked to your boss about the salary?",
            },
            {
                "speaker": "A",
                "gloss": "I WITH-what BOSS DISCUSS he-TELL-i FIRM MONEY NOT-HAVE",
                "text": "I talked to my boss and he said the company has no money.",
            },
            {
                "speaker": "B",
                "gloss": "YOU APPLICATION NEW WRITE SHOULD I you-HELP-i CAN",
                "text": "You should write a new job application — I can help you.",
            },
        ],
    },
]

# Few-shot example for deaf-hearing dialog.
# Deaf turns have both "gloss" and "text"; hearing turns have ONLY "text".
# Derived directly from dialog1.txt.
_DEAF_HEARING_FEW_SHOT: List[Dict] = [
    {
        "type": "dialog-deaf-hearing",
        "topic": "catching up with a friend",
        "sentences": [
            {
                "speaker": "deaf",
                "gloss": "I GLAD SEE YOU LONG-TIME NOT SEE",
                "text": "I'm so glad to see you — it's been a long time.",
            },
            {
                "speaker": "hearing",
                "text": "I'm so glad to see you too! It's been a while. How have you been?",
            },
            {
                "speaker": "deaf",
                "gloss": "I GOOD BUT WORK A-LOT-OF TIRED VERY",
                "text": "I'm doing well but I have a lot of work and I'm very tired.",
            },
            {
                "speaker": "hearing",
                "text": "That sounds exhausting. Are you at least enjoying what you're working on?",
            },
            {
                "speaker": "deaf",
                "gloss": "WORK INTERESTING BUT BOSS FURIOUS EVERY-DAY PROBLEM",
                "text": "The work is interesting but my boss gets furious every day, which is a problem.",
            },
            {
                "speaker": "hearing",
                "text": "That sounds really stressful. Maybe it's time to look for something new.",
            },
        ],
    },
]


# ── Discourse generation ───────────────────────────────────────────────────────

_GLOSS_MORPHOLOGY_RULES = """\
ASL GLOSS NOTATION RULES
========================
1. All sign glosses are UPPERCASE. No articles (a, an, the). No copula (is, are, am).
2. Word order: topic-comment (STORE I GO, not "I go to the store").
3. Negation: NOT-HAVE, NOT-WANT, LIKE NOT, COME NOT.
4. Directional verbs use lowercase pronoun prefix/suffix:
     i-TELL-you   he-EXPLAIN-i   she-HELP-you   we-SHOW-you   he-SEND-i
   Pronouns: i  you  he  she  we  (ALWAYS lowercase — never uppercase)
5. Spatial references use lowercase direction suffix:
     X-right   X-left   X-middle   STAY-right   COME-here   HOSPITAL-to
   Directions: right  left  middle  there  here  to  from  (ALWAYS lowercase)
   ✗ WRONG: COME-HERE  ENTER-HERE  BEACH-TO  WALK-THERE   (uppercase suffix = error)
   ✓ RIGHT:  COME-here  ENTER-here  BEACH-to  WALK-there
6. Compounds use '+':  CHRISTMAS+MARKET   SIGN+LANGUAGE   PLAY+TOY-space
7. Lowercase modifiers:  STRONG-much   WE-all   WE-both   BIG-wide   A-LOT-OF
8. Wh-questions put the wh-word LAST:  NAME YOUR WHAT?   CINEMA START WHEN?
9. Yes/no questions end with YOU or KNOW-WHAT:  COFFEE LIKE YOU?
10. The <WAIT> token is a special system token — do NOT use it in output.

STRICTLY PROHIBITED (these will cause your output to be rejected):
- Do NOT use DESC- as a modifier. Never write DESC-VERY, DESC-DIFFICULT, DESC-GOOD, etc.
  DESC- is a classifier predicate, not a general intensifier.
- Do NOT invent multi-word compounds joined by hyphens, such as:
    STREET-WALK  PAIN-LOCATION  FAST-MUCH  BRIGHT-WIDE  SATURDAY-DAY
    WEEKEND-FUTURE  CREATE-AROUND-HERE  POUR-AROUND-HERE  TAKE-EVERY-DAY
  If a hyphenated form is not shown in the examples or vocabulary, do NOT use it.
- Use simple, separate glosses instead: WALK STREET, PAIN HERE, FAST, BRIGHT, SATURDAY"""

_DISCOURSE_TOPICS = [
    "morning routine", "grocery shopping", "family dinner", "weekend plans",
    "doctor appointment", "trip to the city", "sports and exercise",
    "work day", "learning sign language", "meeting an old friend",
    "holiday preparation", "cooking a meal", "asking for directions",
    "school or university", "visiting relatives", "weather and seasons",
    "birthday celebration", "job interview", "buying a bicycle", "hotel stay",
]


def _vocab_block(prompt_roots: List[str]) -> str:
    return "  " + "  ".join(prompt_roots)


def _build_discourse_prompt(
    n_groups: int,
    topic: str,
    prompt_roots: List[str],
    sentences_per_group: int = 5,
) -> str:
    """Prompt for monologue and deaf-deaf dialog (all turns have gloss + text)."""
    examples_json = json.dumps(_DISCOURSE_FEW_SHOT, indent=2, ensure_ascii=False)
    return f"""{_GLOSS_MORPHOLOGY_RULES}

VOCABULARY REFERENCE
====================
Use ONLY the base signs listed below. You may freely form directional variants
(i-ROOT-you), spatial variants (ROOT-right), compounds (ROOT1+ROOT2), and
modifier suffixes (ROOT-much), but the ROOT itself must come from this list.

{_vocab_block(prompt_roots)}

TASK
====
Generate {n_groups} connected discourse group(s) on the topic: "{topic}"
Each group must have exactly {sentences_per_group} sentences that form a coherent
narrative or conversation. For dialogs, alternate between speaker "A" and "B".

Output ONLY valid JSON — a list of objects. Each object has:
  "type":      "monologue" or "dialog"
  "topic":     short topic string
  "sentences": list of objects, each with:
                 "gloss"   (uppercase ASL gloss sequence)
                 "text"    (natural English translation, 6–30 words)
                 "speaker" (only for dialog: "A" or "B")

EXAMPLES (do NOT copy — generate entirely new content):
{examples_json}

Rules for your output:
- Gloss sequences: 3–15 words, ALL-CAPS except directional/spatial affixes
- English sentences: grammatically complete, 6–30 words
- Consecutive sentences must be topically connected (co-reference, discourse connectives)
- No duplicate gloss sequences
- Output ONLY the JSON list, no commentary
"""


def _build_deaf_hearing_prompt(
    n_groups: int,
    topic: str,
    prompt_roots: List[str],
    sentences_per_group: int = 6,
) -> str:
    """
    Prompt for deaf-hearing dialog.

    Deaf turns have both gloss + text (translation target).
    Hearing turns have ONLY text (English speech — context only, no gloss).
    """
    examples_json = json.dumps(_DEAF_HEARING_FEW_SHOT, indent=2, ensure_ascii=False)
    return f"""{_GLOSS_MORPHOLOGY_RULES}

VOCABULARY REFERENCE
====================
Use ONLY the base signs listed below for the Deaf user's gloss sequences.
You may form directional variants (i-ROOT-you), spatial variants (ROOT-right),
compounds (ROOT1+ROOT2), and modifier suffixes (ROOT-much).

{_vocab_block(prompt_roots)}

TASK
====
Generate {n_groups} realistic dialog(s) between a Deaf user and a Hearing user,
on the topic: "{topic}"

IMPORTANT RULES FOR THIS TYPE:
- The Deaf user communicates in ASL gloss (signed language).
- The Hearing user communicates in natural spoken English (NO gloss).
- Each dialog must have exactly {sentences_per_group} turns total, alternating speakers.
- Start with the Deaf user or the Hearing user — vary across dialogs.

Output ONLY valid JSON — a list of objects. Each object has:
  "type":      "dialog-deaf-hearing"
  "topic":     short topic string
  "sentences": list of turn objects. Each turn is EITHER:

    Deaf user turn:
      {{"speaker": "deaf", "gloss": "ASL GLOSS SEQUENCE", "text": "English translation."}}

    Hearing user turn:
      {{"speaker": "hearing", "text": "Natural English speech."}}
    ← Hearing turns have NO "gloss" field.

EXAMPLE (do NOT copy — generate entirely new content):
{examples_json}

Rules:
- Deaf gloss: 3–15 words, ALL-CAPS except directional/spatial affixes
- All English text: grammatically complete, 6–35 words
- Turns must be conversationally coherent (each responds to the previous)
- The hearing user's replies should acknowledge what the deaf user signed
- Output ONLY the JSON list, no commentary
"""


def _parse_discourse_json(raw: str) -> List[Dict]:
    """Extract JSON discourse groups from raw LLM output, tolerating markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            l for l in text.splitlines() if not l.startswith("```")
        ).strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError:
        return []
    groups = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "sentences" not in item or not isinstance(item["sentences"], list):
            continue
        dtype  = item.get("type", "monologue")
        dtopic = item.get("topic", "")
        sents  = []
        for s in item["sentences"]:
            if not isinstance(s, dict):
                continue
            speaker = s.get("speaker", "")
            t = s.get("text", "").strip()
            # Uppercase first, then normalize known suffix capitalization errors
            g_raw = s.get("gloss", "").strip().upper()
            g = _normalize_gloss(g_raw) if g_raw else ""
            if not t:
                continue
            if speaker == "hearing":
                # Hearing turn: English only, no gloss required
                sents.append({"speaker": "hearing", "text": t})
            elif g:
                # Deaf turn (or deaf-deaf dialog): must have gloss
                entry = {"gloss": g, "text": t}
                if speaker:
                    entry["speaker"] = speaker
                sents.append(entry)
        # A valid group needs at least 2 turns and at least 1 deaf (gloss) turn
        if len(sents) >= 2 and any("gloss" in s for s in sents):
            groups.append({"type": dtype, "topic": dtopic, "sentences": sents})
    return groups


# Matches tokens that are entirely uppercase multi-segment compounds (no lowercase),
# e.g. STREET-WALK, BRIGHT-WIDE, SATURDAY-DAY, CREATE-AROUND-HERE.
# Legitimate all-uppercase compounds like NOT-HAVE, A-LOT-OF are in full_forms
# and are excluded by the vocabulary check below.
_MULTI_UPPER_COMPOUND_RE = re.compile(r'^[A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)+$')

# Uppercase direction words that should always be lowercase suffixes.
_UPPERCASE_DIRECTION_RE = re.compile(
    r'[A-Z]+-(?:HERE|THERE|RIGHT|LEFT|MIDDLE|TO|FROM)$'
)


def _has_hallucinated_form(gloss: str, full_forms: Set[str]) -> bool:
    """
    Return True if gloss contains a morphologically hallucinated token:
      - A multi-uppercase-segment compound not attested in the corpus
        (e.g. STREET-WALK, BRIGHT-WIDE, SATURDAY-DAY)
      - A token with an uppercase direction suffix that should be lowercase
        (e.g. COME-HERE, BEACH-TO — should be COME-here, BEACH-to)
    """
    for tok in gloss.split():
        if _UPPERCASE_DIRECTION_RE.match(tok):
            return True
        if _MULTI_UPPER_COMPOUND_RE.match(tok) and tok not in full_forms:
            return True
    return False


def _validate_discourse_group(
    group: Dict,
    root_vocab: Set[str],
    full_forms: Set[str],
    oov_threshold: float,
) -> Tuple[bool, float]:
    """
    Validate all deaf (gloss) turns in a group:
      1. Gloss length 3–20 tokens.
      2. No hallucinated compound forms (multi-uppercase not in vocab,
         or uppercase direction suffixes).
      3. Average vocabulary coverage ≥ oov_threshold.
    Hearing turns are English text — skipped.
    """
    coverages = []
    for sent in group["sentences"]:
        gloss = sent.get("gloss", "")
        if not gloss:
            continue  # hearing turn — skip
        tokens = gloss.split()
        if not (3 <= len(tokens) <= 20):
            return False, 0.0
        if _has_hallucinated_form(gloss, full_forms):
            return False, 0.0
        _, cov = validate_gloss_vocabulary(gloss, root_vocab, full_forms, threshold=0.0)
        coverages.append(cov)
    if not coverages:
        return False, 0.0
    avg_coverage = sum(coverages) / len(coverages)
    return avg_coverage >= oov_threshold, avg_coverage


async def generate_discourse_groups(
    n_total: int = 200,
    backend_name: str = "gemini",
    output_path: Optional[Path] = None,
    sentences_per_group: int = 5,
    groups_per_call: int = 3,
    oov_threshold: float = 0.80,
    aslg_top_n: int = 400,
    type_weights: Dict[str, float] = None,
) -> List[Dict]:
    """
    Generate connected discourse groups and save to JSONL.

    Three types are generated in a weighted mix (configurable via type_weights):
      - "monologue"          – single deaf signer narrative
      - "dialog"             – two deaf signers conversing
      - "dialog-deaf-hearing"– deaf signer + hearing speaker (hearing turns: English only)

    Each JSONL record:
      {"type": str, "topic": str, "sentences": [...]}

      Deaf turns:    {"speaker": "deaf"|"A"|"B", "gloss": str, "text": str}
      Hearing turns: {"speaker": "hearing", "text": str}   ← no "gloss" key

    Args:
        n_total:             Total discourse groups to generate.
        backend_name:        LLM backend ("gemini" recommended).
        output_path:         Output JSONL. Defaults to data/synthetic_discourse.jsonl.
        sentences_per_group: Turns per group (5–6 recommended).
        groups_per_call:     Groups requested per LLM call.
        oov_threshold:       Min average gloss vocabulary coverage to accept a group.
        aslg_top_n:          Top-N ASLG roots to include in the prompt vocabulary.
        type_weights:        Dict mapping type → relative weight. Default: 40% monologue,
                             30% deaf-deaf dialog, 30% deaf-hearing dialog.
    """
    if type_weights is None:
        type_weights = {"monologue": 0.4, "dialog": 0.3, "dialog-deaf-hearing": 0.3}
    _types  = list(type_weights.keys())
    _weights = [type_weights[t] for t in _types]

    output_path = output_path or (cfg.paths.data / "synthetic_discourse.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing output
    existing: List[Dict] = []
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        logger.info(f"Resuming: {len(existing)} groups already saved.")

    # Build vocabulary
    full_forms, root_vocab, prompt_roots = build_gloss_vocabulary(aslg_top_n=aslg_top_n)

    backend = _load_backend(backend_name)
    logger.info(
        f"Generating {n_total} discourse groups via {backend.name}  →  {output_path}\n"
        f"  types: {type_weights}\n"
        f"  vocab: {len(full_forms)} forms / {len(root_vocab)} roots / "
        f"{len(prompt_roots)} in prompt\n"
        f"  oov_threshold={oov_threshold}, ~{sentences_per_group} sentences/group"
    )

    all_groups: List[Dict] = list(existing)
    topics  = _DISCOURSE_TOPICS * (n_total // len(_DISCOURSE_TOPICS) + 2)
    random.shuffle(topics)

    rejected  = 0
    topic_idx = 0

    with open(output_path, "a", encoding="utf-8") as outf:
        while len(all_groups) < n_total:
            topic     = topics[topic_idx % len(topics)]
            topic_idx += 1
            dtype     = random.choices(_types, weights=_weights, k=1)[0]

            if dtype == "dialog-deaf-hearing":
                # Deaf-hearing dialogs have one extra turn for natural alternation
                prompt = _build_deaf_hearing_prompt(
                    groups_per_call, topic, prompt_roots, sentences_per_group + 1
                )
            else:
                prompt = _build_discourse_prompt(
                    groups_per_call, topic, prompt_roots, sentences_per_group
                )

            try:
                raw    = await _raw_generate(backend, prompt)
                groups = _parse_discourse_json(raw)
            except Exception as e:
                logger.warning(f"Generation failed (type={dtype}, topic='{topic}'): {e}")
                continue

            for group in groups:
                if len(all_groups) >= n_total:
                    break
                passes, coverage = _validate_discourse_group(
                    group, root_vocab, full_forms, oov_threshold
                )
                if passes:
                    all_groups.append(group)
                    outf.write(json.dumps(group, ensure_ascii=False) + "\n")
                    outf.flush()
                else:
                    rejected += 1
                    logger.debug(
                        f"  Rejected (type={group.get('type')}, "
                        f"topic='{topic}', coverage={coverage:.2f})"
                    )

            logger.info(
                f"  {len(all_groups)}/{n_total}  "
                f"(type={dtype}, rejected={rejected}, topic='{topic}')"
            )

    logger.info(
        f"Done. Saved {len(all_groups)} groups to {output_path}  "
        f"(rejected {rejected} OOV groups)"
    )
    await backend.close()
    return all_groups


def load_discourse_groups(
    path: Optional[Path] = None,
    split: str = "train",
    test_size: int = 200,
    seed: int = 42,
) -> List[List[Dict]]:
    """
    Load saved discourse groups, optionally split into train / test sets.

    The first `test_size` lines of the JSONL (deterministic order) form the
    held-out test set; the rest are used for training.  This way the test set
    is fixed regardless of how many more groups are appended during generation.

    Args:
        path:      JSONL file.  Defaults to data/synthetic_discourse.jsonl.
        split:     "train", "test", or "all".
        test_size: Number of groups reserved for the test split (default 200).
        seed:      Unused — split is positional, not random, for reproducibility.

    Returns a list of sentence lists in the format expected by make_context_pairs():
      Deaf turns:    {"gloss": str, "text": str}
      Hearing turns: {"text": str}  (context only)
    """
    path = path or (cfg.paths.data / "synthetic_discourse.jsonl")
    if not path.exists():
        return []

    raw_groups = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_groups.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if split == "test":
        raw_groups = raw_groups[:test_size]
    elif split == "train":
        raw_groups = raw_groups[test_size:]
    # else "all" — use everything

    groups = []
    for obj in raw_groups:
        sents = []
        for s in obj.get("sentences", []):
            if s.get("speaker") == "hearing" and "text" in s:
                sents.append({"text": s["text"]})
            elif "gloss" in s and "text" in s:
                sents.append({"gloss": s["gloss"], "text": s["text"]})
        if len(sents) >= 2 and any("gloss" in s for s in sents):
            groups.append(sents)
    return groups


# ── Single-pair generation (original mode, unchanged) ─────────────────────────

_TOPICS = [
    "weather", "daily routine", "family", "emotions", "food", "work",
    "health", "travel", "numbers and time", "sports", "education",
    "greetings and farewells", "requests and needs", "questions",
    "opinions and preferences", "location and directions",
]

_GLOSS_RULES = """\
ASL gloss rules:
- Words are ALL-CAPS
- No articles (a, an, the)
- No copula (is, are, am) in simple sentences
- Topic-comment word order: STORE I GO (not "I go to the store")
- Negation: add NOT or IX-NEG after the verb (LIKE NOT)
- Yes/no questions: add YOU or KNOW-WHAT at the end (EAT YOU?)
- WH-questions: WH-word at end (NAME YOUR WHAT?)
- Fingerspelling: preceded by # (e.g., #JOB)
- Classifiers and spatial references are simplified to IX
- Plurals: repeat or add MANY/MUCH
"""

_FEW_SHOT_EXAMPLES = [
    ("STORE I GO NEED BREAD MILK", "I need to go to the store to buy bread and milk."),
    ("YESTERDAY WORK FINISH TIRED VERY", "Yesterday I finished work and was very tired."),
    ("YOUR NAME WHAT", "What is your name?"),
    ("DOCTOR APPOINTMENT TOMORROW IX", "I have a doctor appointment tomorrow."),
    ("COFFEE LIKE YOU?", "Do you like coffee?"),
    ("WEATHER TODAY HOT VERY NOT LIKE I", "I don't like how hot the weather is today."),
    ("FAMILY FIVE MOTHER FATHER SISTER TWO", "My family has five members: mother, father, and two sisters."),
    ("HELP ME PLEASE", "Please help me."),
]


def _build_prompt(n_pairs: int, topic: str) -> str:
    examples_text = "\n".join(
        f"  GLOSS: {g}\n  ENGLISH: {e}"
        for g, e in random.sample(_FEW_SHOT_EXAMPLES, min(4, len(_FEW_SHOT_EXAMPLES)))
    )
    return f"""{_GLOSS_RULES}

Generate {n_pairs} new ASL gloss–English translation pairs about: {topic}

Use the few-shot examples below as style guidance:
{examples_text}

Output ONLY valid JSON — a list of objects, each with keys "gloss" and "text".
Example format:
[
  {{"gloss": "GLOSS WORDS HERE", "text": "English sentence here."}},
  ...
]

Ensure:
- Glosses are 3–12 words, ALL-CAPS, no punctuation
- English sentences are grammatically complete (6–25 words)
- Diverse vocabulary and sentence structures
- No duplicate glosses
"""


def _parse_json_pairs(raw: str) -> List[Dict[str, str]]:
    """Extract JSON list from raw LLM output, tolerating markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines if not l.startswith("```")
        ).strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        data = json.loads(text[start:end])
        return [
            d for d in data
            if isinstance(d, dict) and "gloss" in d and "text" in d
            and d["gloss"].strip() and d["text"].strip()
        ]
    except json.JSONDecodeError:
        return []


def _validate_pair(gloss: str, text: str) -> bool:
    """Lightweight sanity check on a generated pair."""
    gloss_words = gloss.strip().split()
    text_words  = text.strip().split()
    if not (3 <= len(gloss_words) <= 20):
        return False
    if not (4 <= len(text_words) <= 40):
        return False
    if gloss != gloss.upper():
        return False
    return True


# ── Shared async helper ───────────────────────────────────────────────────────

async def _raw_generate(backend, prompt: str) -> str:
    """Send an arbitrary prompt to the backend and return the raw string response."""
    if hasattr(backend, "_chat"):
        return await backend._chat(prompt, temperature=0.7)
    from src.backends.base import Direction
    result = await backend.translate(prompt, direction=Direction.GLOSS_TO_TEXT)
    return result.raw_response or result.translation


# ── Single-pair generation loop ───────────────────────────────────────────────

async def _generate_batch(backend, topic: str, n: int = 10) -> List[Dict[str, str]]:
    prompt = _build_prompt(n, topic)
    try:
        raw   = await _raw_generate(backend, prompt)
        pairs = _parse_json_pairs(raw)
        return [
            {"gloss": p["gloss"].upper().strip(), "text": p["text"].strip()}
            for p in pairs
            if _validate_pair(p["gloss"].upper().strip(), p["text"].strip())
        ]
    except Exception as e:
        logger.warning(f"Batch generation failed for topic '{topic}': {e}")
        return []


async def generate_synthetic_pairs(
    n_total: int = 500,
    backend_name: str = "groq",
    output_path: Optional[Path] = None,
    pairs_per_call: int = 10,
) -> List[Dict[str, str]]:
    """
    Generate `n_total` synthetic gloss–text pairs and save to JSONL.
    """
    output_path = output_path or (cfg.paths.data / "synthetic_pairs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = _load_backend(backend_name)
    logger.info(f"Generating {n_total} synthetic pairs via {backend.name}  →  {output_path}")

    all_pairs: List[Dict[str, str]] = []
    seen_glosses: set = set()

    topics = _TOPICS * (n_total // (len(_TOPICS) * pairs_per_call) + 2)
    random.shuffle(topics)

    for topic in topics:
        if len(all_pairs) >= n_total:
            break
        batch = await _generate_batch(backend, topic, n=pairs_per_call)
        new_pairs = [p for p in batch if p["gloss"] not in seen_glosses]
        seen_glosses.update(p["gloss"] for p in new_pairs)
        all_pairs.extend(new_pairs)
        logger.info(f"  {len(all_pairs)}/{n_total}  (topic: {topic})")

    all_pairs = all_pairs[:n_total]

    with output_path.open("w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(all_pairs)} synthetic pairs to {output_path}")
    await backend.close()
    return all_pairs


def load_synthetic_pairs(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Load previously generated synthetic pairs from JSONL."""
    path = path or (cfg.paths.data / "synthetic_pairs.jsonl")
    if not path.exists():
        return []
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pairs


# ── Timestamp annotation ──────────────────────────────────────────────────────

_TIMESTAMP_PROMPT_TEMPLATE = """\
You are annotating ASL (American Sign Language) discourse groups with realistic \
per-gloss timestamps, simulating how a native signer produces signs in real time.

## ASL Temporal Patterns

### Within a sentence
- Each gloss takes 300–650ms to produce (mean ~450ms).
- Common, short signs (I, YOU, GO, HAVE, NOT): 300–400ms.
- Complex or directional signs (i-TELL-you, he-COME-middle): 500–650ms.
- Compound signs (EXCELLENT+MARKET): 550–700ms.
- The first gloss of a sentence has a short startup: add 50–150ms extra.
- The last gloss before a long pause is often slightly extended: add 50–100ms.

### Between sentences (inter-sentence gap)
This is the most important signal — it distinguishes sentence boundaries from \
within-sentence pauses. Use the content and discourse structure to decide:

| Transition type                           | Gap range   |
|-------------------------------------------|-------------|
| Quick question → answer (same topic)      | 1500–2500ms |
| Statement → related follow-up             | 2500–3500ms |
| Topic continuation, slight reframing      | 3000–4500ms |
| Topic shift or new subject                | 4500–7000ms |
| Emotional statement, pause for effect     | 3500–6000ms |
| End of one speaker, start of another      | 1500–3000ms |

### Hearing turns (deaf-hearing dialogs)
Assign a single timestamp for when the hearing person's message arrives \
(i.e., when the deaf user would read/receive it). This is the time from \
the last deaf gloss plus 2000–5000ms, depending on how quickly the hearing \
person responds.

## Real examples (from actual sign language stream data)

Monologue excerpt (cumulative ms offsets from start of group):
     0ms → NOW
   450ms → i-TELL-you
  1150ms → YESTERDAY
  1750ms → I
  2050ms → HOLLAND
  2650ms → DRIVE
  6000ms → IN-THE-MORNING       ← 3350ms gap = sentence boundary
  6650ms → I
  6950ms → 6
  7300ms → CLOCK
  7800ms → EARLY
  8300ms → GET-UP
 13000ms → WEATHER              ← 4700ms gap = sentence boundary
 13500ms → NICE
 14000ms → SUN
 14500ms → WITHOUT-sth
 15150ms → CLOUDS

Dialog excerpt (two speakers, cumulative ms):
   500ms → [Hearing]: "Good morning, how are you?"
  3800ms → MORNING              ← 3300ms gap (deaf responds)
  4250ms → FINE
  4650ms → YOU
  8100ms → [Hearing]: "I am well, thanks for asking."
 11200ms → GOOD                 ← 3100ms gap
 11600ms → YOU
 12050ms → DO WHAT TODAY

## Output format

Return ONLY valid JSON with this exact structure.
For deaf turns with N gloss tokens: provide exactly N cumulative timestamps (ms from \
start of this group, t=0).
For hearing turns (no glosses): provide a single "timestamp_ms" value.
Preserve the original sentence order. The number of sentence entries in your output \
MUST match the number of sentences in the input.

{{
  "sentences": [
    {{"gloss_timestamps_ms": [500, 950, 1380, 1820, 2350]}},
    {{"gloss_timestamps_ms": [6200, 6650, 7100, 7480]}},
    {{"timestamp_ms": 9800}},
    ...
  ]
}}

## Discourse group to annotate

Type: {dtype}
Topic: {topic}

{formatted_sentences}

Annotate this group with realistic timestamps following the patterns above.
Return ONLY the JSON object, no explanation."""


def _format_group_for_timestamp_prompt(group: Dict) -> str:
    """Format a discourse group's sentences for the timestamp annotation prompt."""
    lines = []
    for i, sent in enumerate(group["sentences"]):
        speaker = sent.get("speaker", "deaf")
        if speaker == "hearing":
            lines.append(f"  Turn {i+1} [Hearing]: \"{sent['text']}\"")
        else:
            gloss = sent.get("gloss", "")
            n_tokens = len(gloss.split()) if gloss else 0
            lines.append(f"  Turn {i+1} [Deaf, {n_tokens} glosses]: {gloss}")
            lines.append(f"    English: \"{sent['text']}\"")
    return "\n".join(lines)


def _parse_timestamp_response(raw: str, group: Dict) -> Optional[List]:
    """
    Parse the LLM's timestamp annotation response.

    Returns a list of dicts, one per sentence:
      - Deaf turns: {"gloss_timestamps_ms": [int, ...]}
      - Hearing turns: {"timestamp_ms": int}
    Returns None on parse/validation failure.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.startswith("```")).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None

    sents_data = data.get("sentences", [])
    group_sents = group["sentences"]

    if len(sents_data) != len(group_sents):
        return None

    result = []
    prev_max_ts = 0

    for i, (sd, gs) in enumerate(zip(sents_data, group_sents)):
        speaker = gs.get("speaker", "deaf")

        if speaker == "hearing":
            ts = sd.get("timestamp_ms")
            if ts is None or not isinstance(ts, (int, float)):
                return None
            ts = int(ts)
            if ts <= prev_max_ts:
                ts = prev_max_ts + 2500  # auto-fix non-monotonic
            result.append({"timestamp_ms": ts})
            prev_max_ts = ts
        else:
            timestamps = sd.get("gloss_timestamps_ms", [])
            n_glosses = len(gs.get("gloss", "").split())
            if len(timestamps) != n_glosses:
                return None
            # Validate and fix monotonicity
            fixed = []
            for j, t in enumerate(timestamps):
                if not isinstance(t, (int, float)):
                    return None
                t = int(t)
                if t <= prev_max_ts:
                    t = prev_max_ts + (350 if j > 0 else 500)
                fixed.append(t)
                prev_max_ts = t
            # Validate plausible intervals
            for j in range(1, len(fixed)):
                interval = fixed[j] - fixed[j - 1]
                if interval < 100 or interval > 2000:
                    return None  # implausible within-sentence gap
            result.append({"gloss_timestamps_ms": fixed})

    return result


async def annotate_discourse_timestamps(
    input_path: Optional[Path] = None,
    backend_name: str = "gemini",
) -> int:
    """
    Annotate existing discourse groups with per-gloss timestamps via LLM.

    Processes one group per API call. Updates the JSONL in-place: each sentence
    gets a "gloss_timestamps_ms" (deaf) or "timestamp_ms" (hearing) field added.
    Groups that already have timestamps are skipped.

    Args:
        input_path:   Path to the discourse JSONL. Defaults to data/synthetic_discourse.jsonl.
        backend_name: LLM backend for annotation (gemini recommended).

    Returns:
        Number of groups newly annotated.
    """
    input_path = input_path or (cfg.paths.data / "synthetic_discourse.jsonl")
    if not input_path.exists():
        logger.error(f"Discourse file not found: {input_path}")
        return 0

    # Load all groups
    all_groups = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_groups.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    logger.info(f"Loaded {len(all_groups)} discourse groups from {input_path}")

    # Check which groups already have timestamps
    def _has_timestamps(group: Dict) -> bool:
        for sent in group.get("sentences", []):
            if sent.get("speaker") == "hearing":
                if "timestamp_ms" not in sent:
                    return False
            else:
                if "gloss_timestamps_ms" not in sent:
                    return False
        return True

    to_annotate = [(i, g) for i, g in enumerate(all_groups) if not _has_timestamps(g)]
    if not to_annotate:
        logger.info("All groups already have timestamps. Nothing to do.")
        return 0
    logger.info(f"{len(to_annotate)} groups need timestamp annotation.")

    backend = _load_backend(backend_name)
    annotated = 0
    failed = 0

    for batch_idx, (group_idx, group) in enumerate(to_annotate):
        formatted = _format_group_for_timestamp_prompt(group)
        prompt = _TIMESTAMP_PROMPT_TEMPLATE.format(
            dtype=group.get("type", "monologue"),
            topic=group.get("topic", ""),
            formatted_sentences=formatted,
        )

        retries = 0
        success = False
        while retries < 3 and not success:
            try:
                raw = await _raw_generate(backend, prompt)
                parsed = _parse_timestamp_response(raw, group)
                if parsed is not None:
                    # Merge timestamps into the group
                    for sent, ts_data in zip(group["sentences"], parsed):
                        sent.update(ts_data)
                    annotated += 1
                    success = True
                else:
                    retries += 1
                    logger.debug(f"  Group {group_idx}: parse failed (attempt {retries})")
            except Exception as e:
                retries += 1
                logger.warning(f"  Group {group_idx}: API error (attempt {retries}): {e}")

        if not success:
            # Fall back to rule-based timestamps
            _apply_rule_based_timestamps(group)
            annotated += 1
            failed += 1

        if (batch_idx + 1) % 25 == 0:
            # Periodic save
            _save_groups(all_groups, input_path)
            logger.info(
                f"  {batch_idx + 1}/{len(to_annotate)} annotated "
                f"(LLM: {annotated - failed}, fallback: {failed})"
            )

    # Final save
    _save_groups(all_groups, input_path)
    logger.info(
        f"Timestamp annotation complete: {annotated} groups annotated "
        f"({annotated - failed} LLM, {failed} rule-based fallback)"
    )
    await backend.close()
    return annotated


def _apply_rule_based_timestamps(group: Dict) -> None:
    """
    Apply deterministic rule-based timestamps as a fallback when LLM annotation fails.
    Uses realistic distributions based on real stream file analysis.
    """
    t = 500  # startup delay
    for i, sent in enumerate(group["sentences"]):
        if i > 0:
            t += random.randint(2500, 5000)

        speaker = sent.get("speaker", "deaf")
        if speaker == "hearing":
            sent["timestamp_ms"] = t
            t += random.randint(1500, 3000)
        else:
            glosses = sent.get("gloss", "").split()
            timestamps = []
            for j, gloss_tok in enumerate(glosses):
                if j > 0:
                    if '+' in gloss_tok or '-' in gloss_tok:
                        interval = random.randint(500, 650)
                    else:
                        interval = random.randint(300, 500)
                    t += interval
                timestamps.append(t)
            sent["gloss_timestamps_ms"] = timestamps


def _save_groups(groups: List[Dict], path: Path) -> None:
    """Write all groups back to JSONL (atomic overwrite)."""
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for g in groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ── Backend factory ───────────────────────────────────────────────────────────

def _load_backend(name: str):
    name = name.lower()
    if name == "groq":
        from src.backends.groq_backend import GroqBackend
        return GroqBackend()
    elif name == "gemini":
        from src.backends.gemini_backend import GeminiBackend
        return GeminiBackend()
    elif name in ("ollama", "gpt_oss"):
        from src.backends.ollama_backend import OllamaBackend
        return OllamaBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}. Choose from: groq, gemini, ollama")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic ASL gloss–text data via LLM."
    )
    p.add_argument(
        "--mode", type=str, default="pairs",
        choices=["pairs", "discourse", "timestamps", "vocab"],
        help=(
            "pairs: single gloss-text pairs (original mode); "
            "discourse: connected multi-sentence groups; "
            "timestamps: annotate existing discourse groups with per-gloss timestamps; "
            "vocab: extract and cache vocabulary only"
        ),
    )
    p.add_argument("--n",       type=int,   default=500,    help="Number of pairs/groups to generate")
    p.add_argument("--backend", type=str,   default="gemini", choices=["groq", "gemini", "ollama"])
    p.add_argument("--output",  type=str,   default=None,   help="Output JSONL file path")
    p.add_argument(
        "--sentences", type=int, default=5,
        help="Sentences per discourse group (discourse mode only)"
    )
    p.add_argument(
        "--oov-threshold", type=float, default=0.80,
        help="Min vocabulary coverage to accept a generated group (discourse mode)"
    )
    p.add_argument(
        "--aslg-top-n", type=int, default=400,
        help="Number of top ASLG roots to include in prompt vocabulary"
    )
    p.add_argument(
        "--force-rebuild-vocab", action="store_true",
        help="Rebuild vocabulary cache even if it already exists"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out  = Path(args.output) if args.output else None

    if args.mode == "vocab":
        full_forms, root_vocab, prompt_roots = build_gloss_vocabulary(
            aslg_top_n=args.aslg_top_n,
            force_rebuild=args.force_rebuild_vocab,
        )
        print(f"Vocabulary: {len(full_forms)} full forms, {len(root_vocab)} roots")
        print(f"Top {len(prompt_roots)} prompt roots: {' '.join(prompt_roots[:20])} ...")

    elif args.mode == "timestamps":
        asyncio.run(annotate_discourse_timestamps(
            input_path=out,
            backend_name=args.backend,
        ))

    elif args.mode == "discourse":
        asyncio.run(generate_discourse_groups(
            n_total=args.n,
            backend_name=args.backend,
            output_path=out,
            sentences_per_group=args.sentences,
            oov_threshold=args.oov_threshold,
            aslg_top_n=args.aslg_top_n,
        ))

    else:  # pairs
        asyncio.run(generate_synthetic_pairs(
            n_total=args.n,
            backend_name=args.backend,
            output_path=out,
        ))
