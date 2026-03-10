# bench-cleanser

Automated contamination detection for SWE-bench Verified. Identifies tasks where gold patches or fail-to-pass (F2P) tests exceed the problem description, producing unfair evaluation criteria for software engineering agents.

## Problem

SWE-bench Verified (500 tasks) is the primary benchmark for evaluating software engineering agents. However, some tasks have **contaminated evaluation criteria** that penalize agents for correctly solving the *described* problem:

- **Excess patches**: Gold patches include refactoring, style changes, or features not described in the problem statement
- **Excess tests**: F2P tests assert on behavior not described in the problem (off-topic assertions)
- **Vague specifications**: Problem statements too ambiguous to determine a unique correct solution

Agents that correctly solve the *described* problem may fail these tasks because evaluation criteria test *undescribed* behavior. bench-cleanser quantifies this contamination at per-hunk and per-assertion granularity.

## Architecture

### v2 Pipeline (Recommended): Intent-Matching

The v2 pipeline uses a 5-stage architecture that extracts ground-truth intent from the problem statement and matches it against the gold patch and F2P tests:

```
Stage 1:   PARSE              Extract diffs from gold patch + test patch
Stage 1.5: CODE VISITATION     Clone repo, extract full test/function source (optional)
Stage 2:   INTENT              Extract ground-truth intent from problem statement (LLM)
Stage 3:   STRUCTURAL DIFF     AST-level function/class change analysis
Stage 4:   INTENT MATCHING     Classify hunks + tests against intent (LLM)
Stage 5:   TRIAGE & REPORT     4-category scoring + actionable recommendations
```

### 4 Verdict Categories

| Category | Description | Recommended Action |
|----------|-------------|-------------------|
| **EXCESS_PATCH** | Gold patch includes changes beyond what the task describes | Filter UNRELATED hunks from evaluation |
| **EXCESS_TEST** | F2P tests verify behavior beyond the task description | Exclude OFF_TOPIC assertions from pass/fail |
| **VAGUE_SPEC** | Problem statement is ambiguous; multiple valid solutions exist | Interpret results with caution |
| **CLEAN** | No contamination detected | No action needed |

### Classification Granularity

Each gold patch hunk is classified as:
- **REQUIRED** -- Directly implements the described fix
- **ANCILLARY** -- Supports the fix but isn't described (imports, infrastructure)
- **UNRELATED** -- Changes behavior not described in the problem

Each F2P test is classified as:
- **ALIGNED** -- Test targets the described problem
- **TANGENTIAL** -- Test partially targets the problem
- **UNRELATED** -- Test doesn't target the described problem

Each test assertion is classified as:
- **ON_TOPIC** -- Assertion checks behavior described in the problem
- **OFF_TOPIC** -- Assertion checks behavior NOT described in the problem

### Scoring

```
excess_patch_score = (unrelated_hunks + 0.5 * ancillary_hunks) / total_hunks
excess_test_score  = (off_topic_assertions + unrelated_tests * avg_assertions) / total_assertions
combined_score     = 1 - (1 - excess_patch) * (1 - excess_test) * (1 - vague_spec)
```

Severity thresholds (configurable):
- **CLEAN**: combined < 0.2
- **MINOR**: 0.2 <= combined < 0.5
- **MODERATE**: 0.5 <= combined < 0.8
- **SEVERE**: combined >= 0.8

### v1 Pipeline (Legacy)

The v1 pipeline uses a 7-category overlapping taxonomy (OVERTEST, OVERPATCH, SNEAKY_TEST_MOD, SCOPE_CREEP, TEST_DESC_MISALIGN, CIRCULAR_DEPENDENCY, AMBIGUOUS_SPEC). It is retained for backward compatibility but v2 is recommended for all new analysis.

## Installation

```bash
git clone <repo-url>
cd bench-cleanser
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
# source .venv/bin/activate

pip install -r requirements.txt
```

### Requirements

- **Python 3.12+**
- **Azure OpenAI access** (CloudGPT) with Azure CLI authentication (`az login`)
- **rich** (optional) -- enhanced terminal progress display during batch runs

### Dependencies

| Package | Purpose |
|---------|---------|
| `datasets` | HuggingFace datasets for loading SWE-bench |
| `openai` | Azure OpenAI API client |
| `pyyaml` | Configuration file parsing |
| `tqdm` | Progress bars (fallback when rich not installed) |
| `azure-identity` | Azure AD authentication |
| `azure-identity-broker` | Token brokering for Azure |
| `msal` | Microsoft Authentication Library |
| `requests` | HTTP client |

## Configuration

Edit `config.yaml` to match your environment:

```yaml
llm:
  base_url: "https://cloudgpt-openai.azure-api.net/"
  api_version: "2025-04-01-preview"
  model: "gpt-5.2-20251211"
  max_tokens: 4096
  reasoning_effort: "high"        # Controls model reasoning depth
  max_concurrent_requests: 10
  retry_attempts: 7               # Exponential backoff retries on transient errors
  retry_delay_seconds: 5.0        # Base delay between retries (capped at 60s)

pipeline:
  concurrency: 3                  # Parallel task processing
  cache_dir: ".cache/llm_responses"
  output_dir: "output"

thresholds:
  clean_max: 0.2
  minor_max: 0.5
  moderate_max: 0.8

code_visitation:
  enabled: true                   # Clone repos for full source context
  repo_cache_dir: ".cache/repos"
  clone_timeout_seconds: 120
  max_source_context_lines: 200
```

### Authentication

bench-cleanser authenticates to Azure OpenAI via Azure CLI:

```bash
az login
```

## Usage

### Full batch analysis (v2 pipeline)

```bash
python run_pipeline.py --v2 --dataset verified --max-tasks 500
```

### Single task analysis

```bash
python run_pipeline.py --v2 --instance-id django__django-15916
```

### v1 pipeline (legacy)

```bash
python run_pipeline.py --dataset verified --max-tasks 100
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--v2` | Use v2 intent-matching pipeline (recommended) | v1 |
| `--config PATH` | Path to configuration YAML file | `config.yaml` |
| `--dataset {verified,lite,both}` | Which SWE-bench dataset(s) to analyze | `both` |
| `--max-tasks N` | Maximum tasks per dataset | `500` |
| `--instance-id ID` | Analyze a single instance (overrides `--dataset`) | -- |
| `--output DIR` | Override output directory | from config |
| `--concurrency N` | Parallel task processing | from config |
| `-v, --verbose` | Enable DEBUG logging | off |

## Output

### Per-task JSON Report

Each analyzed task produces a JSON report in `<output_dir>/reports/`:

```json
{
  "instance_id": "django__django-15916",
  "severity": "MODERATE",
  "combined_score": 0.55,
  "intent": {
    "core_requirement": "Allow ModelForm Meta to specify formfield_callback",
    "behavioral_contract": "BEFORE: ... AFTER: ...",
    "acceptance_criteria": [
      "modelform_factory preserves base form's callback"
    ],
    "out_of_scope": "Inheritance behavior of factory-produced forms",
    "ambiguity_score": 0.3
  },
  "excess_patch": {
    "score": 0.0,
    "total_hunks": 1,
    "required": 1,
    "ancillary": 0,
    "unrelated": 0,
    "hunks": [
      {
        "hunk_index": 0,
        "file": "django/forms/models.py",
        "verdict": "REQUIRED",
        "confidence": 0.95,
        "reason": "Directly implements the described fix"
      }
    ]
  },
  "excess_test": {
    "score": 0.33,
    "total_tests": 1,
    "aligned": 0,
    "tangential": 1,
    "unrelated": 0,
    "total_assertions": 3,
    "on_topic": 2,
    "off_topic": 1,
    "has_modified_tests": false,
    "tests": [
      {
        "test_id": "tests/forms/test_modelform.py::test_custom_callback",
        "test_name": "test_custom_callback",
        "intent_match": "TANGENTIAL",
        "assertions": [
          {"statement": "assertEqual(widget, Textarea)", "verdict": "ON_TOPIC"},
          {"statement": "assertEqual(callback_count, 1)", "verdict": "ON_TOPIC"},
          {"statement": "assertIsInstance(form, InheritedForm)", "verdict": "OFF_TOPIC"}
        ]
      }
    ]
  },
  "vague_spec": {
    "score": 0.3,
    "reasoning": "Mostly clear with minor edge cases undefined"
  },
  "recommendations": [
    "EXCESS_TEST: 1/3 assertions test behavior beyond problem scope."
  ]
}
```

### Summary CSV

Generated at `<output_dir>/summary.csv` with columns:

```
instance_id, severity, combined_score, excess_patch_score, excess_test_score,
vague_spec_score, patch_hunks_total, patch_required, patch_ancillary,
patch_unrelated, tests_total, tests_aligned, tests_tangential, tests_unrelated,
assertions_total, assertions_on_topic, assertions_off_topic,
has_modified_test, recommendations
```

### Summary Statistics

Generated at `<output_dir>/summary_stats.json` with severity distribution, mean/median combined scores, and per-category averages.

## Project Structure

```
bench_cleanser/
  __init__.py
  models.py                      # Data models and enums (v1 + v2)
  pipeline.py                    # Pipeline orchestrator (v1 + v2 batch/single)
  llm_client.py                  # Azure OpenAI client with retry and caching
  cache.py                       # Disk-based LLM response cache
  data_loader.py                 # SWE-bench dataset loading (HuggingFace)
  repo_manager.py                # Git repo cloning and management
  code_visitor.py                # Source code extraction from cloned repos
  static_analysis.py             # Python AST: imports, calls, assertions
  analysis/
    scope_analyzer.py            # Stage 2: Intent extraction (LLM)
    structural_diff.py           # Stage 3: AST-level structural analysis
    patch_analyzer.py            # Stage 4A: Patch-intent matching (LLM)
    test_analyzer.py             # Stage 4B: Test-intent matching (LLM)
    cross_ref.py                 # Cross-reference analysis (v1)
  classification/
    scorer.py                    # Stage 5: Scoring and report building (v1 + v2)
    taxonomy.py                  # Category/verdict definitions and thresholds
  parsing/
    patch_parser.py              # Unified diff parser (gold patch)
    test_parser.py               # Test patch parser + F2P matching
tests/
  test_scorer.py                 # Unit tests for scoring logic (v1 + v2)
run_pipeline.py                  # CLI entry point
config.yaml                      # Pipeline configuration
cloudgpt.py                      # Azure AD token provider
requirements.txt                 # Python dependencies
```

## Error Handling

bench-cleanser is designed to **fail loud** rather than produce incorrect results:

- **LLM failures**: All transient errors (HTTP 500, rate limits, timeouts, connection errors) are retried with exponential backoff (base delay 5s, capped at 60s, up to 7 attempts). Non-retryable errors propagate immediately.
- **No silent fallbacks**: If all LLM retries are exhausted, the pipeline raises `RuntimeError` rather than returning empty or degenerate results. Pipeline errors are surfaced as `SEVERE` reports with `PIPELINE_ERROR:` prefixes so they are visible in summary statistics.
- **SDK retry disabled**: The OpenAI SDK's built-in retry mechanism is disabled (`max_retries=0`) to prevent dual-layer retry storms. All retry logic is handled by bench-cleanser's own backoff implementation.
- **Caching**: Successful LLM responses are cached to disk. Subsequent runs reuse cached results, reducing API calls and enabling incremental reruns after transient failures.

## Testing

```bash
python -m pytest tests/ -v
```

## v1 vs v2 Comparison

| Aspect | v1 (7-category) | v2 (4-verdict) |
|--------|-----------------|----------------|
| Categories | 7 overlapping | 4 non-overlapping |
| Taxonomy | OVERTEST, OVERPATCH, SNEAKY_TEST_MOD, SCOPE_CREEP, TEST_DESC_MISALIGN, CIRCULAR_DEP, AMBIGUOUS_SPEC | EXCESS_PATCH, EXCESS_TEST, VAGUE_SPEC, CLEAN |
| Granularity | Hunk-level | Hunk + assertion-level |
| Ground truth | Scope analysis | Intent extraction with acceptance criteria |
| Structural analysis | Python AST only | Python AST with structural diff |
| False positives | Known issues (aligned SNEAKY_TEST_MOD, doc-file heuristic) | Reduced via intent matching |
| Output | Category confidence scores | Actionable per-hunk/per-assertion verdicts |
