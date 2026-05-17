"""Stage 2: Intent extraction and problem decomposition.

The LLM analyzes the problem_statement WITHOUT seeing the gold patch to:
1. Extract acceptance criteria (intent)
2. Decompose the problem into bug / suggested fix / legitimacy
3. Identify specific code entities (files, functions, classes, variables)

Uses structured output with strict JSON schema enforcement.
"""

from __future__ import annotations

import logging

from bench_cleanser.llm_client import LLMClient
from bench_cleanser.models import (
    IntentStatement,
    ProblemCodeContext,
    ProblemDecomposition,
    TaskRecord,
)
from bench_cleanser.prompts import load as _load_prompt
from bench_cleanser.schemas import IntentExtractionResponse

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = _load_prompt("intent_extraction")


def _build_user_prompt(
    record: TaskRecord,
    problem_code_context: ProblemCodeContext | None = None,
) -> str:
    parts = [
        f"Instance ID: {record.instance_id}",
        "",
        f"Problem Statement:\n{record.problem_statement}",
    ]
    if record.requirements:
        parts.append(f"\nRequirements:\n{record.requirements}")
    if record.interface:
        parts.append(f"\nInterface:\n{record.interface}")

    # Add pre-patch code context if available
    if problem_code_context:
        ctx_parts: list[str] = [
            "\n=== CODEBASE CONTEXT (pre-patch state, NOT the fix) ===",
            "The following source code is from BEFORE any patch was applied.",
            "Use it to understand what code the problem references.",
            "Do NOT use this to infer what the fix should look like.",
        ]
        if problem_code_context.relevant_directory_tree:
            ctx_parts.append(f"\nDirectory structure:\n{problem_code_context.relevant_directory_tree}")
        for path, content in problem_code_context.mentioned_file_contents.items():
            ctx_parts.append(f"\n--- {path} ---\n{content}")
        for name, source in problem_code_context.mentioned_entity_sources.items():
            ctx_parts.append(f"\n--- {name} ---\n{source}")
        parts.append("\n".join(ctx_parts))

    # Repo name last — avoid leading with a signal that triggers training data recall
    parts.append(f"\nRepository: {record.repo}")
    return "\n".join(parts) + "\n"


async def extract_intent(
    record: TaskRecord,
    llm: LLMClient,
    problem_code_context: ProblemCodeContext | None = None,
) -> IntentStatement:
    """Stage 2: extract intent and decompose problem statement.

    The LLM is given ONLY the problem_statement (never the gold patch).
    When available, pre-patch source code is included for grounding.
    Uses structured output with strict schema enforcement.
    Returns an IntentStatement with acceptance criteria and a
    ProblemDecomposition with entity tracking.
    """
    user_prompt = _build_user_prompt(record, problem_code_context)

    max_attempts = 3
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            result: IntentExtractionResponse = await llm.query_structured(
                INTENT_SYSTEM_PROMPT,
                user_prompt,
                IntentExtractionResponse,
                skip_cache=(attempt > 1),
            )

            if result.core_requirement and result.acceptance_criteria:
                break
            logger.warning(
                "Intent extraction attempt %d/%d for %s returned empty core fields",
                attempt, max_attempts, record.instance_id,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Intent extraction attempt %d/%d for %s failed: %s",
                attempt, max_attempts, record.instance_id, exc,
            )
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Intent extraction failed for {record.instance_id} "
                    f"after {max_attempts} attempts: {last_error}"
                )

    decomposition = ProblemDecomposition(
        bug_description=result.bug_description,
        suggested_fix=result.suggested_fix,
        legitimacy=result.legitimacy,
        mentioned_files=result.mentioned_files,
        mentioned_functions=result.mentioned_functions,
        mentioned_classes=result.mentioned_classes,
        mentioned_variables=result.mentioned_variables,
        mentioned_modules=result.mentioned_modules,
    )

    return IntentStatement(
        instance_id=record.instance_id,
        core_requirement=result.core_requirement,
        behavioral_contract=result.behavioral_contract,
        acceptance_criteria=result.acceptance_criteria,
        out_of_scope=result.out_of_scope,
        ambiguity_score=result.ambiguity_score,
        raw_llm_response=result.model_dump_json(),
        decomposition=decomposition,
    )
