"""Load agent trajectories from various sources."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re

from bench_cleanser.trajectory.models import ActionType, TrajectoryAction, TrajectoryRecord

logger = logging.getLogger(__name__)


def load_from_jsonl(
    path: str | pathlib.Path,
    instance_ids: set[str] | None = None,
) -> list[TrajectoryRecord]:
    """Load trajectories from a local JSONL file.

    Each line should be a JSON object with at minimum:
    - instance_id
    - trajectory or actions (list of action dicts)
    - model_patch or final_patch
    """
    filepath = pathlib.Path(path)
    if not filepath.exists():
        logger.warning("Trajectory file not found: %s", filepath)
        return []

    records = []
    for line_num, line in enumerate(filepath.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            iid = data.get("instance_id", "")
            if instance_ids and iid not in instance_ids:
                continue
            records.append(TrajectoryRecord.from_dict(data))
        except Exception as exc:
            logger.warning("Failed to parse line %d of %s: %s", line_num, filepath, exc)

    logger.info("Loaded %d trajectories from %s", len(records), filepath)
    return records


def load_from_json_dir(
    dir_path: str | pathlib.Path,
    instance_ids: set[str] | None = None,
) -> list[TrajectoryRecord]:
    """Load trajectories from a directory of individual JSON files.

    Each JSON file should contain a single trajectory record.
    """
    dirpath = pathlib.Path(dir_path)
    if not dirpath.is_dir():
        logger.warning("Trajectory directory not found: %s", dirpath)
        return []

    records = []
    for json_file in dirpath.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            iid = data.get("instance_id", "")
            if instance_ids and iid not in instance_ids:
                continue
            records.append(TrajectoryRecord.from_dict(data))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", json_file, exc)

    logger.info("Loaded %d trajectories from %s", len(records), dirpath)
    return records


def load_from_huggingface(
    dataset_name: str,
    split: str = "train",
    instance_ids: set[str] | None = None,
    agent_name: str = "",
) -> list[TrajectoryRecord]:
    """Load trajectories from a HuggingFace dataset.

    Many SWE-bench agent repos upload trajectory datasets. This loader
    normalizes the common fields (instance_id, trajectory, model_patch).

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to load.
        instance_ids: Only include these instance IDs.
        agent_name: Override agent_name field (useful when dataset doesn't include it).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library required for HuggingFace loading")
        return []

    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as exc:
        logger.error("Failed to load HuggingFace dataset %s: %s", dataset_name, exc)
        return []

    records = []
    for row in ds:
        iid = row.get("instance_id", "")
        if instance_ids and iid not in instance_ids:
            continue

        # Normalize trajectory field
        trajectory = row.get("trajectory", [])
        if isinstance(trajectory, str):
            try:
                trajectory = json.loads(trajectory)
            except json.JSONDecodeError:
                trajectory = []

        data = {
            "instance_id": iid,
            "agent_name": agent_name or row.get("model_name_or_path", ""),
            "trajectory": trajectory,
            "model_patch": row.get("model_patch", ""),
            "resolved": row.get("resolved", False),
        }
        records.append(TrajectoryRecord.from_dict(data))

    logger.info(
        "Loaded %d trajectories from HuggingFace %s (split=%s)",
        len(records), dataset_name, split,
    )
    return records


def load_from_docent(
    collection_id: str,
    api_key: str,
    server_url: str = "https://api.docent.transluce.org",
    instance_ids: set[str] | None = None,
    model_name: str | None = None,
    dql_query: str | None = None,
) -> list[TrajectoryRecord]:
    """Load agent trajectories from Docent platform.

    Uses the docent-python SDK to query agent runs via DQL and fetch
    full trajectory data for each matching run.

    Args:
        collection_id: Docent collection UUID.
        api_key: Docent API key.
        server_url: Docent API server URL.
        instance_ids: Only include these instance IDs.
        model_name: Filter by model name in DQL query.
        dql_query: Custom DQL query (overrides default).
    """
    try:
        from docent import Docent
    except ImportError:
        logger.error("docent-python library required for Docent loading. pip install docent-python")
        return []

    client = Docent(api_key=api_key, api_url=server_url + "/rest")

    if dql_query is None:
        # Build default DQL query
        where_clauses = []
        if model_name:
            where_clauses.append(f"(ar.metadata_json->>'model_name' = '{model_name}')")
        where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        dql_query = f"""SELECT
  ar.id AS agent_run_id,
  ar.metadata_json->>'instance_id' AS metadata_instance_id,
  ar.metadata_json->>'model_name' AS metadata_model_name,
  ar.metadata_json->>'resolved' AS metadata_resolved,
  ar.metadata_json->>'turns' AS metadata_turns,
  ar.created_at AS created_at
FROM agent_runs ar
{where_str}
ORDER BY ar.created_at ASC"""

    logger.info("Executing DQL query on collection %s", collection_id)
    result = client.execute_dql(collection_id, dql_query)
    df = client.dql_result_to_df_experimental(result)

    logger.info("DQL returned %d agent runs", len(df))

    # Filter to target instance_ids first to avoid fetching unnecessary runs
    if instance_ids:
        filtered_rows = [(idx, row) for idx, row in df.iterrows()
                         if row.get("metadata_instance_id", "") in instance_ids]
    else:
        filtered_rows = list(df.iterrows())

    total_to_fetch = len(filtered_rows)
    logger.info("Fetching %d agent run transcripts from Docent", total_to_fetch)

    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        use_rich = True
    except ImportError:
        use_rich = False

    records: list[TrajectoryRecord] = []

    def _fetch_and_parse(row, progress=None, task_id=None):
        run_id = row.get("agent_run_id", "")
        iid = row.get("metadata_instance_id", "")

        resolved_str = str(row.get("metadata_resolved", "false")).lower()
        resolved = resolved_str in ("true", "1", "yes")
        agent_model = row.get("metadata_model_name", "")

        try:
            agent_run = client.get_agent_run(collection_id, run_id)
        except Exception as exc:
            logger.warning("Failed to fetch agent run %s: %s", run_id, exc)
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)
            return None

        # Parse transcript messages into TrajectoryActions
        # Docent SDK: agent_run.transcripts is a list of Transcript objects
        # Each Transcript has .messages (list of UserMessage/AssistantMessage)
        transcripts = getattr(agent_run, "transcripts", None) or []
        if transcripts and len(transcripts) > 0:
            messages = getattr(transcripts[0], "messages", []) or []
        else:
            # Fallback to older API patterns
            messages = getattr(agent_run, "transcript", None) or []
            if hasattr(messages, "messages"):
                messages = messages.messages
        if not isinstance(messages, list):
            messages = []

        actions = []
        final_patch = ""
        raw_messages = []

        for msg in messages:
            if isinstance(msg, dict):
                raw_messages.append(msg)
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                tool_name = block.get("name", "")
                                tool_input = json.dumps(block.get("input", {}))[:50000]
                                action_type = _map_tool_to_action_type(tool_name)
                                actions.append(TrajectoryAction(
                                    action_type=action_type,
                                    content=tool_input,
                                    tool_name=tool_name,
                                    role=role,
                                ))
                            elif block.get("type") == "tool_result":
                                result_content = block.get("content", "")
                                if isinstance(result_content, list):
                                    result_content = "\n".join(
                                        b.get("text", "") for b in result_content
                                        if isinstance(b, dict) and b.get("type") == "text"
                                    )
                                if actions:
                                    actions[-1].observation = str(result_content)[:50000]
                    content = "\n".join(text_parts)

                if role == "assistant" and content:
                    actions.append(TrajectoryAction(
                        action_type=ActionType.THINK,
                        content=content,
                        role=role,
                    ))
                elif role == "tool" and content:
                    if actions:
                        actions[-1].observation = str(content)[:50000]
                    else:
                        actions.append(TrajectoryAction(
                            action_type=ActionType.OTHER,
                            content=str(content),
                            role=role,
                        ))
            else:
                # Handle Docent Pydantic message objects (UserMessage, AssistantMessage)
                msg_role = getattr(msg, "role", "")
                msg_content = getattr(msg, "content", "") or ""
                msg_text = getattr(msg, "text", "") or ""
                tool_calls = getattr(msg, "tool_calls", None) or []
                tool_call_id = getattr(msg, "tool_call_id", None)

                raw_messages.append({
                    "role": msg_role,
                    "content": str(msg_content)[:2000] if msg_content else "",
                    "has_tool_calls": bool(tool_calls),
                    "tool_call_id": tool_call_id,
                })

                # Parse tool calls from assistant messages
                if tool_calls:
                    for tc in tool_calls:
                        tc_name = getattr(tc, "function", None)
                        if tc_name and hasattr(tc_name, "name"):
                            tool_name = tc_name.name
                            tool_args = getattr(tc_name, "arguments", "")
                        elif isinstance(tc, dict):
                            func = tc.get("function", {})
                            tool_name = func.get("name", "") if isinstance(func, dict) else str(func)
                            tool_args = func.get("arguments", "") if isinstance(func, dict) else ""
                        else:
                            tool_name = str(getattr(tc, "name", getattr(tc, "type", "unknown")))
                            tool_args = str(getattr(tc, "input", getattr(tc, "arguments", "")))

                        action_type = _map_tool_to_action_type(tool_name)
                        actions.append(TrajectoryAction(
                            action_type=action_type,
                            content=str(tool_args)[:50000],
                            tool_name=tool_name,
                            role=msg_role,
                        ))

                if msg_role == "assistant" and (msg_content or msg_text):
                    text = str(msg_content or msg_text)
                    if text.strip():
                        actions.append(TrajectoryAction(
                            action_type=ActionType.THINK,
                            content=text,
                            role=msg_role,
                        ))
                elif msg_role == "user" and tool_call_id and (msg_content or msg_text):
                    # Tool response (user message with tool_call_id = observation)
                    obs = str(msg_content or msg_text)[:50000]
                    if actions:
                        actions[-1].observation = obs
                    else:
                        actions.append(TrajectoryAction(
                            action_type=ActionType.OTHER,
                            content=obs,
                            role=msg_role,
                        ))
                elif msg_role == "user" and (msg_content or msg_text):
                    # Regular user observation (e.g., tool output)
                    obs = str(msg_content or msg_text)
                    if actions and not actions[-1].observation:
                        actions[-1].observation = obs[:50000]

        for action in reversed(actions):
            if action.action_type in (ActionType.EDIT, ActionType.WRITE) and action.content:
                final_patch = action.content
                break

        turns = 0
        try:
            turns = int(row.get("metadata_turns", 0))
        except (ValueError, TypeError):
            turns = len([a for a in actions if a.role == "assistant"])

        record = TrajectoryRecord(
            instance_id=iid,
            agent_name=agent_model,
            actions=actions,
            final_patch=final_patch,
            resolved=resolved,
            passed_tests=resolved,
            model_name=agent_model,
            total_tokens=0,
            turn_count=turns,
            raw_messages=raw_messages,
        )

        if progress is not None and task_id is not None:
            status = f"loaded:{len(records)+1} actions:{len(actions)} {iid[:40]}"
            progress.update(task_id, advance=1, status=status)

        return record

    if use_rich:
        console = Console()
        with Progress(
            SpinnerColumn(), TextColumn("[bold magenta]docent-fetch"),
            BarColumn(bar_width=40), MofNCompleteColumn(),
            TimeElapsedColumn(), TextColumn("[dim]{task.fields[status]}"),
            console=console,
        ) as progress:
            task_id = progress.add_task("Fetching", total=total_to_fetch, status="Starting...")
            for _, row in filtered_rows:
                result = _fetch_and_parse(row, progress, task_id)
                if result is not None:
                    records.append(result)
    else:
        for _, row in filtered_rows:
            result = _fetch_and_parse(row)
            if result is not None:
                records.append(result)

    logger.info("Loaded %d trajectories from Docent collection %s", len(records), collection_id)
    return records


def _map_tool_to_action_type(tool_name: str) -> ActionType:
    """Map Docent/Claude tool names to ActionType."""
    tool_name_lower = tool_name.lower()
    if any(t in tool_name_lower for t in ("edit", "replace", "patch")):
        return ActionType.EDIT
    if any(t in tool_name_lower for t in ("write", "create_file")):
        return ActionType.WRITE
    if any(t in tool_name_lower for t in ("read", "cat", "view")):
        return ActionType.READ
    if any(t in tool_name_lower for t in ("bash", "terminal", "execute", "run", "shell")):
        return ActionType.TERMINAL
    if any(t in tool_name_lower for t in ("search", "grep", "find", "glob")):
        return ActionType.SEARCH
    if any(t in tool_name_lower for t in ("browser", "web", "fetch")):
        return ActionType.BROWSE
    return ActionType.OTHER


def load_trajectories(
    source: str,
    instance_ids: set[str] | None = None,
    agent_name: str = "",
    hf_split: str = "train",
    api_key: str = "",
    model_filter: str = "",
) -> list[TrajectoryRecord]:
    """Auto-detect source type and load trajectories.

    Args:
        source: Path to JSONL file, JSON directory, HuggingFace dataset name,
            or Docent collection UUID.
        instance_ids: Only include these instance IDs.
        agent_name: Override agent name (for HuggingFace sources).
        hf_split: Split for HuggingFace datasets.
        api_key: API key for Docent loading (or set DOCENT_API_KEY env var).
        model_filter: Filter by model name (for Docent sources).
    """
    # Check if source looks like a Docent collection UUID
    if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", source):
        docent_key = api_key or os.environ.get("DOCENT_API_KEY", "")
        if not docent_key:
            logger.error("Docent API key required. Pass api_key or set DOCENT_API_KEY env var.")
            return []
        return load_from_docent(
            collection_id=source,
            api_key=docent_key,
            instance_ids=instance_ids,
            model_name=model_filter or None,
        )

    path = pathlib.Path(source)

    if path.is_file() and path.suffix in (".jsonl", ".json"):
        if path.suffix == ".jsonl":
            return load_from_jsonl(path, instance_ids)
        # Single JSON file — treat as single-record
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records = []
                for item in data:
                    iid = item.get("instance_id", "")
                    if instance_ids and iid not in instance_ids:
                        continue
                    records.append(TrajectoryRecord.from_dict(item))
                return records
            else:
                return [TrajectoryRecord.from_dict(data)]
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
            return []

    if path.is_dir():
        return load_from_json_dir(path, instance_ids)

    # Assume HuggingFace dataset name
    return load_from_huggingface(
        source, split=hf_split, instance_ids=instance_ids, agent_name=agent_name,
    )
