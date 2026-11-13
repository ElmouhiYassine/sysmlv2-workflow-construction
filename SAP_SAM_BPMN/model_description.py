from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

if __package__:
    from .bpmn_parser import extract_bpmn
else:
    from bpmn_parser import extract_bpmn

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional at runtime for CSV loading only
    pd = None

DescriptionKind = Literal["task", "lane", "pool"]
OutputFormat = Literal["json", "text"]
LLMBackend = Callable[[str, Dict[str, Any]], Any]

TASK_TYPES = {
    "Task",
    "Function",
    "Subprocess",
    "CollapsedSubprocess",
    "StartNoneEvent",
    "EndNoneEvent",
}


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())



def sanitize_name(value: str, fallback: str = "UnnamedElement") -> str:
    text = normalize_text(value)
    if not text:
        return fallback
    text = re.sub(r"[^\w\s]", "", text)
    parts = text.split()
    return "".join(part.capitalize() for part in parts) or fallback



def is_extracted_bpmn(payload: Dict[str, Any]) -> bool:
    return all(key in payload for key in ("tasks", "nodes", "edges", "shape_index"))


@dataclass
class ElementDescription:
    id: str
    kind: DescriptionKind
    bpmn_type: str
    name: str
    documentation: str
    pool_id: Optional[str]
    lane_id: Optional[str]
    pool_name: str
    lane_name: str
    action_name: str
    source_text: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DescriptionRequest:
    include_kinds: Sequence[DescriptionKind] = ("task",)
    use_llm: bool = False
    include_prompt: bool = False
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str = "http://localhost:11434"
    llm_timeout: int = 90
    llm_retries: int = 1
    llm_temperature: float = 0.2
    style: str = "concise"


@dataclass
class DescriptionResult:
    elements: List[ElementDescription]
    by_id: Dict[str, str]
    by_action_name: Dict[str, str]
    summary: str
    model_description: str = ""
    prompt: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elements": [asdict(element) for element in self.elements],
            "by_id": self.by_id,
            "by_action_name": self.by_action_name,
            "summary": self.summary,
            "model_description": self.model_description or self.summary,
            "prompt": self.prompt,
            "meta": self.meta,
        }



def _task_elements(bpmn: Dict[str, Any]) -> Iterable[ElementDescription]:
    all_nodes: Dict[str, Dict[str, Any]] = {}
    all_nodes.update(bpmn.get("tasks", {}))
    all_nodes.update(bpmn.get("nodes", {}))

    for element in all_nodes.values():
        bpmn_type = element.get("task_type") or element.get("gateway_type") or element.get("node_type") or element.get("type")
        if bpmn_type not in TASK_TYPES:
            continue

        name = normalize_text(element.get("name"))
        documentation = normalize_text(element.get("documentation"))
        pool_name = normalize_text(element.get("pool_name")) or "UnspecifiedPool"
        lane_name = normalize_text(element.get("lane_name")) or "UnspecifiedLane"
        source_parts = [part for part in (name, documentation, lane_name, pool_name) if part]

        yield ElementDescription(
            id=element.get("id", ""),
            kind="task",
            bpmn_type=bpmn_type,
            name=name,
            documentation=documentation,
            pool_id=element.get("pool"),
            lane_id=element.get("lane"),
            pool_name=pool_name,
            lane_name=lane_name,
            action_name=sanitize_name(name, fallback=bpmn_type),
            source_text=" | ".join(source_parts),
            metadata={
                "original_type": element.get("type"),
            },
        )



def _lane_elements(bpmn: Dict[str, Any]) -> Iterable[ElementDescription]:
    for lane in bpmn.get("lanes", {}).values():
        name = normalize_text(lane.get("name"))
        pool_id = lane.get("pool")
        pool = bpmn.get("pools", {}).get(pool_id) or bpmn.get("collapsed_pools", {}).get(pool_id) or {}
        pool_name = normalize_text(pool.get("name")) or "UnspecifiedPool"
        source_parts = [part for part in (name, pool_name) if part]

        yield ElementDescription(
            id=lane.get("id", ""),
            kind="lane",
            bpmn_type=lane.get("type", "Lane"),
            name=name,
            documentation="",
            pool_id=pool_id,
            lane_id=lane.get("id"),
            pool_name=pool_name,
            lane_name=name or "UnspecifiedLane",
            action_name=sanitize_name(name, fallback="Lane"),
            source_text=" | ".join(source_parts),
        )



def _pool_elements(bpmn: Dict[str, Any]) -> Iterable[ElementDescription]:
    all_pools: Dict[str, Dict[str, Any]] = {}
    all_pools.update(bpmn.get("pools", {}))
    all_pools.update(bpmn.get("collapsed_pools", {}))

    for pool in all_pools.values():
        name = normalize_text(pool.get("name"))
        lane_count = len(pool.get("lanes", []))
        task_count = len(pool.get("tasks", []))
        source_text = f"{name} | lanes={lane_count} | tasks={task_count}" if name else f"lanes={lane_count} | tasks={task_count}"

        yield ElementDescription(
            id=pool.get("id", ""),
            kind="pool",
            bpmn_type=pool.get("type", "Pool"),
            name=name,
            documentation="",
            pool_id=pool.get("id"),
            lane_id=None,
            pool_name=name or "UnspecifiedPool",
            lane_name="",
            action_name=sanitize_name(name, fallback="Pool"),
            source_text=source_text,
            metadata={
                "lane_count": lane_count,
                "task_count": task_count,
            },
        )



def collect_describable_elements(
    bpmn: Dict[str, Any],
    include_kinds: Sequence[DescriptionKind] = ("task",),
) -> List[ElementDescription]:
    include = set(include_kinds)
    elements: List[ElementDescription] = []

    if "task" in include:
        elements.extend(_task_elements(bpmn))
    if "lane" in include:
        elements.extend(_lane_elements(bpmn))
    if "pool" in include:
        elements.extend(_pool_elements(bpmn))

    return elements



def describe_element_rule_based(element: ElementDescription) -> str:
    if element.kind == "task":
        if element.documentation and element.name and element.documentation.lower() != element.name.lower():
            return f"{element.name}. {element.documentation}"
        if element.documentation:
            return element.documentation
        if element.name:
            return f"{element.name} is performed by {element.lane_name} in the {element.pool_name} process."
        return f"A {element.bpmn_type} occurs in the {element.pool_name} process."

    if element.kind == "lane":
        if element.name:
            return f"{element.name} is a responsibility lane within the {element.pool_name} process."
        return f"An unnamed lane exists within the {element.pool_name} process."

    if element.kind == "pool":
        lane_count = element.metadata.get("lane_count", 0)
        task_count = element.metadata.get("task_count", 0)
        if element.name:
            return f"{element.name} is a BPMN pool containing {lane_count} lane(s) and {task_count} task-like element(s)."
        return f"An unnamed BPMN pool contains {lane_count} lane(s) and {task_count} task-like element(s)."

    return element.source_text or f"BPMN element {element.id}"



def _resolve_shape_info(bpmn: Dict[str, Any], shape_id: str) -> Dict[str, str]:
    tasks = bpmn.get("tasks", {})
    nodes = bpmn.get("nodes", {})
    shape_index = bpmn.get("shape_index", {})

    if shape_id in tasks:
        t = tasks[shape_id]
        return {
            "id": shape_id,
            "type": normalize_text(t.get("task_type") or t.get("type") or "Task"),
            "name": normalize_text(t.get("name") or shape_id),
            "documentation": normalize_text(t.get("documentation")),
            "pool_name": normalize_text(t.get("pool_name")) or "UnspecifiedPool",
            "lane_name": normalize_text(t.get("lane_name")) or "UnspecifiedLane",
        }

    if shape_id in nodes:
        n = nodes[shape_id]
        return {
            "id": shape_id,
            "type": normalize_text(n.get("gateway_type") or n.get("node_type") or n.get("type") or "Node"),
            "name": normalize_text(n.get("name") or shape_id),
            "documentation": normalize_text(n.get("documentation")),
            "pool_name": normalize_text(n.get("pool_name")) or "UnspecifiedPool",
            "lane_name": normalize_text(n.get("lane_name")) or "UnspecifiedLane",
        }

    idx = shape_index.get(shape_id, {})
    return {
        "id": shape_id,
        "type": normalize_text(idx.get("type") or "Unknown"),
        "name": normalize_text(idx.get("name") or shape_id),
        "documentation": "",
        "pool_name": "UnspecifiedPool",
        "lane_name": "UnspecifiedLane",
    }


def _collect_model_prompt_context(bpmn: Dict[str, Any]) -> Dict[str, Any]:
    all_pools = {}
    all_pools.update(bpmn.get("pools", {}))
    all_pools.update(bpmn.get("collapsed_pools", {}))
    lanes = bpmn.get("lanes", {})

    task_items = []
    for tid, t in bpmn.get("tasks", {}).items():
        task_items.append(
            {
                "id": tid,
                "name": normalize_text(t.get("name") or tid),
                "type": normalize_text(t.get("task_type") or "Task"),
                "documentation": normalize_text(t.get("documentation")),
                "pool_name": normalize_text(t.get("pool_name")) or "UnspecifiedPool",
                "lane_name": normalize_text(t.get("lane_name")) or "UnspecifiedLane",
            }
        )

    flow_items = []
    for eid, edge in bpmn.get("edges", {}).items():
        if edge.get("type") != "SequenceFlow":
            continue
        src = edge.get("src")
        tgt = edge.get("tgt")
        if not src or not tgt:
            continue

        src_info = _resolve_shape_info(bpmn, src)
        tgt_info = _resolve_shape_info(bpmn, tgt)

        edge_name = normalize_text(edge.get("name"))
        props = edge.get("props") if isinstance(edge.get("props"), dict) else {}
        cond_expr = normalize_text(props.get("conditionexpression"))
        cond_type = normalize_text(props.get("conditiontype"))
        condition = cond_expr or edge_name

        flow_items.append(
            {
                "id": eid,
                "src_id": src,
                "tgt_id": tgt,
                "src_name": src_info["name"],
                "tgt_name": tgt_info["name"],
                "src_type": src_info["type"],
                "tgt_type": tgt_info["type"],
                "src_pool": src_info["pool_name"],
                "tgt_pool": tgt_info["pool_name"],
                "condition": condition,
                "condition_type": cond_type,
            }
        )

    return {
        "pool_names": [normalize_text(p.get("name") or pid) for pid, p in all_pools.items()],
        "lane_names": [normalize_text(l.get("name") or lid) for lid, l in lanes.items()],
        "tasks": task_items,
        "sequence_flows": flow_items,
    }


def build_llm_prompt(bpmn: Dict[str, Any], elements: Sequence[ElementDescription], style: str = "concise") -> str:
    style_map = {
        "concise": "Use short, direct sentences (1-2 sentences per element).",
        "narrative": "Use fluent narrative style with natural transitions while staying factual.",
        "formal": "Use formal professional business language.",
        "functional": "Focus on operational purpose, actor responsibility, and expected outcome.",
    }
    style_instruction = style_map.get(style, style_map["concise"])

    context = _collect_model_prompt_context(bpmn)

    lines = [
        "You are an expert business process analyst.",
        "Generate one coherent natural-language description for the whole BPMN model.",
        "Describe the end-to-end flow, decisions, success/failure paths, and handoffs between pools/lanes.",
        "Return JSON only with this structure:",
        '{"model_description": "<single paragraph description of the whole model>", "summary": "<short one-sentence summary>"}',
        "Use only the provided BPMN data (tasks, documentation, pools, lanes, sequence flows, conditions).",
        "Do not invent missing steps or actors.",
        style_instruction,
        "",
        "Target style example (follow this storytelling quality while staying faithful to current data):",
        "A candidate enters the recruitment process and first undergoes an HR interview. If the HR interview is unsuccessful, the candidate is rejected; otherwise, the process continues with a technical interview by the IT team. If the technical evaluation fails, the candidate is technically rejected; if it succeeds, a security check is performed. Candidates who pass the security check proceed to contract validation by HR, after which they are either hired if the contract is approved or rejected otherwise.",
        "",
        "Model context:",
        "Pools: " + ", ".join(context["pool_names"] or ["(none)"]),
        "Lanes: " + ", ".join(context["lane_names"] or ["(none)"]),
        "",
        "Tasks and events:",
    ]

    for task in context["tasks"]:
        lines.append(
            f"- id={task['id']} | type={task['type']} | name={task['name']} | documentation={task['documentation']} | "
            f"pool={task['pool_name']} | lane={task['lane_name']}"
        )

    lines.append("")
    lines.append("Sequence flows:")
    for flow in context["sequence_flows"]:
        cond = flow["condition"] or "(no condition)"
        lines.append(
            f"- {flow['src_name']} ({flow['src_type']}, {flow['src_pool']}) -> {flow['tgt_name']} ({flow['tgt_type']}, {flow['tgt_pool']}) | condition={cond}"
        )

    lines.append("")
    lines.append("Also fill 'summary' with one compact sentence.")

    return "\n".join(lines)



def _coerce_llm_output(raw_output: Any, valid_ids: Iterable[str]) -> Dict[str, Any]:
    if isinstance(raw_output, str):
        raw_output = _parse_llm_json_text(raw_output)

    if not isinstance(raw_output, dict):
        raise ValueError("LLM output must be a dict or a JSON string.")

    by_id = raw_output.get("by_id", {})
    if by_id is None:
        by_id = {}
    if not isinstance(by_id, dict):
        raise ValueError("LLM output field 'by_id' must be a dict when provided.")

    allowed_ids = set(valid_ids)
    clean_by_id = {
        key: normalize_text(value)
        for key, value in by_id.items()
        if key in allowed_ids and normalize_text(value)
    }

    return {
        "by_id": clean_by_id,
        "model_description": normalize_text(raw_output.get("model_description")),
        "summary": normalize_text(raw_output.get("summary")),
    }


def _parse_llm_json_text(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("LLM output was empty.")

    # 1) direct JSON output
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) fenced markdown block
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?})\s*```", text, flags=re.IGNORECASE)
    if fenced:
        obj = json.loads(fenced.group(1))
        if isinstance(obj, dict):
            return obj

    # 3) best-effort: extract first JSON object span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj

    raise ValueError("LLM output is not valid JSON.")


def call_ollama_backend(
    prompt: str,
    context: Dict[str, Any],
    *,
    model: str,
    base_url: str,
    timeout: int,
    temperature: float,
) -> Dict[str, Any]:
    del context  # prompt already contains the necessary context payload

    endpoint = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(temperature),
        },
    }

    req = urllib_request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as exc:
        details = ""
        try:
            details = exc.read().decode("utf-8", errors="replace")
        except Exception:
            details = ""
        if details:
            raise RuntimeError(f"Ollama HTTP {exc.code}: {details}") from exc
        raise RuntimeError(f"Ollama HTTP {exc.code}: {exc.reason}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    response_payload = json.loads(body)
    if not isinstance(response_payload, dict):
        raise RuntimeError("Unexpected Ollama response type.")

    raw_text = response_payload.get("response", "")
    return _parse_llm_json_text(raw_text)


def build_open_source_llm_backend(request: DescriptionRequest) -> Optional[LLMBackend]:
    provider = (request.llm_provider or "").strip().lower()
    if provider != "ollama":
        return None

    def _backend(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return call_ollama_backend(
            prompt,
            context,
            model=request.llm_model,
            base_url=request.llm_base_url,
            timeout=request.llm_timeout,
            temperature=request.llm_temperature,
        )

    return _backend



def build_summary(elements: Sequence[ElementDescription], by_id: Dict[str, str]) -> str:
    task_count = sum(1 for element in elements if element.kind == "task")
    lane_count = sum(1 for element in elements if element.kind == "lane")
    pool_count = sum(1 for element in elements if element.kind == "pool")

    pool_names = sorted({element.pool_name for element in elements if element.pool_name})
    highlights = []
    for element in elements:
        if element.kind != "task":
            continue
        desc = by_id.get(element.id)
        if desc:
            highlights.append(desc)
        if len(highlights) == 3:
            break

    parts = [f"The BPMN model contains {task_count} task-like element(s), {lane_count} lane(s), and {pool_count} pool(s)."]
    if pool_names:
        parts.append("Pools: " + ", ".join(pool_names) + ".")
    if highlights:
        parts.append("Key activities: " + " ".join(highlights))
    return " ".join(parts)


def build_fallback_model_description(bpmn: Dict[str, Any]) -> str:
    context = _collect_model_prompt_context(bpmn)
    if not context["tasks"]:
        return "The BPMN model does not contain identifiable task-like elements to narrate."

    starts = [t for t in context["tasks"] if t["type"] == "StartNoneEvent"]
    ends = [t for t in context["tasks"] if t["type"] == "EndNoneEvent"]
    first = starts[0] if starts else context["tasks"][0]

    parts = [
        f"The process starts with {first['name']} in {first['pool_name']} ({first['lane_name']})."
    ]

    decision_flows = [f for f in context["sequence_flows"] if f["condition"]]
    if decision_flows:
        snippets = []
        for flow in decision_flows[:4]:
            snippets.append(
                f"If '{flow['condition']}', the flow moves from {flow['src_name']} to {flow['tgt_name']}"
            )
        parts.append(". ".join(snippets) + ".")

    if ends:
        end_names = ", ".join(e["name"] for e in ends)
        parts.append(f"The process can terminate at {end_names}.")

    return " ".join(parts)



def describe_bpmn(
    model_or_bpmn: Dict[str, Any],
    llm_backend: Optional[LLMBackend] = None,
    request: Optional[DescriptionRequest] = None,
) -> DescriptionResult:
    request = request or DescriptionRequest()
    bpmn = model_or_bpmn if is_extracted_bpmn(model_or_bpmn) else extract_bpmn(model_or_bpmn)
    elements = collect_describable_elements(bpmn, include_kinds=request.include_kinds)

    by_id = {element.id: describe_element_rule_based(element) for element in elements}
    model_description = build_fallback_model_description(bpmn)
    prompt = None
    provider = "rule_based"
    llm_attempted = False
    llm_success = False
    fallback_reason = ""
    llm_error = ""

    resolved_llm_backend = llm_backend or build_open_source_llm_backend(request)

    if request.use_llm and resolved_llm_backend is not None:
        llm_attempted = True
        prompt = build_llm_prompt(bpmn, elements, style=request.style)
        last_exc: Optional[Exception] = None
        for _ in range(max(1, request.llm_retries + 1)):
            try:
                llm_raw = resolved_llm_backend(prompt, {"elements": [asdict(e) for e in elements]})
                llm_data = _coerce_llm_output(llm_raw, by_id.keys())
                by_id.update(llm_data["by_id"])
                provider = f"llm:{request.llm_provider}:{request.llm_model}"
                llm_success = True
                model_description = llm_data["model_description"] or llm_data["summary"] or model_description
                summary = llm_data["summary"] or model_description
                break
            except Exception as exc:
                last_exc = exc
        else:
            provider = f"rule_based_after_llm_error:{type(last_exc).__name__ if last_exc else 'UnknownError'}"
            fallback_reason = "llm_error"
            llm_error = str(last_exc or "unknown LLM error")
            summary = model_description
    else:
        summary = model_description
        if request.include_prompt:
            prompt = build_llm_prompt(bpmn, elements, style=request.style)
        if request.use_llm and resolved_llm_backend is None:
            fallback_reason = "unsupported_llm_provider"

    by_action_name = {element.action_name: by_id[element.id] for element in elements}

    completed_elements: List[ElementDescription] = []
    for element in elements:
        completed_elements.append(
            ElementDescription(
                id=element.id,
                kind=element.kind,
                bpmn_type=element.bpmn_type,
                name=element.name,
                documentation=element.documentation,
                pool_id=element.pool_id,
                lane_id=element.lane_id,
                pool_name=element.pool_name,
                lane_name=element.lane_name,
                action_name=element.action_name,
                source_text=element.source_text,
                description=by_id[element.id],
                metadata=element.metadata,
            )
        )

    return DescriptionResult(
        elements=completed_elements,
        by_id=by_id,
        by_action_name=by_action_name,
        summary=summary,
        model_description=model_description,
        prompt=prompt if request.include_prompt else None,
        meta={
            "provider": provider,
            "element_count": len(completed_elements),
            "included_kinds": list(request.include_kinds),
            "llm_enabled": request.use_llm,
            "llm_provider": request.llm_provider,
            "llm_model": request.llm_model,
            "llm_attempted": llm_attempted,
            "llm_success": llm_success,
            "fallback_reason": fallback_reason,
            "llm_error": llm_error,
            "style": request.style,
            "description_scope": "whole_model",
        },
    )



def pair_with_sysml(sysml_code: str, description_result: DescriptionResult) -> Dict[str, Any]:
    return {
        "sysml": sysml_code,
        "descriptions": description_result.to_dict(),
    }



def render_text_report(result: DescriptionResult) -> str:
    meta = result.meta or {}
    lines = [
        "# BPMN model description",
        "",
        "## Generation",
        f"provider: {meta.get('provider', 'unknown')}",
        f"llm_enabled: {meta.get('llm_enabled', False)}",
        f"llm_attempted: {meta.get('llm_attempted', False)}",
        f"llm_success: {meta.get('llm_success', False)}",
    ]
    if meta.get("fallback_reason"):
        lines.append(f"fallback_reason: {meta.get('fallback_reason')}")
    if meta.get("llm_error"):
        lines.append(f"llm_error: {meta.get('llm_error')}")

    lines.extend(["", "## Model Description", result.model_description or result.summary])

    lines.extend(["", "## Scope", "This description is generated for the whole BPMN model and can be paired with SysML output."])
    return "\n".join(lines)


def render_text_report_elements(result: DescriptionResult) -> str:
    lines = ["# BPMN element descriptions", "", "## Summary", result.summary, "", "## Elements"]
    for element in result.elements:
        lines.append(f"- [{element.kind}] {element.action_name} ({element.id}) :: {element.description}")
    return "\n".join(lines)



def load_model_from_json(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)



def load_model_from_csv(csv_path: Path, row_index: int = 0, column: str = "Model JSON") -> Dict[str, Any]:
    if pd is None:
        raise RuntimeError("pandas is required to load BPMN models from CSV files.")

    df = pd.read_csv(csv_path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index {row_index} is outside the CSV bounds (0..{len(df) - 1}).")
    if column not in df.columns:
        raise KeyError(f"Column '{column}' was not found in the CSV file.")

    return json.loads(df.loc[row_index, column])



def parse_kind_option(kind_option: str) -> Sequence[DescriptionKind]:
    if kind_option == "tasks":
        return ("task",)
    if kind_option == "structure":
        return ("lane", "pool")
    return ("task", "lane", "pool")


DEFAULT_CSV_COLUMN = "Model JSON"
DEFAULT_MODE = "whole_model"
DEFAULT_KIND = "tasks"
DEFAULT_OUTPUT_FORMAT = "text"
DEFAULT_INCLUDE_PROMPT = False
DEFAULT_USE_LLM = True
DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_LLM_MODEL = "qwen2.5:7b"
DEFAULT_LLM_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_TIMEOUT = 90
DEFAULT_LLM_RETRIES = 1
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_STYLE = "narrative"


def run_description(csv_path: str, row_index: int = 0) -> str:
    model = load_model_from_csv(Path(csv_path), row_index=row_index, column=DEFAULT_CSV_COLUMN)

    result = describe_bpmn(
        model,
        request=DescriptionRequest(
            include_kinds=parse_kind_option(DEFAULT_KIND),
            include_prompt=DEFAULT_INCLUDE_PROMPT,
            use_llm=DEFAULT_USE_LLM,
            llm_provider=DEFAULT_LLM_PROVIDER,
            llm_model=DEFAULT_LLM_MODEL,
            llm_base_url=DEFAULT_LLM_BASE_URL,
            llm_timeout=DEFAULT_LLM_TIMEOUT,
            llm_retries=max(0, DEFAULT_LLM_RETRIES),
            llm_temperature=DEFAULT_LLM_TEMPERATURE,
            style=DEFAULT_STYLE,
        ),
    )

    if DEFAULT_OUTPUT_FORMAT == "json":
        payload = result.to_dict()
        payload["mode"] = DEFAULT_MODE
        return json.dumps(payload, indent=2, ensure_ascii=False)

    if DEFAULT_MODE == "elements":
        return render_text_report_elements(result)
    return render_text_report(result)



def main(csv_path: str, row_index: int = 0, output_path: Optional[str] = None) -> str:
    rendered = run_description(csv_path=csv_path, row_index=row_index)

    if output_path:
        Path(output_path).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)

    return rendered


if __name__ == "__main__":
    # Direct call style (no CLI args): update these values in code when needed.
    main(csv_path="sapsam_sysml.csv", row_index=0)
