from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from .query_parser import QueryParseResult


class EvidenceFormatter:
    def build_bundle(
        self,
        query_info: QueryParseResult,
        events: Sequence[Dict[str, Any]],
        entities: Sequence[Dict[str, Any]],
        registry: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summary_stats = {
            "num_event_evidence": len(events),
            "num_entity_evidence": len(entities),
            "num_tracked_entities": len(registry),
            "matched_entity_ids": list(query_info.entity_ids),
            "matched_entity_tags": list(query_info.entity_tags),
            "matched_entity_labels": list(query_info.entity_labels),
        }
        evidence_json = {
            "query": query_info.query,
            "normalized_query": query_info.normalized_query,
            "subqueries": list(query_info.subqueries),
            "query_hints": query_info.to_dict(),
            "events": list(events),
            "entities": list(entities),
            "summary_stats": summary_stats,
        }
        evidence_text = self.format_evidence_text(evidence_json)
        bundle = dict(evidence_json)
        bundle["evidence_text"] = evidence_text
        bundle["evidence_json"] = evidence_json
        return bundle

    def format_evidence_text(self, bundle: Dict[str, Any]) -> str:
        lines: List[str] = [
            "=== STG Evidence Bundle ===",
            f"Query: {bundle['query']}",
            f"Normalized Query: {bundle['normalized_query']}",
            f"Subqueries: {', '.join(bundle.get('subqueries', [])) or '(none)'}",
        ]

        hints = bundle.get("query_hints", {})
        lines.append(
            "Hints: "
            f"entity_tags={hints.get('entity_tags', [])}, "
            f"entity_labels={hints.get('entity_labels', [])}, "
            f"relation_keywords={hints.get('relation_keywords', [])}, "
            f"temporal_keywords={hints.get('temporal_keywords', [])}, "
            f"intents={hints.get('query_intents', [])}"
        )
        lines.append("--- Event Evidence ---")
        events = bundle.get("events", [])
        if not events:
            lines.append("(no event evidence above threshold)")
        else:
            for rank, item in enumerate(events, start=1):
                lines.append(
                    f"{rank}. [{item.get('event_type', 'event')}] "
                    f"id={item.get('memory_id')} score={item.get('final_score', item.get('score', 0.0)):.3f} "
                    f"frames={item.get('frame_start')}-{item.get('frame_end')} "
                    f"entities={item.get('entity_tags', item.get('entities', []))} "
                    f"| {item.get('summary', '')}"
                )
        lines.append("--- Entity Evidence ---")
        entities = bundle.get("entities", [])
        if not entities:
            lines.append("(no entity evidence above threshold)")
        else:
            for rank, item in enumerate(entities, start=1):
                lines.append(
                    f"{rank}. [{item.get('entity_id', 'entity')}] "
                    f"id={item.get('memory_id')} score={item.get('final_score', item.get('score', 0.0)):.3f} "
                    f"frame={item.get('frame_index')} "
                    f"bbox={item.get('bbox')} "
                    f"| {item.get('description', '')}"
                )
        stats = bundle.get("summary_stats", {})
        lines.append("--- Summary Stats ---")
        lines.append(json.dumps(stats, ensure_ascii=False, sort_keys=True))
        return "\n".join(lines)

    def format_evidence_for_llm(
        self,
        bundle: Dict[str, Any],
        *,
        max_events: int = 8,
        max_entities: int = 4,
    ) -> Dict[str, Any]:
        events = []
        for item in bundle.get("events", [])[:max_events]:
            events.append(
                {
                    "memory_id": item.get("memory_id"),
                    "memory_type": item.get("memory_type"),
                    "event_type": item.get("event_type"),
                    "frame_start": item.get("frame_start"),
                    "frame_end": item.get("frame_end"),
                    "entities": item.get("entities", []),
                    "entity_tags": item.get("entity_tags", []),
                    "entity_labels": item.get("entity_labels", []),
                    "summary": item.get("summary", ""),
                    "confidence": item.get("confidence"),
                    "source": item.get("source"),
                    "score": item.get("final_score", item.get("score")),
                }
            )
        entities = []
        for item in bundle.get("entities", [])[:max_entities]:
            entities.append(
                {
                    "memory_id": item.get("memory_id"),
                    "memory_type": item.get("memory_type"),
                    "entity_id": item.get("entity_id"),
                    "tag": item.get("tag"),
                    "label": item.get("label"),
                    "frame_index": item.get("frame_index"),
                    "frame_start": item.get("frame_start"),
                    "frame_end": item.get("frame_end"),
                    "bbox": item.get("bbox"),
                    "attributes": item.get("attributes", []),
                    "relations": item.get("relations", []),
                    "total_displacement": item.get("total_displacement"),
                    "description": item.get("description", ""),
                    "confidence": item.get("confidence"),
                    "score": item.get("final_score", item.get("score")),
                }
            )
        return {
            "query": bundle.get("query"),
            "normalized_query": bundle.get("normalized_query"),
            "subqueries": bundle.get("subqueries", []),
            "summary_stats": bundle.get("summary_stats", {}),
            "events": events,
            "entities": entities,
        }

    def build_grounded_prompt(
        self,
        query: str,
        llm_evidence: Dict[str, Any],
    ) -> Dict[str, str]:
        system_prompt = (
            "You are a grounded spatio-temporal video QA assistant. "
            "Answer strictly from the provided STG evidence. "
            "If evidence is insufficient, say so explicitly. "
            "Return only JSON with this schema: "
            '{"answer": str, "sufficient_evidence": bool, "used_event_ids": [str], '
            '"used_entity_ids": [str], "short_rationale": str}.'
        )
        user_prompt = (
            "Question:\n"
            f"{query}\n\n"
            "Structured Evidence JSON:\n"
            f"{json.dumps(llm_evidence, ensure_ascii=False, indent=2)}\n\n"
            "Rules:\n"
            "1. Use only the evidence above.\n"
            "2. Cite memory IDs in used_event_ids and used_entity_ids.\n"
            "3. If the evidence cannot support a reliable answer, set sufficient_evidence=false.\n"
            "4. Return JSON only.\n"
        )
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}
