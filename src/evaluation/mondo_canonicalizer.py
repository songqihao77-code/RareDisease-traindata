from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_mondo_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("MONDO:"):
        return text
    if "MONDO_" in text:
        return "MONDO:" + text.rsplit("MONDO_", 1)[1].split("/", 1)[0]
    return text


def _basic_property_values(meta: dict[str, Any]) -> Iterable[dict[str, Any]]:
    values = meta.get("basicPropertyValues", []) if isinstance(meta, dict) else []
    return values if isinstance(values, list) else []


@dataclass(frozen=True)
class CanonicalizationResult:
    original_id: str
    canonical_id: str
    is_obsolete: bool
    replacement_id: str
    replacement_source: str
    label: str


class MondoCanonicalizer:
    """Resolve MONDO alternative IDs and curated obsolete replacements for relaxed metrics."""

    def __init__(
        self,
        *,
        names: dict[str, str] | None = None,
        obsolete_ids: set[str] | None = None,
        replacements: dict[str, str] | None = None,
        alt_to_primary: dict[str, str] | None = None,
        source_paths: list[str] | None = None,
    ) -> None:
        self.names = names or {}
        self.obsolete_ids = obsolete_ids or set()
        self.replacements = replacements or {}
        self.alt_to_primary = alt_to_primary or {}
        self.source_paths = source_paths or []

    @classmethod
    def load(
        cls,
        mondo_json_path: str | Path | None = None,
        obsolete_mondo_path: str | Path | None = None,
    ) -> "MondoCanonicalizer":
        names: dict[str, str] = {}
        obsolete_ids: set[str] = set()
        replacements: dict[str, str] = {}
        alt_to_primary: dict[str, str] = {}
        source_paths: list[str] = []

        json_path = Path(mondo_json_path) if mondo_json_path else cls.default_mondo_json_path()
        if json_path and json_path.is_file():
            source_paths.append(str(json_path.resolve()))
            graph = json.loads(json_path.read_text(encoding="utf-8"))["graphs"][0]
            for node in graph.get("nodes", []):
                mondo_id = _normalize_mondo_id(node.get("id"))
                if not mondo_id.startswith("MONDO:"):
                    continue
                if node.get("lbl"):
                    names[mondo_id] = str(node["lbl"])
                meta = node.get("meta", {}) or {}
                if meta.get("deprecated"):
                    obsolete_ids.add(mondo_id)
                for bpv in _basic_property_values(meta):
                    pred = str(bpv.get("pred", ""))
                    val = _normalize_mondo_id(bpv.get("val"))
                    if pred.endswith("hasAlternativeId") and val.startswith("MONDO:"):
                        alt_to_primary[val] = mondo_id
                    if meta.get("deprecated") and pred.endswith("IAO_0100001") and val.startswith("MONDO:"):
                        replacements[mondo_id] = val

        csv_path = Path(obsolete_mondo_path) if obsolete_mondo_path else cls.default_obsolete_csv_path()
        if csv_path and csv_path.is_file():
            source_paths.append(str(csv_path.resolve()))
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    old_id = _normalize_mondo_id(row.get("old_mondo_id"))
                    if not old_id.startswith("MONDO:"):
                        continue
                    obsolete_ids.add(old_id)
                    if row.get("old_label"):
                        names.setdefault(old_id, str(row["old_label"]))
                    replacement = _normalize_mondo_id(row.get("replaced_by_mondo_id"))
                    if replacement.startswith("MONDO:"):
                        replacements[old_id] = replacement
                        if row.get("replaced_by_label"):
                            names.setdefault(replacement, str(row["replaced_by_label"]))

        return cls(
            names=names,
            obsolete_ids=obsolete_ids,
            replacements=replacements,
            alt_to_primary=alt_to_primary,
            source_paths=source_paths,
        )

    @staticmethod
    def default_mondo_json_path() -> Path | None:
        candidates = [
            PROJECT_ROOT / "data" / "raw_data" / "mondo.json",
            PROJECT_ROOT / "data" / "raw_data" / "mondo-base.json",
        ]
        return next((path for path in candidates if path.is_file()), None)

    @staticmethod
    def default_obsolete_csv_path() -> Path | None:
        candidates = [
            PROJECT_ROOT / "v59_rare_disease_authoritative_diagnostic_guide_final" / "obsolete_mondo_final.csv",
        ]
        return next((path for path in candidates if path.is_file()), None)

    def canonicalize(self, mondo_id: Any) -> str:
        normalized = _normalize_mondo_id(mondo_id)
        primary = self.alt_to_primary.get(normalized, normalized)
        return self.replacements.get(primary, primary)

    def explain(self, mondo_id: Any) -> CanonicalizationResult:
        normalized = _normalize_mondo_id(mondo_id)
        primary = self.alt_to_primary.get(normalized, normalized)
        replacement = self.replacements.get(primary, "")
        canonical = replacement or primary
        return CanonicalizationResult(
            original_id=normalized,
            canonical_id=canonical,
            is_obsolete=primary in self.obsolete_ids or normalized in self.obsolete_ids,
            replacement_id=replacement,
            replacement_source="obsolete_replacement" if replacement else "",
            label=self.names.get(normalized) or self.names.get(primary) or "",
        )

    def canonicalize_many(self, mondo_ids: Iterable[Any]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for mondo_id in mondo_ids:
            canonical = self.canonicalize(mondo_id)
            if canonical and canonical not in seen:
                seen.add(canonical)
                out.append(canonical)
        return out

    def is_obsolete(self, mondo_id: Any) -> bool:
        normalized = _normalize_mondo_id(mondo_id)
        primary = self.alt_to_primary.get(normalized, normalized)
        return normalized in self.obsolete_ids or primary in self.obsolete_ids
