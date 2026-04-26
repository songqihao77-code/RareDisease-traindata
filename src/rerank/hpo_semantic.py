from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


HPO_ONTOLOGY_CANDIDATES = (
    "raw_data/hp.json",
    "data/raw_data/hp-base.json",
    "data/raw_data/hp-base.obo",
    "raw_data/hp.obo",
)


def normalize_hpo_id(value: str) -> str:
    text = str(value).strip()
    if text.startswith("http://purl.obolibrary.org/obo/HP_"):
        return text.rsplit("/", 1)[-1].replace("_", ":")
    if text.startswith("HP_"):
        return text.replace("_", ":")
    return text


def find_hpo_ontology(project_root: Path, explicit_path: Path | None = None) -> Path | None:
    if explicit_path is not None:
        return explicit_path if explicit_path.is_file() else None
    for rel_path in HPO_ONTOLOGY_CANDIDATES:
        path = project_root / rel_path
        if path.is_file():
            return path
    return None


def _parse_json_edges(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    graphs = payload.get("graphs", [])
    parent_map: dict[str, set[str]] = defaultdict(set)
    for graph in graphs:
        for edge in graph.get("edges", []):
            predicate = str(edge.get("pred", ""))
            if not predicate.endswith(("is_a", "subClassOf")) and "BFO_0000050" not in predicate:
                continue
            child = normalize_hpo_id(str(edge.get("sub", "")))
            parent = normalize_hpo_id(str(edge.get("obj", "")))
            if child.startswith("HP:") and parent.startswith("HP:"):
                parent_map[child].add(parent)
    return parent_map


def _parse_obo_edges(path: Path) -> dict[str, set[str]]:
    parent_map: dict[str, set[str]] = defaultdict(set)
    current_id: str | None = None
    in_term = False
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line == "[Term]":
            in_term = True
            current_id = None
            continue
        if line.startswith("[") and line != "[Term]":
            in_term = False
            current_id = None
            continue
        if not in_term:
            continue
        if line.startswith("id: HP:"):
            current_id = normalize_hpo_id(line.split("id:", 1)[1].strip())
            parent_map.setdefault(current_id, set())
        elif current_id is not None and line.startswith("is_a: HP:"):
            parent_id = normalize_hpo_id(line.split("is_a:", 1)[1].split("!", 1)[0].strip())
            parent_map[current_id].add(parent_id)
    return parent_map


def load_parent_map(path: Path) -> dict[str, set[str]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _parse_json_edges(path)
    if suffix == ".obo":
        return _parse_obo_edges(path)
    raise ValueError(f"Unsupported HPO ontology file type: {path}")


def build_ancestor_map(parent_map: dict[str, set[str]]) -> dict[str, set[str]]:
    ancestors: dict[str, set[str]] = {}
    for term_id in parent_map:
        seen: set[str] = set()
        queue: deque[str] = deque(parent_map.get(term_id, set()))
        while queue:
            parent = queue.popleft()
            if parent in seen:
                continue
            seen.add(parent)
            queue.extend(parent_map.get(parent, set()) - seen)
        ancestors[term_id] = seen
    return ancestors


class HpoSemanticMatcher:
    def __init__(self, ancestors: dict[str, set[str]]) -> None:
        self.ancestors = ancestors
        self.available = bool(ancestors)

    @classmethod
    def from_project(cls, project_root: Path, explicit_path: Path | None = None) -> tuple["HpoSemanticMatcher", dict[str, Any]]:
        ontology_path = find_hpo_ontology(project_root, explicit_path)
        if ontology_path is None:
            return cls({}), {
                "available": False,
                "ontology_path": None,
                "warning": "No local HPO ontology file found; semantic features are set to 0.",
            }
        parent_map = load_parent_map(ontology_path)
        ancestors = build_ancestor_map(parent_map)
        return cls(ancestors), {
            "available": True,
            "ontology_path": str(ontology_path.resolve()),
            "num_terms": int(len(parent_map)),
            "num_terms_with_ancestors": int(sum(bool(values) for values in ancestors.values())),
        }

    def related(self, left: str, right: str) -> bool:
        left = normalize_hpo_id(left)
        right = normalize_hpo_id(right)
        if left == right:
            return True
        return right in self.ancestors.get(left, set()) or left in self.ancestors.get(right, set())

    def score(
        self,
        *,
        case_hpos: set[str],
        disease_hpos: set[str],
        hpo_specificity: dict[str, float],
    ) -> dict[str, float]:
        if not self.available or not case_hpos or not disease_hpos:
            return {
                "semantic_ic_overlap": 0.0,
                "semantic_coverage_score": 0.0,
            }

        matched_case: set[str] = set()
        matched_disease: set[str] = set()
        for case_hpo in case_hpos:
            for disease_hpo in disease_hpos:
                if self.related(case_hpo, disease_hpo):
                    matched_case.add(case_hpo)
                    matched_disease.add(disease_hpo)

        case_ic_total = float(sum(hpo_specificity.get(hpo_id, 0.0) for hpo_id in case_hpos))
        matched_ic_total = float(sum(hpo_specificity.get(hpo_id, 0.0) for hpo_id in matched_case))
        case_cov = len(matched_case) / len(case_hpos) if case_hpos else 0.0
        disease_cov = len(matched_disease) / len(disease_hpos) if disease_hpos else 0.0
        return {
            "semantic_ic_overlap": matched_ic_total / case_ic_total if case_ic_total > 0 else 0.0,
            "semantic_coverage_score": 0.5 * (case_cov + disease_cov),
        }

