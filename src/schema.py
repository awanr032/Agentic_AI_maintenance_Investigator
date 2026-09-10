"""Loads MaintIE's entity/relation taxonomy from data/scheme.json.

scheme.json is a source input (like gold_release.json / silver_release.json),
not pipeline-generated data, so it's read directly here rather than through
src/store.py — store.py owns pipeline *outputs* only (see its module
docstring).

Shape of scheme.json: `{"entity": [...], "relation": [...]}`, where each of
the two top-level lists holds a tree of nodes. Every node looks like:

    {"name": "Gas", "fullname": "PhysicalObject/Substance/Gas",
     "children": [...], ...other MaintIE-internal fields ignored here}

`fullname` is the exact dotted-path string used in the data's `"type"`
fields; `children` is a (possibly empty) list of the same shape.

Design note — resolved, not guessed at silently (per CLAUDE.md's instruction
for this project): design-spec.md §3.1 describes "224 leaf entity types",
but walking the actual tree shows 224 is the *total* node count (every
level, root classes down to leaves) — only 196 of those are true leaves
(nodes with no children). This matters because gold_release.json itself
labels entities with internal-category fullnames too, not always the
deepest leaf (checked directly: of 155 distinct entity types used in gold,
9 are non-leaf nodes like plain "PhysicalObject" or "State/UndesirableState",
0 are absent from the scheme tree). So the allowed label set handed to the
Extraction Agent must be every fullname in the tree, not leaves only —
`flatten_types()` below does not filter by leaf-ness for that reason. For
relations, MaintIE's own data happens to only ever use the 6 leaf relation
types (the one internal node, "hasParticipant", is never used bare) — but
`flatten_types()` doesn't special-case that; it stays generic and returns
everything, which happens to already match observed usage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

DEFAULT_SCHEME_PATH = Path(__file__).resolve().parent.parent / "data" / "scheme.json"


def load_scheme(path: Path | None = None) -> dict[str, Any]:
    """Load the raw `{"entity": [...], "relation": [...]}` tree from disk."""
    scheme_path = path or DEFAULT_SCHEME_PATH
    with scheme_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _walk(nodes: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield every node in a scheme sub-tree, itself and all descendants."""
    for node in nodes:
        yield node
        yield from _walk(node.get("children", []))


def flatten_types(nodes: list[dict[str, Any]]) -> list[str]:
    """All distinct `fullname` strings anywhere in a scheme sub-tree.

    Includes internal category nodes as well as leaves — see module
    docstring for why that's required, not an oversight. Pass
    `scheme["entity"]` or `scheme["relation"]`. Sorted for a stable,
    diffable prompt when this is injected into an LLM call.
    """
    return sorted({node["fullname"] for node in _walk(nodes)})


def entity_types(scheme: dict[str, Any] | None = None) -> list[str]:
    """The full allowed entity-type label set, e.g. for `extract()`'s
    `allowed_entity_types` argument (design-spec.md §5.1)."""
    return flatten_types((scheme or load_scheme())["entity"])


def relation_types(scheme: dict[str, Any] | None = None) -> list[str]:
    """The full allowed relation-type label set, e.g. for `extract()`'s
    `allowed_relation_types` argument (design-spec.md §5.1)."""
    return flatten_types((scheme or load_scheme())["relation"])


if __name__ == "__main__":
    # Quick sanity check when run directly: `py -m src.schema` from repo root.
    s = load_scheme()
    ents = entity_types(s)
    rels = relation_types(s)
    print(f"entity types: {len(ents)}")
    print(f"relation types: {len(rels)} -> {rels}")
