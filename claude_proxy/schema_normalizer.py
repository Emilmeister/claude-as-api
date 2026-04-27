"""Rewrite JSON Schemas so they avoid `oneOf` / `anyOf` constructs.

Why: Claude Code's structured-output pipeline injects internal tools into the Anthropic
API request body when the user-provided schema uses `oneOf` / `anyOf`. One of those
internal tools ships with a malformed `input_schema` and the API rejects the call with
`tools.N.custom.input_schema.type: Field required`. We side-step the entire injection
path by ensuring the schema we send is union-free.

Conversion strategy (best-effort, may relax strictness):

1. **Discriminated union** (every branch is an object containing the same `const` field):
   merge into a flat object whose discriminator becomes an `enum` and whose other
   properties are the union of all branch properties. Only the discriminator stays
   `required`; per-branch required fields are dropped because the model emits one
   branch's payload at a time and the missing branch's fields would otherwise fail.

2. **Heterogeneous object union** (all branches are objects but no shared discriminator):
   merge into a single object whose properties are the union of all branches', with no
   `required` array. Loses cross-branch exclusivity but keeps the type structure.

3. **Mixed or non-object union**: fall back to the first branch verbatim. The caller
   loses the alternation but the schema stays valid.

Recursion: every dict / list value in the tree is normalized, so nested unions are
handled too.
"""
from __future__ import annotations

from typing import Any


def normalize(schema: Any) -> Any:
    """Public entry: return a deep copy of `schema` with all unions rewritten."""
    return _walk(schema)


def _walk(node: Any) -> Any:
    if isinstance(node, list):
        return [_walk(x) for x in node]
    if not isinstance(node, dict):
        return node

    # Recurse into children first so nested unions get rewritten before this level
    # collapses them.
    rewritten = {k: _walk(v) for k, v in node.items()}

    for union_key in ("oneOf", "anyOf"):
        if union_key in rewritten:
            branches = rewritten.pop(union_key) or []
            merged = _merge_union(branches)
            rewritten = _combine(rewritten, merged)

    return rewritten


def _merge_union(branches: list) -> dict:
    branches = [b for b in branches if isinstance(b, dict)]
    if not branches:
        return {}

    if all(_looks_like_object(b) for b in branches):
        discriminator = _find_discriminator(branches)
        if discriminator:
            return _merge_with_discriminator(branches, discriminator)
        return _merge_object_branches(branches)

    # Heterogeneous: prefer first object-like branch, else first branch.
    for b in branches:
        if _looks_like_object(b):
            return b
    return branches[0]


def _looks_like_object(b: dict) -> bool:
    if b.get("type") == "object":
        return True
    # A schema with `properties` but no explicit type is conventionally an object.
    return "properties" in b and "type" not in b


def _find_discriminator(branches: list[dict]) -> str | None:
    """Return the name of a field that holds a `const` in every branch."""
    consts_per_branch: list[set[str]] = []
    for b in branches:
        props = b.get("properties") or {}
        consts = {
            name for name, spec in props.items()
            if isinstance(spec, dict) and "const" in spec
        }
        if not consts:
            return None
        consts_per_branch.append(consts)
    shared = set.intersection(*consts_per_branch)
    if not shared:
        return None
    # Stable pick: prefer common discriminator names, else lexicographically first.
    for preferred in ("kind", "type", "tag", "discriminator"):
        if preferred in shared:
            return preferred
    return sorted(shared)[0]


def _merge_with_discriminator(branches: list[dict], discriminator: str) -> dict:
    enum_values: list = []
    merged_props: dict[str, Any] = {}
    for b in branches:
        props = b.get("properties") or {}
        disc_spec = props.get(discriminator) or {}
        if "const" in disc_spec and disc_spec["const"] not in enum_values:
            enum_values.append(disc_spec["const"])
        for name, spec in props.items():
            if name == discriminator:
                continue
            # Last-write-wins is fine: if two branches both define the same field
            # with different shapes, we lose precision but the schema stays valid.
            merged_props[name] = spec
    merged_props[discriminator] = {"enum": enum_values}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": merged_props,
        "required": [discriminator],
    }


def _merge_object_branches(branches: list[dict]) -> dict:
    merged_props: dict[str, Any] = {}
    for b in branches:
        for name, spec in (b.get("properties") or {}).items():
            merged_props[name] = spec
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": merged_props,
    }


def _combine(outer: dict, merged: dict) -> dict:
    """Merge a flattened-union schema (`merged`) into the outer schema that contained it.

    Outer-level keys take precedence; `properties` and `required` are unioned so the
    caller doesn't lose siblings that lived alongside the original `oneOf`.
    """
    if not merged:
        return outer
    out = dict(outer)

    for key, value in merged.items():
        if key not in out:
            out[key] = value
            continue
        if key == "properties" and isinstance(value, dict) and isinstance(out[key], dict):
            combined = dict(value)
            combined.update(out[key])  # outer wins on collisions
            out[key] = combined
        elif key == "required" and isinstance(value, list) and isinstance(out[key], list):
            out[key] = list(dict.fromkeys([*out[key], *value]))
        # Other collisions: outer keeps its value.

    # If outer didn't declare a type but the merge produced one, propagate.
    if "type" not in outer and "type" in merged:
        out.setdefault("type", merged["type"])
    return out
