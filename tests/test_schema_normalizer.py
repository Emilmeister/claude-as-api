from claude_proxy.schema_normalizer import normalize


def _has_unions(schema) -> bool:
    if isinstance(schema, dict):
        if "oneOf" in schema or "anyOf" in schema:
            return True
        return any(_has_unions(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_unions(v) for v in schema)
    return False


def test_passthrough_when_no_union():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }
    out = normalize(schema)
    assert out == schema
    assert not _has_unions(out)


def test_discriminated_union_flattens_with_enum():
    schema = {
        "oneOf": [
            {"type": "object", "properties": {"kind": {"const": "circle"}, "radius": {"type": "number"}},
             "required": ["kind", "radius"]},
            {"type": "object", "properties": {"kind": {"const": "square"}, "side": {"type": "number"}},
             "required": ["kind", "side"]},
        ],
    }
    out = normalize(schema)
    assert not _has_unions(out)
    assert out["type"] == "object"
    assert out["properties"]["kind"] == {"enum": ["circle", "square"]}
    assert "radius" in out["properties"] and "side" in out["properties"]
    assert out["required"] == ["kind"]


def test_anyof_objects_without_discriminator_merge_props():
    schema = {
        "anyOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "object", "properties": {"b": {"type": "integer"}}},
        ],
    }
    out = normalize(schema)
    assert not _has_unions(out)
    assert out["type"] == "object"
    assert set(out["properties"].keys()) == {"a", "b"}
    assert "required" not in out


def test_mixed_type_union_falls_back_to_first_object_branch():
    schema = {
        "oneOf": [
            {"type": "string"},
            {"type": "object", "properties": {"x": {"type": "integer"}}},
        ],
    }
    out = normalize(schema)
    assert not _has_unions(out)
    # Should pick the object branch.
    assert out.get("type") == "object"
    assert "x" in out["properties"]


def test_nested_union_inside_property_is_normalized():
    schema = {
        "type": "object",
        "properties": {
            "shape": {
                "oneOf": [
                    {"type": "object", "properties": {"kind": {"const": "a"}, "x": {"type": "integer"}}},
                    {"type": "object", "properties": {"kind": {"const": "b"}, "y": {"type": "string"}}},
                ],
            },
        },
        "required": ["shape"],
    }
    out = normalize(schema)
    assert not _has_unions(out)
    assert out["properties"]["shape"]["properties"]["kind"] == {"enum": ["a", "b"]}
    assert out["required"] == ["shape"]


def test_union_alongside_outer_properties_is_combined():
    schema = {
        "type": "object",
        "properties": {"base": {"type": "string"}},
        "required": ["base"],
        "oneOf": [
            {"type": "object", "properties": {"kind": {"const": "x"}, "x_val": {"type": "integer"}}},
            {"type": "object", "properties": {"kind": {"const": "y"}, "y_val": {"type": "string"}}},
        ],
    }
    out = normalize(schema)
    assert not _has_unions(out)
    # Outer properties preserved, branch fields merged in.
    assert "base" in out["properties"]
    assert "kind" in out["properties"]
    assert "x_val" in out["properties"] and "y_val" in out["properties"]
    # Outer required survives; branch-required fields are dropped (they're per-branch).
    assert "base" in out["required"]
    assert "kind" in out["required"]


def test_const_discriminator_picks_kind_over_alphabetical():
    schema = {
        "oneOf": [
            {"type": "object", "properties": {
                "aaa_marker": {"const": "ax"}, "kind": {"const": "x"}}},
            {"type": "object", "properties": {
                "aaa_marker": {"const": "ay"}, "kind": {"const": "y"}}},
        ],
    }
    out = normalize(schema)
    assert out["properties"]["kind"] == {"enum": ["x", "y"]}
    assert out["required"] == ["kind"]


def test_dedup_in_discriminator_enum():
    schema = {
        "oneOf": [
            {"type": "object", "properties": {"kind": {"const": "a"}, "x": {"type": "integer"}}},
            {"type": "object", "properties": {"kind": {"const": "a"}, "y": {"type": "string"}}},
        ],
    }
    out = normalize(schema)
    assert out["properties"]["kind"] == {"enum": ["a"]}


def test_normalize_does_not_mutate_input():
    schema = {
        "oneOf": [
            {"type": "object", "properties": {"kind": {"const": "a"}}},
            {"type": "object", "properties": {"kind": {"const": "b"}}},
        ],
    }
    original = {
        "oneOf": [
            {"type": "object", "properties": {"kind": {"const": "a"}}},
            {"type": "object", "properties": {"kind": {"const": "b"}}},
        ],
    }
    normalize(schema)
    assert schema == original
