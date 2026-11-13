from Analysis.normalize_rag_pool import MAX_BATCH_SOURCE_CHARS, _batch_code


def _item(origin: str, package: str = "Workflow") -> dict[str, str]:
    return {
        "origin": origin,
        "package": package,
        "normalized_sysml": (
            "private import MOSAICO::*;\n"
            "private import ScalarValues::*;\n"
            f"package {package} {{ action def Run {{ }} }}"
        ),
    }


def test_batch_package_aliases_are_unique_per_validation_attempt():
    item = _item("sopbench:1")

    first = _batch_code([item], attempt=1)
    second = _batch_code([item], attempt=2)

    assert first is not None
    assert second is not None
    assert first != second
    assert "package Workflow" not in first
    assert "package Workflow" not in second
    assert "package Workflow" in item["normalized_sysml"]


def test_qualified_package_references_fall_back_from_batching():
    item = _item("sopbench:2")
    item["normalized_sysml"] = item["normalized_sysml"].replace(
        "private import ScalarValues::*;",
        "private import ScalarValues::*;\nprivate import Workflow::Run;",
    )

    assert _batch_code([item], attempt=1) is None


def test_oversized_batch_falls_back_to_smaller_requests():
    first = _item("sopbench:3")
    second = _item("sopbench:4")
    first["normalized_sysml"] = "x" * (MAX_BATCH_SOURCE_CHARS // 2 + 1)
    second["normalized_sysml"] = "y" * (MAX_BATCH_SOURCE_CHARS // 2 + 1)

    assert _batch_code([first, second], attempt=1) is None
