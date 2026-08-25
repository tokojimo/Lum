"""Regression checks for workbook-parser routing in the Streamlit entry point."""

import ast
from pathlib import Path


def test_import_mode_is_checked_before_the_selected_parser_is_called():
    """Endpoint files must never be eagerly sent to the kinetic parser."""
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    routing = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "per_time_mode"
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "cached_single_time_workbook"
            for statement in node.body for call in ast.walk(statement)
        )
    ]
    assert routing, "app.py doit choisir le parseur avant de lire le classeur"

    branch = routing[0]
    endpoint_calls = {
        call.func.id for statement in branch.body for call in ast.walk(statement)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    legacy_calls = {
        call.func.id for statement in branch.orelse for call in ast.walk(statement)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert endpoint_calls == {"cached_single_time_workbook"}
    assert "cached_kinetic_workbook" in legacy_calls


def test_mixed_mode_offers_per_workbook_parser_selection():
    source = Path("app.py").read_text(encoding="utf-8")
    assert 'mixed_mode = import_mode == "Mode mixte"' in source
    assert 'f"Contenu — {upload.name}"' in source
    assert "combine_mixed_tables(kinetic_inputs, endpoint_inputs, mapping)" in source


def test_single_luminescence_sheet_is_selected_without_a_selectbox():
    source = Path("app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    single_sheet_branches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "len(lum_sheets) == 1" in ast.get_source_segment(source, node.test)
    ]
    assert single_sheet_branches
    calls = [call for statement in single_sheet_branches[0].body for call in ast.walk(statement)
             if isinstance(call, ast.Call)]
    assert not any(isinstance(call.func, ast.Attribute) and call.func.attr == "selectbox"
                   for call in calls)
