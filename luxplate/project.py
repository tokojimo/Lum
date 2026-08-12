"""Portable, non-executable LuxPlate project archives."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from io import BytesIO, StringIO
import json
from pathlib import PurePosixPath
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import pandas as pd

from .blanks import BlankCorrectionResult
from .kinetics import KineticsResult
from .normalization import NormalizationResult
from .workflow import CompleteAnalysisResult

PROJECT_VERSION = 1
MAX_ARCHIVE_BYTES = 250 * 1024 * 1024
MAX_MEMBER_BYTES = 100 * 1024 * 1024
RESULT_TYPES = {result.__name__: result for result in (
    BlankCorrectionResult, NormalizationResult, KineticsResult, CompleteAnalysisResult
)}
PROJECT_KEYS = (
    "source_name", "source_identity", "long_data", "qc_journal",
    "validated_qc_journal", "qc_validated", "blank_correction_result",
    "normalization_result", "kinetics_result", "guided_decisions",
    "guided_complete_result",
    # Guided-analysis context needed to reopen a project at the same point of
    # the workflow, without requiring the original workbooks again.
    "guided_signature", "guided_media", "guided_strains", "guided_min_od",
    "guided_consecutive", "guided_window", "guided_r2", "guided_figure_families",
    "guided_figure_panels", "guided_figure_lum_scale", "guided_export_dpi",
    "guided_directional_comparisons_stack",
)


def _encode(value: object, tables: dict[str, pd.DataFrame], path: str) -> object:
    if isinstance(value, pd.DataFrame):
        table_name = f"tables/{path}.json"
        tables[table_name] = value
        return {"kind": "dataframe", "path": table_name}
    if is_dataclass(value) and value.__class__.__name__ in RESULT_TYPES:
        return {"kind": "result", "type": value.__class__.__name__, "fields": {
            field.name: _encode(getattr(value, field.name), tables, f"{path}_{field.name}")
            for field in fields(value)
        }}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return {"kind": "sequence", "items": [
            _encode(item, tables, f"{path}_{index}") for index, item in enumerate(value)
        ]}
    raise TypeError(f"État de projet non pris en charge : {type(value).__name__}")


def export_project(state: dict[str, object]) -> bytes:
    """Return a portable ZIP containing the supported portion of ``state``."""
    tables: dict[str, pd.DataFrame] = {}
    encoded = {key: _encode(state[key], tables, key) for key in PROJECT_KEYS if key in state}
    manifest = {"format": "luxplate-project", "version": PROJECT_VERSION, "state": encoded}
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for name, table in tables.items():
            archive.writestr(name, table.to_json(orient="table", index=False, date_format="iso"))
    return output.getvalue()


def _decode(value: object, archive: ZipFile) -> object:
    if not isinstance(value, dict) or "kind" not in value:
        return value
    if value["kind"] == "dataframe":
        path = str(value.get("path", ""))
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise ValueError("Chemin de tableau invalide dans le projet.")
        return pd.read_json(StringIO(archive.read(path).decode("utf-8")), orient="table")
    if value["kind"] == "sequence":
        return [_decode(item, archive) for item in value.get("items", [])]
    if value["kind"] == "result":
        result_type = RESULT_TYPES.get(str(value.get("type", "")))
        if result_type is None:
            raise ValueError("Type de résultat inconnu dans le projet.")
        expected = {field.name for field in fields(result_type)}
        supplied = set(value.get("fields", {}))
        if supplied != expected:
            raise ValueError("Le résultat sauvegardé est incomplet ou incompatible.")
        return result_type(**{name: _decode(item, archive) for name, item in value["fields"].items()})
    raise ValueError("Contenu inconnu dans le projet.")


def import_project(payload: bytes) -> dict[str, object]:
    """Validate and decode a project produced by :func:`export_project`."""
    if not payload or len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError("Le fichier projet est vide ou trop volumineux.")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            if "project.json" not in archive.namelist():
                raise ValueError("Ce fichier ne contient pas de projet LuxPlate.")
            if any(info.file_size > MAX_MEMBER_BYTES for info in archive.infolist()):
                raise ValueError("Un élément du projet est trop volumineux.")
            manifest = json.loads(archive.read("project.json"))
            if manifest.get("format") != "luxplate-project":
                raise ValueError("Format de projet non reconnu.")
            if manifest.get("version") != PROJECT_VERSION:
                raise ValueError("Cette version de projet n'est pas compatible avec l'application.")
            encoded = manifest.get("state")
            if not isinstance(encoded, dict) or not set(encoded).issubset(PROJECT_KEYS):
                raise ValueError("État de projet invalide.")
            return {key: _decode(value, archive) for key, value in encoded.items()}
    except (BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Le fichier projet est endommagé ou invalide.") from error
