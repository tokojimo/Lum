from __future__ import annotations

import re
import sys
from pathlib import Path
import pandas as pd

DEFAULT_INPUT_CSV = Path(r"C:\Users\morav\Desktop\Analyse luminescance\Data\Manip_Optimisation\NORM_Essai_SCFM1_1_format_long_Luminescence_200_02\Essai_SCFM1_1_format_long_Luminescence_200_02_corrige_blancs_normalise_DO.csv")


def read_csv_robust(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding=enc,
                engine="python",
                sep=",",
            )
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Impossible de lire le CSV: {last_error}")


def shift_group_numbers(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    def repl(match: re.Match[str]) -> str:
        return f"Groupe {int(match.group(1)) + 18}"

    return re.sub(r"\bGroupe\s+(\d+)\b", repl, text)


def shift_blanc_names(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{int(match.group(2)) + 3}"

    return re.sub(r"\b(Blanc|blanc)(\d+)\b", repl, text)


def contains_verif_cross(row: pd.Series) -> bool:
    return any("verif cross" in str(value).lower() for value in row)


def main() -> None:
    input_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_CSV

    if not input_csv.exists():
        raise FileNotFoundError(f"Fichier introuvable: {input_csv}")

    df = read_csv_robust(input_csv)

    unwanted_souches = {
        "SCFM2 (25%) RPMI (75%) 14.1Ac",
        "SCFM2 (50%) RPMI (50%) 14.1Ac",
        "SCFM2 (75%) RPMI (25%) 14.1Ac",
        "SCFM1 (25%) RPMI (75%) 14.1Ac",
        "SCFM1 (50%) RPMI (50%) 14.1Ac",
        "SCFM1 (75%) RPMI (25%) 14.1Ac",
    }

    if "souche" in df.columns:
        df = df[~df["souche"].astype(str).str.strip().isin(unwanted_souches)].copy()

    df = df[~df.apply(contains_verif_cross, axis=1)].copy()

    for col in df.columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace("SCFM2", "SCFM1", regex=False)
        df[col] = df[col].map(shift_blanc_names)
        df[col] = df[col].map(shift_group_numbers)

    output_path = input_csv.with_name(input_csv.stem + "_modifie.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Fichier créé : {output_path}")
    print(f"Lignes restantes : {len(df)}")


if __name__ == "__main__":
    main()
