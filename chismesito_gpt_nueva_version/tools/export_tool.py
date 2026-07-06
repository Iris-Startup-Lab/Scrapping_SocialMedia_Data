# -*- coding: utf-8 -*-
"""tools/export_tool.py — Exportacion a CSV y Excel."""

import logging
import pandas as pd
import tempfile
from pathlib import Path
from langchain.tools import tool

logger = logging.getLogger(__name__)


@tool
def export_to_csv(data_json: str) -> dict:
    """
    Exporta comentarios a CSV.

    Args:
        data_json: JSON string con la lista de comentarios

    Returns:
        {"success": bool, "filepath": str, "error": str}
    """
    try:
        import json
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        df = pd.DataFrame(data)
        path = Path(tempfile.gettempdir()) / "chismesito_export.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return {"success": True, "filepath": str(path), "error": None}
    except Exception as e:
        return {"success": False, "filepath": "", "error": str(e)}


@tool
def export_to_excel(data_json: str) -> dict:
    """
    Exporta comentarios a Excel (.xlsx).

    Args:
        data_json: JSON string con la lista de comentarios

    Returns:
        {"success": bool, "filepath": str, "error": str}
    """
    try:
        import json
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
        df = pd.DataFrame(data)
        path = Path(tempfile.gettempdir()) / "chismesito_export.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")
        return {"success": True, "filepath": str(path), "error": None}
    except Exception as e:
        return {"success": False, "filepath": "", "error": str(e)}
