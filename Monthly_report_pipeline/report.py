from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.io as pio


DATE_COLS = ["Submit Date", "Assigned Date", "Close Date", "Targeted Due Date", "Last Transfer Date"]

# Cell values that mean "no jurisdiction recorded" rather than naming a real place.
JURISDICTION_BLANKS = {"", "unknown", "n/a", "na", "none", "null", "tbd", "-", "--", "."}


# ---------------------------------------------------------------------------
# Jurisdiction canonicalization
# ---------------------------------------------------------------------------
# The workbook spells the same place several ways -- "Harris Co. - Texas",
# "Houston (Harris County, TX)", "Washington DC" vs "Washington, DC". Filters,
# charts and the by-jurisdiction rollup all group on the canonical name below;
# ticket rows keep whatever the sheet actually says.
CANONICAL_JURISDICTIONS = [
    "Maricopa Co. - Arizona", "Alameda Co. - California", "Los Angeles Co. - California",
    "Orange Co. - California", "Riverside Co. - California", "Sacramento Co. - California",
    "San Bernadino Co. -California", "San Diego Co. - California", "San Francisco Co. - California",
    "Broward Co. - Florida", "Duval Co. - Florida", "Hillsborough Co. - Florida",
    "Miami-Dade Co. - Florida", "Orange Co. - Florida", "Palm Beach Co. - Florida",
    "Pinellas Co. - Florida", "Cobb Co. - Georgia", "Dekalb Co. - Georgia",
    "Fulton Co. - Georgia", "Gwinnett Co. - Georgia", "Cook Co. - Illinois",
    "Marion Co. - Indiana", "East Baton Rough Parish - Louisiana", "Orleans Parish - Louisiana",
    "Baltimore City - Maryland", "Montgomery Co. - Maryland", "Prince George's Co. - Maryland",
    "Suffolk Co. - Massachusetts", "Wayne Co. - Michigan", "Clark Co. - Neveda",
    "Essex Co. - New Jersey", "Hudson Co. - New Jersey", "Bronx Co. - New York",
    "Kings Co. - New York", "New York Co. - New York", "Queens Co. - New York",
    "Mecklenburg Co. - North Carolina", "Cuyahoga Co. - Ohio", "Franklin Co. - Ohio",
    "Hamilton Co. - Ohio", "Philadelphia Co. - Pennsylvania", "Shelby Co. - Tennessee",
    "Bexar Co. - Texas", "Dallas Co. - Texas", "Harris Co. - Texas", "Tarrant Co. - Texas",
    "Travis Co. - Texas", "King Co. - Washington", "San Juan Municipio - Puerto Rico",
    "Washington, DC", "Alabama", "Arkansas", "Kentucky", "Mississippi", "Missouri",
    "Oklahoma", "South Carolina",
]

STATE_ABBR = {
    "al": "alabama", "az": "arizona", "ar": "arkansas", "ca": "california",
    "dc": "district of columbia", "fl": "florida", "ga": "georgia", "il": "illinois",
    "in": "indiana", "ky": "kentucky", "la": "louisiana", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "ms": "mississippi", "mo": "missouri",
    "nv": "nevada", "nj": "new jersey", "ny": "new york", "nc": "north carolina",
    "oh": "ohio", "ok": "oklahoma", "pa": "pennsylvania", "pr": "puerto rico",
    "sc": "south carolina", "tn": "tennessee", "tx": "texas", "wa": "washington",
}

# Cities the sheets use in place of the county/parish that is the real jurisdiction.
CITY_TO_PLACE = {
    "atlanta": ("fulton", "georgia"),            "austin": ("travis", "texas"),
    "baton rouge": ("east baton rouge", "louisiana"),
    "boston": ("suffolk", "massachusetts"),      "chicago": ("cook", "illinois"),
    "columbus": ("franklin", "ohio"),            "dallas": ("dallas", "texas"),
    "detroit": ("wayne", "michigan"),            "fort lauderdale": ("broward", "florida"),
    "fort worth": ("tarrant", "texas"),          "houston": ("harris", "texas"),
    "indianapolis": ("marion", "indiana"),       "las vegas": ("clark", "nevada"),
    "memphis": ("shelby", "tennessee"),          "new orleans": ("orleans", "louisiana"),
    "new york": ("new york", "new york"),        "newark": ("essex", "new jersey"),
    "philadelphia": ("philadelphia", "pennsylvania"),
    "phoenix": ("maricopa", "arizona"),          "sacramento": ("sacramento", "california"),
    "san bernadino": ("san bernardino", "california"),
    "san bernardino": ("san bernardino", "california"),
    "san juan": ("san juan", "puerto rico"),     "santa ana": ("orange", "california"),
    "seattle": ("king", "washington"),           "washington": ("washington", "district of columbia"),
    "washington dc": ("washington", "district of columbia"),
    "west palm beach": ("palm beach", "florida"),
}


def _nfkd(s: Any) -> str:
    """Normalize unicode (non-breaking spaces, curly quotes, dashes) and collapse spaces."""
    out = unicodedata.normalize("NFKD", str(s if s is not None else ""))
    out = out.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", out).strip()


def _place_key(place: str, state: str) -> tuple[str, str]:
    """Reduce a place + state pair to a comparable key."""
    pk = _nfkd(place).casefold()
    pk = re.sub(r"\b(county|counties|co\.|co|parish|municipio|city)\b", " ", pk)
    pk = re.sub(r"[^a-z' ]", " ", pk)
    pk = re.sub(r"\s+", " ", pk).strip()
    # Spelling variants used inconsistently across the sheets.
    pk = pk.replace("bernadino", "bernardino").replace("rough", "rouge").replace("dekalb", "de kalb")
    sk = _nfkd(state).casefold()
    sk = STATE_ABBR.get(sk, sk)
    if sk in ("states", "state", "territory"):
        sk = ""
    return pk, sk


def _build_jurisdiction_index(names: list[str]) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for name in names:
        n = _nfkd(name)
        if n == "Washington, DC":
            place, state = "washington", "district of columbia"
        elif re.search(r"\s-\s*", n):
            place, state = re.split(r"\s-\s*", n, maxsplit=1)
        else:
            place, state = n, ""
        key = _place_key(place, state)
        index[key] = name
        index.setdefault((key[0], ""), name)   # place-only fallback
    return index


JURISDICTION_INDEX = _build_jurisdiction_index(CANONICAL_JURISDICTIONS)


def canonical_jurisdiction(raw: Any) -> str | None:
    """
    Resolve one raw jurisdiction spelling to its canonical name.
    Returns None when the value is blank or cannot be matched.
    """
    n = _nfkd(raw)
    if not n or n.casefold() in JURISDICTION_BLANKS:
        return None

    place, state = n, ""
    paren = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", n)
    if paren:
        outer, inner = paren.group(1).strip(), paren.group(2).strip()
        if "," in inner:                       # "Houston (Harris County, TX)"
            county, st = inner.rsplit(",", 1)
            place, state = county.strip(), st.strip()
        else:                                  # "Alabama (AL)" / "Dallas (TX)"
            place, state = outer, inner.strip()
    elif re.search(r"\s-\s*", n):              # "Harris Co. - Texas"
        place, state = re.split(r"\s-\s*", n, maxsplit=1)
    elif n.casefold().startswith("washington"):
        place, state = "washington", "district of columbia"

    key = _place_key(place, state)
    for candidate in (key, (key[0], "")):
        if candidate in JURISDICTION_INDEX:
            return JURISDICTION_INDEX[candidate]

    # Fall back to the city -> county lookup, on the parsed place and on the
    # bare outer name ("Houston (Harris County, TX)" -> "houston").
    tries = [key[0]]
    if paren:
        tries.append(_place_key(paren.group(1), "")[0])
    for t in tries:
        city = CITY_TO_PLACE.get(t)
        if not city:
            continue
        ckey = _place_key(city[0], city[1])
        for candidate in (ckey, (ckey[0], "")):
            if candidate in JURISDICTION_INDEX:
                return JURISDICTION_INDEX[candidate]
    return None


def standardize_jurisdiction_series(s: pd.Series) -> pd.Series:
    """Canonical name where we recognize it; otherwise the tidied original."""
    cleaned = s.fillna("").astype(str).map(_nfkd)
    resolved = cleaned.map(canonical_jurisdiction)
    fallback = cleaned.where(
        ~cleaned.str.casefold().isin(JURISDICTION_BLANKS) & (cleaned != ""), "Unknown"
    )
    return resolved.fillna(fallback)


# ---------------------------------------------------------------------------
# Focus Area standardization
# ---------------------------------------------------------------------------
# The raw "Focus Area" typed on a ticket is kept verbatim everywhere a ticket is
# shown (table rows, ticket detail). Only the Focus Area chart uses the collapsed
# 10-category version below, so the bar chart stays readable.
FOCUS_AREA_CATEGORIES = [
    "Program Planning and Sustainability",
    "Fiscal, Procurement, and Grant Management",
    "Evaluation, Quality Improvement, and Reporting",
    "Data Systems and Data-to-Care",
    "Care Engagement and Service Delivery",
    "Clinical Care and Treatment",
    "Justice-Involved Populations",
    "Community Engagement and Partnerships",
    "Training, Orientation, and Capacity Building",
    "Prevention, Implementation, and Innovation",
]

FOCUS_AREA_MAP = {
    # 1. Program Planning and Sustainability
    "work plan": "Program Planning and Sustainability",
    "workplan": "Program Planning and Sustainability",
    "work plan updates": "Program Planning and Sustainability",
    "ta for work plan updating": "Program Planning and Sustainability",
    "work plan and budget": "Program Planning and Sustainability",
    "workplan and budget": "Program Planning and Sustainability",
    "sustainability": "Program Planning and Sustainability",
    "sustainability planning": "Program Planning and Sustainability",

    # 2. Fiscal, Procurement, and Grant Management
    "unspent funds": "Fiscal, Procurement, and Grant Management",
    "program income": "Fiscal, Procurement, and Grant Management",
    "allowable costs": "Fiscal, Procurement, and Grant Management",
    "procurement": "Fiscal, Procurement, and Grant Management",
    "procurement and allowable costs": "Fiscal, Procurement, and Grant Management",
    "procurement/allowable costs": "Fiscal, Procurement, and Grant Management",
    "contracting and fiscal management": "Fiscal, Procurement, and Grant Management",
    "subrecipient monitoring": "Fiscal, Procurement, and Grant Management",
    "sub-recipient monitoring": "Fiscal, Procurement, and Grant Management",

    # 3. Evaluation, Quality Improvement, and Reporting
    "evaluation": "Evaluation, Quality Improvement, and Reporting",
    "program evaluation": "Evaluation, Quality Improvement, and Reporting",
    "evaluation strategies": "Evaluation, Quality Improvement, and Reporting",
    "data evaluation plan": "Evaluation, Quality Improvement, and Reporting",
    "data collection and evaluation": "Evaluation, Quality Improvement, and Reporting",
    "continuous quality improvement": "Evaluation, Quality Improvement, and Reporting",
    "cqi": "Evaluation, Quality Improvement, and Reporting",
    "corrective action plan": "Evaluation, Quality Improvement, and Reporting",
    "cap/site visit findings": "Evaluation, Quality Improvement, and Reporting",
    "data reporting definitions": "Evaluation, Quality Improvement, and Reporting",
    "needs assessment": "Evaluation, Quality Improvement, and Reporting",
    "updated needs assessment": "Evaluation, Quality Improvement, and Reporting",
    "needs assessment (ehe strategy)": "Evaluation, Quality Improvement, and Reporting",

    # 4. Data Systems and Data-to-Care
    "power bi dashboards": "Data Systems and Data-to-Care",
    "data dashboard development": "Data Systems and Data-to-Care",
    "building data dashboard": "Data Systems and Data-to-Care",
    "data sharing": "Data Systems and Data-to-Care",
    "pharmacy data-to-care": "Data Systems and Data-to-Care",
    "pharmacy data to care": "Data Systems and Data-to-Care",
    "prescription refill data": "Data Systems and Data-to-Care",
    "calculator/modeling activity": "Data Systems and Data-to-Care",

    # 5. Care Engagement and Service Delivery
    "rapid start": "Care Engagement and Service Delivery",
    "linkage to care activities": "Care Engagement and Service Delivery",
    "linkage-to-care activities": "Care Engagement and Service Delivery",
    "retention and re-engagement": "Care Engagement and Service Delivery",
    "retention and re-engagement of people with hiv": "Care Engagement and Service Delivery",
    "emergency department testing": "Care Engagement and Service Delivery",
    "housing": "Care Engagement and Service Delivery",
    "telehealth/telemedicine": "Care Engagement and Service Delivery",

    # 6. Clinical Care and Treatment
    "long-acting injectable art": "Clinical Care and Treatment",
    "genotypic resistance testing protocol development": "Clinical Care and Treatment",
    "hiv and aging": "Clinical Care and Treatment",

    # 7. Justice-Involved Populations
    "jail linkage-to-care program": "Justice-Involved Populations",
    "subrecipient jail linkage to care program": "Justice-Involved Populations",
    "jail/prison coordination": "Justice-Involved Populations",
    "coordination with the jails/prisons": "Justice-Involved Populations",
    "criminal justice/corrections activities": "Justice-Involved Populations",
    "criminal justice/corrections activitites": "Justice-Involved Populations",

    # 8. Community Engagement and Partnerships
    "community engagement": "Community Engagement and Partnerships",
    "partnerships": "Community Engagement and Partnerships",
    "nontraditional partner goals review": "Community Engagement and Partnerships",
    "non-traditional partners goals review": "Community Engagement and Partnerships",
    "marketing and information campaigns": "Community Engagement and Partnerships",
    "alternative language guidance": "Community Engagement and Partnerships",
    "pl cares app peer learning": "Community Engagement and Partnerships",
    "pl cares health and wellness app peer learning": "Community Engagement and Partnerships",

    # 9. Training, Orientation, and Capacity Building
    "gu introduction meeting": "Training, Orientation, and Capacity Building",
    "gu introduction meetings": "Training, Orientation, and Capacity Building",
    "follow-up gu intro meeting": "Training, Orientation, and Capacity Building",
    "gu introduction/health and wellness training": "Training, Orientation, and Capacity Building",
    "ehe orientation and onboarding": "Training, Orientation, and Capacity Building",
    "ehe orientation and onboarding support": "Training, Orientation, and Capacity Building",
    "provider education": "Training, Orientation, and Capacity Building",
    "capacity building": "Training, Orientation, and Capacity Building",
    "capacity-building training": "Training, Orientation, and Capacity Building",
    "would love feedback and support in improving our capacity building trainings": "Training, Orientation, and Capacity Building",
    "hiv peer navigator protocols and training resources": "Training, Orientation, and Capacity Building",
    "hiv peer navigator program protocols and training resources": "Training, Orientation, and Capacity Building",

    # 10. Prevention, Implementation, and Innovation
    "prevention": "Prevention, Implementation, and Innovation",
    "implementation science": "Prevention, Implementation, and Innovation",
    "identifying ebi/eii": "Prevention, Implementation, and Innovation",
    "identifying ebi or eii": "Prevention, Implementation, and Innovation",
    "innovation development": "Prevention, Implementation, and Innovation",
    "developing new innovations": "Prevention, Implementation, and Innovation",
}

# Fallback keyword rules, applied in order, for spellings that appear later and are
# not yet in FOCUS_AREA_MAP. First match wins.
FOCUS_AREA_KEYWORDS = [
    (("work plan", "workplan", "sustainab"), "Program Planning and Sustainability"),
    (("procure", "allowable cost", "unspent", "program income", "fiscal", "contracting",
      "subrecipient monitoring", "sub-recipient monitoring", "budget"),
     "Fiscal, Procurement, and Grant Management"),
    (("evaluat", "quality improvement", "cqi", "corrective action", "site visit",
      "needs assessment", "reporting definition"),
     "Evaluation, Quality Improvement, and Reporting"),
    (("dashboard", "power bi", "data sharing", "data to care", "data-to-care",
      "refill", "calculator", "modeling"),
     "Data Systems and Data-to-Care"),
    (("rapid start", "linkage to care", "linkage-to-care", "retention", "re-engagement",
      "emergency department", "housing", "telehealth", "telemedicine"),
     "Care Engagement and Service Delivery"),
    (("injectable", "art ", "genotypic", "resistance testing", "aging", "clinic"),
     "Clinical Care and Treatment"),
    (("jail", "prison", "corrections", "criminal justice", "justice-involved"),
     "Justice-Involved Populations"),
    (("community engagement", "partner", "marketing", "campaign", "language", "pl cares"),
     "Community Engagement and Partnerships"),
    (("training", "orientation", "onboarding", "capacity building", "capacity-building",
      "provider education", "gu intro", "peer navigator"),
     "Training, Orientation, and Capacity Building"),
    (("prevention", "implementation science", "ebi", "eii", "innovation"),
     "Prevention, Implementation, and Innovation"),
]

FOCUS_AREA_UNMAPPED = "Other / Unspecified"


def _focus_key(value: str) -> str:
    """Normalize a raw focus-area string down to a lookup key."""
    s = str(value or "").replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s+", " ", s).strip().strip(".;,").strip().casefold()
    return s


def standardize_focus_area(value: Any) -> str:
    """Map one raw Focus Area value onto one of the 10 standardized categories."""
    key = _focus_key(value)
    if not key or key in JURISDICTION_BLANKS:
        return FOCUS_AREA_UNMAPPED
    if key in FOCUS_AREA_MAP:
        return FOCUS_AREA_MAP[key]
    # Already a standardized category name?
    for cat in FOCUS_AREA_CATEGORIES:
        if key == cat.casefold():
            return cat
    for needles, cat in FOCUS_AREA_KEYWORDS:
        if any(n in key for n in needles):
            return cat
    return FOCUS_AREA_UNMAPPED


def canonicalize_jurisdiction(s: pd.Series) -> pd.Series:
    """
    Collapse whitespace and casing variants of the same jurisdiction onto one spelling.

    'District of Columbia', 'District of Columbia ' and 'district of columbia' are one
    jurisdiction, not three. For each case-folded key the most frequently used original
    spelling wins, so real capitalization ('District of Columbia', not 'District Of
    Columbia') is preserved. Blank-ish entries all become 'Unknown'.
    """
    cleaned = (
        s.fillna("")
         .astype(str)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )
    key = cleaned.str.casefold()
    is_blank = key.isin(JURISDICTION_BLANKS)

    real = cleaned[~is_blank]
    if real.empty:
        return pd.Series("Unknown", index=s.index)

    # Dominant spelling per case-folded key; ties resolve alphabetically for stability.
    tally = real.groupby(real.str.casefold()).value_counts()
    best = {
        k: tally[k].sort_index().idxmax()
        for k in tally.index.get_level_values(0).unique()
    }

    out = key.map(best)
    return out.where(~is_blank, "Unknown").fillna("Unknown")


# -----------------------------
# A) Read multi-sheet workbook
# -----------------------------
def read_hrsa_workbook(xlsx_path: str) -> dict[str, pd.DataFrame]:
    """
    Reads the HRSA workbook and returns a dict of DataFrames.
    Expected sheets (based on your file): Main, Interaction, Delivery
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"Excel file not found: {xlsx_path}\n"
            f"Tip: pass an absolute path, e.g. python build_hrsa_dashboard_advanced.py --xlsx /full/path/file.xlsx"
        )

    try:
        xls = pd.ExcelFile(xlsx_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open Excel file: {xlsx_path}\n{e}") from e

    sheets = {name: pd.read_excel(xlsx_path, sheet_name=name) for name in xls.sheet_names}
    return sheets


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def normalize_main(df_main: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dates(df_main)

    # Required
    if "Ticket ID" not in df.columns:
        raise ValueError("Main sheet must include 'Ticket ID' column.")

    if "Submit Date" not in df.columns:
        raise ValueError(
            "Main sheet must include 'Submit Date' column. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Derived month
    df["Submit Month"] = df["Submit Date"].dt.to_period("M").astype(str)

    # Durations
    df["Days to Assign"] = (df["Assigned Date"] - df["Submit Date"]).dt.days if "Assigned Date" in df.columns else np.nan
    df["Days to Close"] = (df["Close Date"] - df["Submit Date"]).dt.days if "Close Date" in df.columns else np.nan

    # Jurisdiction normalize
    if "Jurisdiction" in df.columns:
        df["Jurisdiction"] = canonicalize_jurisdiction(df["Jurisdiction"])
    else:
        df["Jurisdiction"] = "Unknown"
    df["Jurisdiction (Standardized)"] = standardize_jurisdiction_series(df["Jurisdiction"])

    # Raw Focus Area is kept as-is for ticket rows/details; the standardized version
    # is what the Focus Area chart groups on.
    if "Focus Area" in df.columns:
        df["Focus Area"] = df["Focus Area"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        df["Focus Area (Standardized)"] = df["Focus Area"].map(standardize_focus_area)
    else:
        df["Focus Area"] = ""
        df["Focus Area (Standardized)"] = FOCUS_AREA_UNMAPPED
    return df


def normalize_interactions(df_inter: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns in your sheet:
    Ticket ID, Jurisdiction, Date of Interaction, Type of Interaction, Short Summary, Document, Submission Date, Submitted By
    """
    df = df_inter.copy()
    if "Ticket ID" not in df.columns:
        return pd.DataFrame(columns=["Ticket ID"])

    # Parse dates if present
    for c in ["Date of Interaction", "Submission Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Redact submitted by (privacy)
    if "Submitted By" in df.columns:
        df["Submitted By"] = "[redacted]"

    # Make sure strings are strings
    for c in ["Type of Interaction", "Short Summary", "Document"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    # Same canonicalization as the main sheet
    if "Jurisdiction" in df.columns:
        df["Jurisdiction"] = canonicalize_jurisdiction(df["Jurisdiction"])
        df["Jurisdiction (Standardized)"] = standardize_jurisdiction_series(df["Jurisdiction"])
    else:
        df["Jurisdiction"] = "Unknown"
        df["Jurisdiction (Standardized)"] = "Unknown"

    return df


def normalize_deliveries(df_del: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns in your sheet:
    Ticket ID, Date of Delivery, Type of Delivery, Short Summary, Document, Submission Date, Submitted By
    """
    df = df_del.copy()
    if "Ticket ID" not in df.columns:
        return pd.DataFrame(columns=["Ticket ID"])

    for c in ["Date of Delivery", "Submission Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Add delivery month based on Date of Delivery
    if "Date of Delivery" in df.columns:
        df["Delivery Month"] = df["Date of Delivery"].dt.to_period("M").astype(str)

    if "Submitted By" in df.columns:
        df["Submitted By"] = "[redacted]"

    for c in ["Type of Delivery", "Short Summary", "Document"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    return df


# ---------------------------------------------------------------------------
# Additional TA activities: Intensives (ITA) and Peer Learning Networks (PLN)
# ---------------------------------------------------------------------------
ITA_SHEET = "Intensives"
PLN_SHEET = "PLNs"


def _split_jurisdiction_list(value: Any) -> list[str]:
    """'A County, B County' -> ['A County', 'B County'] (blank-safe)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if s.casefold() in {"nan", "nat", "none"}:
        return []
    if not s or s.casefold() in JURISDICTION_BLANKS:
        return []
    parts = [re.sub(r"\s+", " ", p).strip() for p in s.split(",")]
    return [p for p in parts if p and p.casefold() not in JURISDICTION_BLANKS]


def _normalize_activity_sheet(df: pd.DataFrame, jur_col: str, kind: str) -> pd.DataFrame:
    """Shared cleanup for the Intensives / PLNs sheets."""
    cols = ["ID", "Date", "Location", jur_col, "Status", "Focus Area", "Meeting Notes"]
    if df is None or df.empty:
        return pd.DataFrame(columns=["ID", "Date", "Location", "Jurisdictions", "Status", "Focus Area", "Meeting Notes", "Kind"])

    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].copy()
    df = df.rename(columns={jur_col: "Jurisdictions"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Drop the empty filler rows Excel leaves behind.
    df = df[df["Date"].notna() | df["ID"].notna()]
    df = df[~(df["Date"].isna() & df["Jurisdictions"].isna())]

    for c in ["Location", "Status", "Focus Area", "Meeting Notes"]:
        df[c] = df[c].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    df["Kind"] = kind
    return df.reset_index(drop=True)


def read_extra_activities(xlsx_path: str | None) -> dict[str, pd.DataFrame]:
    """
    Read 'HRSA064 Additional TA activities.xlsx'.
    Sheet 'Intensives' -> ITAs, sheet 'PLNs' -> PLNs.
    Missing file or sheets degrade to empty frames rather than failing the build.
    """
    empty = {
        "ita": _normalize_activity_sheet(pd.DataFrame(), "Jurisdictions", "ITA"),
        "pln": _normalize_activity_sheet(pd.DataFrame(), "Jurisdictions Attended", "PLN"),
    }
    if not xlsx_path or not os.path.exists(xlsx_path):
        if xlsx_path:
            print(f"! Additional TA activities workbook not found ({xlsx_path}); ITA/PLN panels will be empty.", file=sys.stderr)
        return empty

    try:
        xls = pd.ExcelFile(xlsx_path)
    except Exception as e:
        print(f"! Could not open {xlsx_path}: {e}", file=sys.stderr)
        return empty

    ita_raw = pd.read_excel(xlsx_path, sheet_name=ITA_SHEET) if ITA_SHEET in xls.sheet_names else pd.DataFrame()
    pln_raw = pd.read_excel(xlsx_path, sheet_name=PLN_SHEET) if PLN_SHEET in xls.sheet_names else pd.DataFrame()

    return {
        "ita": _normalize_activity_sheet(ita_raw, "Jurisdictions", "ITA"),
        "pln": _normalize_activity_sheet(pln_raw, "Jurisdictions Attended", "PLN"),
    }


def activity_records(df: pd.DataFrame, spelling_map: dict[str, str]) -> list[dict[str, Any]]:
    """Turn a normalized ITA/PLN frame into JSON-ready records for the dashboard."""
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        jurs_raw = _split_jurisdiction_list(row.get("Jurisdictions"))
        # Reuse the main sheet's spelling when we recognize the jurisdiction.
        jurs = [spelling_map.get(j.casefold(), j) for j in jurs_raw]
        dt = row.get("Date")
        dt = pd.to_datetime(dt, errors="coerce")
        out.append({
            "id": None if pd.isna(row.get("ID")) else int(row.get("ID")),
            "date": "" if pd.isna(dt) else dt.strftime("%Y-%m-%d"),
            "month": "" if pd.isna(dt) else dt.strftime("%Y-%m"),
            "location": str(row.get("Location") or ""),
            "status": str(row.get("Status") or ""),
            "focus_area": str(row.get("Focus Area") or ""),
            "notes": str(row.get("Meeting Notes") or ""),
            "jurisdictions": jurs,
            "jurisdiction_count": len(jurs),
            "complete": str(row.get("Status") or "").strip().casefold().startswith("complet"),
        })
    out.sort(key=lambda r: (r["date"] or "", r["id"] or 0))
    return out


# -----------------------------
# B) Build payload (monthly + per-ticket + drilldown records)
# -----------------------------
def safe_nanmean(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if len(s) == 0:
        return None
    v = float(np.nanmean(s))
    return v if np.isfinite(v) else None


def compute_payload(
    df_main: pd.DataFrame,
    df_inter: pd.DataFrame,
    df_del: pd.DataFrame,
    df_ita: pd.DataFrame | None = None,
    df_pln: pd.DataFrame | None = None,
    id_col: str = "Ticket ID",
) -> dict[str, Any]:

    # Counts from Interaction & Delivery sheets
    # For interactions: count by Ticket ID, but also include interactions without Ticket ID (they have Jurisdiction)
    inter_counts = (
        df_inter.groupby(id_col).size().reset_index(name="interactions_count")
        if id_col in df_inter.columns and not df_inter.empty else pd.DataFrame(columns=[id_col, "interactions_count"])
    )
    
    # Count interactions without Ticket ID but with Jurisdiction - these should be counted as general interactions
    # Try to assign them to months based on Date of Interaction if available
    inter_without_ticket = 0
    inter_without_ticket_by_month = {}
    if not df_inter.empty and id_col in df_inter.columns:
        # Check for interactions where Ticket ID is null/na but Jurisdiction exists
        if "Jurisdiction" in df_inter.columns:
            inter_no_ticket = df_inter[
                (df_inter[id_col].astype(str).str.strip() == "No Ticket ID")
            ]
            inter_without_ticket = len(inter_no_ticket)
            
            # If Date of Interaction exists, assign to months
            if "Date of Interaction" in inter_no_ticket.columns and inter_without_ticket > 0:
                inter_no_ticket = inter_no_ticket.copy()
                inter_no_ticket["Date of Interaction"] = pd.to_datetime(inter_no_ticket["Date of Interaction"], errors="coerce")
                inter_no_ticket["Interaction Month"] = inter_no_ticket["Date of Interaction"].dt.to_period("M").astype(str)
                inter_without_ticket_by_month = inter_no_ticket.groupby("Interaction Month").size().to_dict()
    
    # Count deliveries per ticket ID (for ticket-level display)
    del_counts = (
        df_del.groupby(id_col).size().reset_index(name="deliveries_count")
        if id_col in df_del.columns and not df_del.empty else pd.DataFrame(columns=[id_col, "deliveries_count"])
    )

    df = df_main.copy()

    # Every interaction and delivery gets a jurisdiction: the one on its ticket when the
    # ticket exists, otherwise the jurisdiction recorded on the row itself. This is what
    # lets interactions with no Ticket ID still roll up by place.
    ticket_jur = {}
    if id_col in df.columns and "Jurisdiction (Standardized)" in df.columns:
        ticket_jur = dict(
            zip(df[id_col].astype(str).str.strip(), df["Jurisdiction (Standardized)"].astype(str))
        )

    def resolve_row_jurisdiction(frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype=str)
        keys = frame[id_col].astype(str).str.strip() if id_col in frame.columns else pd.Series("", index=frame.index)
        from_ticket = keys.map(ticket_jur)
        own = (
            frame["Jurisdiction (Standardized)"].astype(str)
            if "Jurisdiction (Standardized)" in frame.columns
            else pd.Series("Unknown", index=frame.index)
        )
        return from_ticket.fillna(own).replace("", "Unknown").fillna("Unknown")

    if not df_inter.empty:
        df_inter = df_inter.copy()
        df_inter["Jurisdiction (Standardized)"] = resolve_row_jurisdiction(df_inter)
    if not df_del.empty:
        df_del = df_del.copy()
        df_del["Jurisdiction (Standardized)"] = resolve_row_jurisdiction(df_del)

    df = df.merge(inter_counts, on=id_col, how="left").merge(del_counts, on=id_col, how="left")
    df["interactions_count"] = df["interactions_count"].fillna(0).astype(int)
    df["deliveries_count"] = df["deliveries_count"].fillna(0).astype(int)

    # Count deliveries by delivery month (not by TA request submit month)
    # Get all unique months from both Submit Month and Delivery Month
    submit_months = set(df["Submit Month"].dropna().unique())
    delivery_months = set()
    if not df_del.empty and "Delivery Month" in df_del.columns:
        delivery_months = set(df_del["Delivery Month"].dropna().unique())
    all_months = sorted(list(submit_months | delivery_months))

    # Monthly KPIs (overall; filters are done in JS)
    # Group by Submit Month for tickets
    monthly_submit = (
        df.groupby("Submit Month", dropna=False)
          .agg(
              submitted=(id_col, "count"),
              completed=("Close Date", lambda s: int(pd.to_datetime(s, errors="coerce").notna().sum()) if "Close Date" in df.columns else 0),
              avg_days_to_assign=("Days to Assign", safe_nanmean),
              avg_days_to_close=("Days to Close", safe_nanmean),
              interactions=("interactions_count", "sum"),
          )
          .reset_index()
    )

    # Count deliveries by Delivery Month
    monthly_deliveries = (
        df_del.groupby("Delivery Month", dropna=False).size().reset_index(name="deliveries")
        if "Delivery Month" in df_del.columns and not df_del.empty
        else pd.DataFrame(columns=["Delivery Month", "deliveries"])
    )

    # Merge monthly data, ensuring all months are included
    monthly = pd.DataFrame({"Submit Month": all_months})
    monthly = monthly.merge(monthly_submit, on="Submit Month", how="left")
    monthly = monthly.merge(monthly_deliveries.rename(columns={"Delivery Month": "Submit Month"}), on="Submit Month", how="left")
    
    # Fill NaN values
    monthly["submitted"] = monthly["submitted"].fillna(0).astype(int)
    monthly["completed"] = monthly["completed"].fillna(0).astype(int)
    monthly["interactions"] = monthly["interactions"].fillna(0).astype(int)
    monthly["deliveries"] = monthly["deliveries"].fillna(0).astype(int)
    monthly["avg_days_to_assign"] = monthly["avg_days_to_assign"]
    monthly["avg_days_to_close"] = monthly["avg_days_to_close"]
    monthly["completion_rate"] = np.where(
        monthly["submitted"] > 0,
        monthly["completed"] / monthly["submitted"],
        np.nan,
    )
    
    # Add interactions without Ticket ID to monthly totals based on their interaction date
    if inter_without_ticket_by_month:
        for month, count in inter_without_ticket_by_month.items():
            if month in monthly["Submit Month"].values:
                monthly.loc[monthly["Submit Month"] == month, "interactions"] = (
                    monthly.loc[monthly["Submit Month"] == month, "interactions"].values[0] + count
                )
            else:
                # If month doesn't exist in monthly, add a new row
                monthly = pd.concat([
                    monthly,
                    pd.DataFrame([{
                        "Submit Month": month,
                        "submitted": 0,
                        "completed": 0,
                        "interactions": count,
                        "deliveries": 0,
                        "completion_rate": np.nan
                    }])
                ], ignore_index=True)

    # Backlog at EOM (overall)
    months = sorted([m for m in df["Submit Month"].dropna().unique()])
    month_ends = pd.to_datetime([m + "-01" for m in months]) + pd.offsets.MonthEnd(0)

    backlog_rows = []
    for m, mend in zip(months, month_ends):
        submitted_up_to = df[(df["Submit Date"].notna()) & (df["Submit Date"] <= mend)]
        open_as_of = submitted_up_to[(submitted_up_to["Close Date"].isna()) | (submitted_up_to["Close Date"] > mend)]
        backlog_rows.append({"Submit Month": m, "backlog_end_of_month": int(open_as_of.shape[0])})

    monthly = monthly.merge(pd.DataFrame(backlog_rows), on="Submit Month", how="left")

    # Ticket table fields (exclude PII)
    safe_cols = [
        id_col, "Submit Date", "Submit Month", "Jurisdiction", "Jurisdiction (Standardized)",
        "Organization", "Focus Area", "Focus Area (Standardized)", "TA Type", "Priority", "Status",
        "TA Description", "Targeted Due Date", "Assigned Coordinator", "Assigned Coach",
        "Assigned Date", "Close Date", "Days to Assign", "Days to Close",
        "interactions_count", "deliveries_count",
    ]
    safe_cols = [c for c in safe_cols if c in df.columns]
    tickets = df[safe_cols].copy()

    # stringify dates for JSON
    for c in ["Submit Date", "Targeted Due Date", "Assigned Date", "Close Date"]:
        if c in tickets.columns:
            tickets[c] = pd.to_datetime(tickets[c], errors="coerce").dt.strftime("%Y-%m-%d")

    # "Jurisdictions Engaged" — MAIN sheet only. The Interaction sheet is deliberately not
    # unioned in: it carries jurisdictions with no ticket on the main page, which inflated
    # this number. 'Unknown' / blank is not a jurisdiction, so it is excluded too.
    # Spelling was already canonicalized in normalize_main().
    all_jurisdictions = sorted(
        j for j in df["Jurisdiction (Standardized)"].fillna("Unknown").astype(str).unique()
        if j.casefold() not in JURISDICTION_BLANKS and j != "Unknown"
    )
    # Places reached only through an interaction with no ticket still count as engaged.
    engaged_jurisdictions = set(all_jurisdictions)
    for frame in (df_inter, df_del):
        if not frame.empty and "Jurisdiction (Standardized)" in frame.columns:
            engaged_jurisdictions |= {
                j for j in frame["Jurisdiction (Standardized)"].astype(str).unique()
                if j.casefold() not in JURISDICTION_BLANKS and j != "Unknown"
            }
    engaged_jurisdictions = sorted(engaged_jurisdictions)
    total_jurisdictions = len(engaged_jurisdictions)
    
    # Calculate summary statistics
    total_tickets = len(df)
    total_interactions = int(df["interactions_count"].sum() + inter_without_ticket)
    # Count total deliveries from delivery dataframe (not from ticket-based counts)
    total_deliveries = len(df_del) if not df_del.empty else 0
    
    # Calculate completed and in-progress tickets
    if "Close Date" in df.columns:
        completed_tickets = int(df["Close Date"].notna().sum())
        in_progress_tickets = total_tickets - completed_tickets
    else:
        completed_tickets = 0
        in_progress_tickets = total_tickets

    # Drilldown records: interactions + deliveries keyed by Ticket ID
    # Keep only minimal safe fields
    inter_keep = [c for c in ["Ticket ID", "Jurisdiction", "Jurisdiction (Standardized)", "Date of Interaction", "Type of Interaction", "Short Summary", "Document"] if c in df_inter.columns]
    del_keep = [c for c in ["Ticket ID", "Jurisdiction (Standardized)", "Date of Delivery", "Delivery Month", "Type of Delivery", "Short Summary", "Document"] if c in df_del.columns]

    inter_rec = df_inter[inter_keep].copy() if inter_keep else pd.DataFrame(columns=["Ticket ID"])
    del_rec = df_del[del_keep].copy() if del_keep else pd.DataFrame(columns=["Ticket ID"])
    
    # Join deliveries with main tab to get jurisdiction info
    if not del_rec.empty and id_col in del_rec.columns and "Jurisdiction" in df.columns:
        del_rec = del_rec.merge(
            df[[id_col, "Jurisdiction"]].drop_duplicates(),
            on=id_col,
            how="left"
        )
        del_rec["Jurisdiction"] = del_rec["Jurisdiction"].fillna("Unknown").astype(str)
        # Add Jurisdiction to del_keep if it wasn't there originally
        if "Jurisdiction" not in del_keep:
            del_keep.append("Jurisdiction")

    if "Date of Interaction" in inter_rec.columns:
        inter_rec["Date of Interaction"] = pd.to_datetime(inter_rec["Date of Interaction"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "Date of Delivery" in del_rec.columns:
        del_rec["Date of Delivery"] = pd.to_datetime(del_rec["Date of Delivery"], errors="coerce").dt.strftime("%Y-%m-%d")

    # ---- ITAs and PLNs -------------------------------------------------
    spelling_map = {j.casefold(): j for j in engaged_jurisdictions}
    ita_records = activity_records(df_ita, spelling_map) if df_ita is not None and not df_ita.empty else []
    pln_records = activity_records(df_pln, spelling_map) if df_pln is not None and not df_pln.empty else []

    # Jurisdictions that only appear via an ITA/PLN still deserve a filter entry.
    activity_jurs = set()
    for rec in (ita_records + pln_records):
        canon = [canonical_jurisdiction(j) or j for j in rec["jurisdictions"]]
        rec["jurisdictions"] = canon
        activity_jurs |= set(canon)
    filter_jurisdictions = sorted(set(engaged_jurisdictions) | activity_jurs)

    focus_area_categories = [
        c for c in FOCUS_AREA_CATEGORIES
        if c in set(df.get("Focus Area (Standardized)", pd.Series(dtype=str)).unique())
    ]

    return {
        "summary": {
            "total_tickets": total_tickets,
            "completed_tickets": completed_tickets,
            "in_progress_tickets": in_progress_tickets,
            "total_interactions": total_interactions,
            "total_deliveries": total_deliveries,
            "total_jurisdictions": total_jurisdictions,
            # Metrics count delivered sessions only; scheduled ones still show in the tables.
            "total_itas": sum(1 for r in ita_records if r["complete"]),
            "total_plns": sum(1 for r in pln_records if r["complete"]),
            "scheduled_itas": sum(1 for r in ita_records if not r["complete"]),
            "scheduled_plns": sum(1 for r in pln_records if not r["complete"]),
        },
        "months": months,
        "jurisdictions": engaged_jurisdictions,
        "filter_jurisdictions": filter_jurisdictions,
        "focus_area_categories": focus_area_categories,
        "monthly": monthly.to_dict(orient="records"),
        "tickets": tickets.to_dict(orient="records"),
        "interactions": inter_rec.to_dict(orient="records"),
        "deliveries": del_rec.to_dict(orient="records"),
        "itas": ita_records,
        "plns": pln_records,
    }


# -----------------------------
# C) Build HTML (single-file offline)
# -----------------------------
# The page is a plain template string (not an f-string) so the JavaScript below can
# use normal braces. Placeholders are substituted in build_html().
HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    /* ---------- design tokens ---------- */
    :root {
      --surface:        #f4f5f7;
      --surface-card:   #ffffff;
      --surface-sunken: #fafbfc;
      --border:         #e4e7ec;
      --border-strong:  #d0d5dd;
      --ink:            #101828;
      --ink-2:          #475467;
      --ink-3:          #98a2b3;
      --brand:          #17357a;
      --brand-soft:     #eef2fb;

      /* chart series - validated categorical slots (blue, orange, aqua, yellow) */
      --s1: #2a78d6;  --s2: #eb6834;  --s3: #1baf7a;  --s4: #eda100;
      --grid: #eceff3;

      /* stat-tile accents (text-legible steps of the same hues) */
      --c-tickets:#1c5cab; --c-completed:#0f7a4f; --c-progress:#b45309;
      --c-interactions:#256abf; --c-deliveries:#4a3aa7; --c-jurisdictions:#0f766e;
      --c-ita:#b02525; --c-pln:#8a5a00;

      --shadow-sm: 0 1px 2px rgba(16,24,40,.05);
      --shadow-md: 0 1px 3px rgba(16,24,40,.06), 0 8px 24px -12px rgba(16,24,40,.12);
      --radius: 14px;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif;
      margin: 0; background: var(--surface); color: var(--ink);
      font-size: 14px; line-height: 1.5; -webkit-font-smoothing: antialiased;
    }

    /* ---------- header ---------- */
    .header { background: var(--surface-card); padding: 26px 32px 22px; border-bottom: 1px solid var(--border); }
    .header-content { display: flex; align-items: center; justify-content: center; gap: 22px; max-width: 1440px; margin: 0 auto; }
    .header-logo { height: 200px; width: auto; }
    .header-text { text-align: center; }
    .header h1 { margin: 0 0 4px 0; font-size: 34px; font-weight: 750; letter-spacing: -.02em; color: var(--brand); }
    .header .subtitle { margin: 0; font-size: 13px; color: var(--ink-3); }

    .container { max-width: 1440px; margin: 0 auto; padding: 18px 22px 56px; }

    /* ---------- structure ---------- */
    .card {
      background: var(--surface-card); border: 1px solid var(--border); border-radius: var(--radius);
      padding: 18px; box-shadow: var(--shadow-sm); margin-bottom: 18px;
    }
    .sectionHead { display:flex; align-items:baseline; justify-content:space-between; gap:14px; flex-wrap:wrap; margin-bottom: 12px; }
    h2.section { margin: 0; font-size: 17px; font-weight: 700; letter-spacing: -.01em; color: var(--ink); }
    .eyebrow {
      font-size: 21px; font-weight: 750; letter-spacing: -.015em; color: var(--ink);
      margin: 34px 0 14px; padding: 2px 0 2px 13px; border-left: 5px solid var(--brand);
      display: flex; align-items: baseline; gap: 12px;
    }
    .eyebrow:first-of-type { margin-top: 6px; }
    .eyebrow .sub { font-size: 12.5px; font-weight: 500; color: var(--ink-3); letter-spacing: 0; }
    .subnote { font-size: 12px; color: var(--ink-3); margin: 2px 0 10px; }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }

    /* ---------- filter bar ---------- */
    .filterBar {
      position: sticky; top: 0; z-index: 60; margin: 0 0 18px;
      background: rgba(255,255,255,.92); backdrop-filter: saturate(140%) blur(8px);
      border: 1px solid var(--border); border-radius: var(--radius);
      padding: 14px 16px; box-shadow: var(--shadow-md);
    }
    .fgroup { display: flex; flex-direction: column; gap: 5px; }
    label.flab { font-size: 10.5px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: var(--ink-3); }
    select, input[type="text"] {
      padding: 8px 11px; border-radius: 9px; border: 1px solid var(--border-strong);
      background: #fff; font-size: 13px; color: var(--ink); font-family: inherit;
      transition: border-color .12s, box-shadow .12s;
    }
    select:focus, input[type="text"]:focus { outline: none; border-color: var(--s1); box-shadow: 0 0 0 3px rgba(42,120,214,.14); }
    .seg { display: inline-flex; border: 1px solid var(--border-strong); border-radius: 9px; overflow: hidden; background: #fff; }
    .seg button {
      border: 0; background: #fff; padding: 8px 13px; font-size: 12.5px; font-family: inherit;
      color: var(--ink-2); cursor: pointer; border-right: 1px solid var(--border);
    }
    .seg button:last-child { border-right: 0; }
    .seg button:hover { background: var(--surface-sunken); }
    .seg button[aria-pressed="true"] { background: var(--brand-soft); color: var(--brand); font-weight: 650; }
    .btn {
      padding: 8px 12px; border-radius: 9px; border: 1px solid var(--border-strong); background: #fff;
      font-size: 12.5px; font-family: inherit; color: var(--ink-2); cursor: pointer;
    }
    .btn:hover { background: var(--surface-sunken); color: var(--ink); }
    .navlinks { display: flex; gap: 8px; flex-wrap: nowrap; align-items: center; }
    .navlinks .lead { font-size: 10.5px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: var(--ink-3); margin-right: 2px; }
    .navlink {
      display: inline-flex; align-items: center; gap: 6px; text-decoration: none;
      padding: 8px 14px; border-radius: 9px; font-size: 12.5px; font-weight: 650;
      color: #fff; background: var(--brand); border: 1px solid var(--brand);
      box-shadow: var(--shadow-sm); transition: background .12s, transform .12s;
      white-space: nowrap;
    }
    .navlink:hover { background: #0f2557; transform: translateY(-1px); }
    .navlink .dot { width: 8px; height: 8px; border-radius: 2px; background: var(--nav-accent, #fff); opacity: .95; }
    .filterNote { font-size: 12.5px; color: var(--ink-2); background: var(--brand-soft); border: 1px solid #dbe4f7;
                  border-radius: 9px; padding: 8px 12px; margin-top: 12px; }
    .filterNote.off { background: var(--surface-sunken); border-color: var(--border); color: var(--ink-3); }
    .filterNote b { color: var(--ink); }

    .jurBox {
      padding: 8px 11px; border-radius: 9px; border: 1px solid var(--border-strong); background: #fff;
      font-size: 13px; cursor: pointer; min-width: 230px; display: flex; justify-content: space-between;
      align-items: center; gap: 10px;
    }
    .jurBox:hover { border-color: var(--ink-3); }
    .jurDrop {
      display: none; position: absolute; top: calc(100% + 5px); left: 0; background: #fff;
      border: 1px solid var(--border-strong); border-radius: 11px; box-shadow: var(--shadow-md);
      z-index: 200; max-height: 320px; overflow-y: auto; min-width: 290px;
    }
    .jur-checkbox { padding: 7px 11px; cursor: pointer; display: flex; align-items: center; gap: 9px; font-size: 13px; }
    .jur-checkbox:hover { background: var(--surface-sunken); }
    .jur-checkbox input { margin: 0; cursor: pointer; accent-color: var(--s1); }

    /* ---------- stat tiles ---------- */
    .totals { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
    @media (min-width: 720px)  { .totals { grid-template-columns: repeat(4, 1fr); } }
    .tcard {
      background: var(--surface-card); border: 1px solid var(--border); border-radius: 14px;
      padding: 16px 18px 14px; display: flex; flex-direction: column;
      box-shadow: var(--shadow-sm); position: relative; overflow: hidden;
    }
    .tcard::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--accent); }
    .tcard .tlabel {
      font-size: 12px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
      color: var(--ink-2); line-height: 1.2;
    }
    .tcard .tvalue { font-size: 46px; font-weight: 750; line-height: 1.05; color: var(--accent); margin: 6px 0 2px; letter-spacing: -.03em; }
    .tcard .ttotal { font-size: 11px; font-weight: 600; color: var(--ink-3); }
    .tcard .tfiltered { font-size: 12.5px; color: var(--ink-2); margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--border); }
    .tcard .tfiltered b { color: var(--accent); font-weight: 750; font-size: 14px; }
    .tcard.clickable { cursor: pointer; transition: box-shadow .13s, transform .13s; }
    .tcard.clickable:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
    .hint { font-size: 11px; color: var(--ink-3); margin-top: 6px; }

    /* ---------- current-selection strip ---------- */
    .kpis { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    @media (min-width: 960px) { .kpis { grid-template-columns: repeat(8, 1fr); } }
    .kpi { border: 1px solid var(--border); border-radius: 10px; padding: 10px 11px; background: var(--surface-sunken); }
    .kpi .label { color: var(--ink-3); font-size: 10.5px; text-transform: uppercase; letter-spacing: .07em; font-weight: 700; }
    .kpi .value { font-size: 19px; font-weight: 700; margin-top: 3px; color: var(--ink); letter-spacing: -.01em; }

    /* ---------- charts ---------- */
    .charts2 { display: grid; grid-template-columns: 1fr; gap: 18px; }
    @media (min-width: 1120px) { .charts2 { grid-template-columns: 1fr 1fr; } }
    .chartTall { height: 400px; }

    /* ---------- tables ---------- */
    table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 12.5px; }
    th, td { padding: 9px 9px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--border); }
    th {
      color: var(--ink-2); font-weight: 650; font-size: 11px; letter-spacing: .04em; text-transform: uppercase;
      background: var(--surface-sunken); position: sticky; top: 0; z-index: 2;
      box-shadow: inset 0 -1px 0 var(--border);
    }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
    td.clip { max-width: 155px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    td.clip.jur { max-width: 135px; }
    td.overdue { color: #b42318; font-weight: 650; }
    td.overdue::after { content: " !"; font-weight: 800; }
    tbody tr:nth-child(even) { background: #fcfcfd; }
    .tableWrap { max-height: 520px; overflow: auto; border: 1px solid var(--border); border-radius: 11px; background: #fff; }
    .tableWrap.short { max-height: 330px; }
    .tableWrap.mid { max-height: 420px; }
    .clickRow { cursor: pointer; }
    .clickRow:hover td { background: var(--surface-sunken); }
    .clickRow.active td { background: var(--brand-soft); }
    .clickRow.active td:first-child { box-shadow: inset 3px 0 0 var(--brand); }
    td.nowrap { white-space: nowrap; }

    /* Interaction / delivery records: the summaries run to a couple of thousand
       characters, so each record is a block with its full text, not a table cell. */
    .recList { max-height: 460px; overflow-y: auto; padding-right: 4px; }
    .rec { border: 1px solid var(--border); border-radius: 10px; background: #fff; padding: 11px 13px; margin-bottom: 8px; }
    .rec:last-child { margin-bottom: 0; }
    .recHead { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 7px; }
    .recDate { font-size: 12px; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; white-space: nowrap; }
    .recTag { display: inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border);
              background: var(--surface-sunken); font-size: 10.5px; font-weight: 650; color: var(--ink-2); white-space: nowrap; }
    .recDoc { font-size: 11px; color: var(--s1); text-decoration: none; border-bottom: 1px solid rgba(42,120,214,.35); }
    .recDoc:hover { border-bottom-color: var(--s1); }
    .recBody { font-size: 12.5px; line-height: 1.55; color: var(--ink); white-space: pre-wrap; word-wrap: break-word; }
    .recCount { font-size: 11px; font-weight: 700; color: var(--ink-3); letter-spacing: .04em; }
    .recSection { display: flex; align-items: baseline; gap: 8px; margin: 14px 0 7px; }
    .recSection b { font-size: 13px; }

    .split { display: grid; grid-template-columns: 1fr; gap: 16px; }
    @media (min-width: 1120px) { .split { grid-template-columns: 1.1fr 1fr; } }
    .detailBox { border: 1px solid var(--border); border-radius: 11px; padding: 14px; background: var(--surface-sunken); }
    .detailTitle { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
    .detailBox table { background: #fff; }
    .detailBox .tableWrap { max-height: 260px; }
    .detailBox table th:nth-child(3), .detailBox table td:nth-child(3) { min-width: 260px; }

    .muted { color: var(--ink-3); }
    .pill {
      display: inline-block; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--border);
      background: #fff; font-size: 11px; color: var(--ink-2);
    }
    .pill b { color: var(--ink); }
    .pill.accent { border-color: transparent; color: #fff; }
    .pill.count { background: var(--brand-soft); border-color: #dbe4f7; color: var(--brand); font-weight: 700;
                  font-variant-numeric: tabular-nums; }
    .jtag { display: inline-block; padding: 3px 9px; margin: 2px 4px 2px 0; border-radius: 999px;
            background: #fff; border: 1px solid var(--border); font-size: 11px; color: var(--ink-2); }
    .badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 10.5px;
             font-weight: 700; letter-spacing: .03em; text-transform: uppercase; white-space: nowrap; }
    .badge.done { background: #e7f4ee; color: #0f7a4f; border: 1px solid #c5e6d7; }
    .badge.sched { background: #fdf0e6; color: #b45309; border: 1px solid #f6dcc4; }
    .actHead { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:8px;
               padding: 9px 12px; border-radius: 10px; border: 1px solid var(--border); background: var(--surface-sunken); }
    .actHead .name { font-size: 14px; font-weight: 700; letter-spacing: -.01em; }
    .actHead .meta { font-size: 11.5px; color: var(--ink-3); font-weight: 500; }
    .swatch { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:6px; vertical-align:baseline; }
    .activity2 { display: grid; grid-template-columns: 1fr; gap: 16px; }
    @media (min-width: 1120px) { .activity2 { grid-template-columns: 1fr 1fr; } }
    .emptyMsg { padding: 18px; text-align: center; color: var(--ink-3); font-size: 12.5px; }
  </style>
</head>
<body>
  <div class="header">
    <div class="header-content">
      __LOGO_BLOCK__
      <div class="header-text">
        <h1>__TITLE__</h1>
        <p class="subtitle">As of __CURRENT_DATE__</p>
      </div>
    </div>
  </div>

  <div class="container">

  <!-- ============ FILTER BAR ============ -->
  <div class="filterBar">
    <div class="row" style="gap:20px;">
      <div class="fgroup">
        <label class="flab">Time grouping</label>
        <div class="seg" id="granSeg" role="group" aria-label="Time grouping">
          <button type="button" data-gran="month" aria-pressed="true">Monthly</button>
          <button type="button" data-gran="quarter" aria-pressed="false">Quarterly</button>
          <button type="button" data-gran="year" aria-pressed="false">Yearly</button>
        </div>
        <select id="granSel" style="display:none;">
          <option value="month" selected>Monthly</option>
          <option value="quarter">Quarterly</option>
          <option value="year">Yearly</option>
        </select>
      </div>

      <div class="fgroup">
        <label class="flab" for="yearSel">Year</label>
        <select id="yearSel"></select>
      </div>

      <div class="fgroup" id="periodGroup">
        <label class="flab" for="periodSel"><span id="periodLabel">Month</span></label>
        <select id="periodSel"></select>
      </div>

      <div class="fgroup">
        <label class="flab">Jurisdiction</label>
        <div style="position: relative;">
          <div id="jurSelectBox" class="jurBox">
            <span id="jurSelectText">All jurisdictions</span>
            <span style="font-size: 9px; color: var(--ink-3);">&#9660;</span>
          </div>
          <div id="jurDropdown" class="jurDrop">
            <input type="text" id="jurSearch" placeholder="Search jurisdictions..." style="width: 100%; border: none; border-bottom: 1px solid var(--border); border-radius: 11px 11px 0 0;">
            <div id="jurOptions" style="padding: 4px 0;"></div>
          </div>
        </div>
        <input type="hidden" id="jurSel" value="All jurisdictions">
      </div>

      <div class="fgroup">
        <label class="flab" for="ticketSearch">Search</label>
        <input id="ticketSearch" type="text" placeholder="Ticket ID / org / focus area / jurisdiction / summary..." style="min-width:310px;">
      </div>

      <div class="fgroup">
        <label class="flab">&nbsp;</label>
        <button class="btn" id="resetFilters">Reset filters</button>
      </div>

    </div>
    <div class="row" style="margin-top:12px; gap:12px; flex-wrap:nowrap; align-items:center;">
      <div id="filterNote" class="filterNote" style="flex:1; margin-top:0; display:none;"></div>
      <div style="flex:1;" id="filterSpacer"></div>
      <div class="navlinks">
        <span class="lead">Jump to</span>
        <a class="navlink" href="#secJurisdictions" style="--nav-accent: var(--s1);"><span class="dot"></span>Jurisdictions</a>
        <a class="navlink" href="#secTickets" style="--nav-accent: var(--s3);"><span class="dot"></span>Tickets</a>
        <a class="navlink" href="#activitySection" style="--nav-accent: var(--s4);"><span class="dot"></span>ITAs &amp; PLNs</a>
      </div>
    </div>
  </div>

  <!-- ============ TOTAL KPI CARDS ============ -->
  <div class="eyebrow">Program totals</div>
  <div class="totals" id="totalsGrid">
    <div class="tcard" style="--accent: var(--c-tickets);">
      <div class="tlabel">Tickets</div>
      <div class="tvalue" id="totTickets">&mdash;</div>
      <div class="ttotal">TA requests submitted</div>
      <div class="tfiltered" id="fltTickets" style="display:none;"></div>
    </div>
    <div class="tcard" style="--accent: var(--c-completed);">
      <div class="tlabel">Completed</div>
      <div class="tvalue" id="totCompleted">&mdash;</div>
      <div class="ttotal">Tickets closed</div>
      <div class="tfiltered" id="fltCompleted" style="display:none;"></div>
    </div>
    <div class="tcard" style="--accent: var(--c-progress);">
      <div class="tlabel">In progress</div>
      <div class="tvalue" id="totProgress">&mdash;</div>
      <div class="ttotal">Tickets still open</div>
      <div class="tfiltered" id="fltProgress" style="display:none;"></div>
    </div>
    <div class="tcard" style="--accent: var(--c-jurisdictions);">
      <div class="tlabel">Jurisdictions</div>
      <div class="tvalue" id="totJurisdictions">&mdash;</div>
      <div class="ttotal">Places engaged</div>
      <div class="tfiltered" id="fltJurisdictions" style="display:none;"></div>
    </div>
    <div class="tcard" style="--accent: var(--c-interactions);">
      <div class="tlabel">Interactions</div>
      <div class="tvalue" id="totInteractions">&mdash;</div>
      <div class="ttotal">Meetings, calls &amp; emails</div>
      <div class="tfiltered" id="fltInteractions" style="display:none;"></div>
    </div>
    <div class="tcard" style="--accent: var(--c-deliveries);">
      <div class="tlabel">Deliveries</div>
      <div class="tvalue" id="totDeliveries">&mdash;</div>
      <div class="ttotal">Products delivered</div>
      <div class="tfiltered" id="fltDeliveries" style="display:none;"></div>
    </div>
    <div class="tcard clickable" id="cardITA" style="--accent: var(--c-ita);" title="Click to see every ITA">
      <div class="tlabel">ITAs</div>
      <div class="tvalue" id="totITA">&mdash;</div>
      <div class="ttotal">Intensive TAs completed</div>
      <div class="tfiltered" id="fltITA" style="display:none;"></div>
    </div>
    <div class="tcard clickable" id="cardPLN" style="--accent: var(--c-pln);" title="Click to see every PLN">
      <div class="tlabel">PLNs</div>
      <div class="tvalue" id="totPLN">&mdash;</div>
      <div class="ttotal">Peer learning sessions completed</div>
      <div class="tfiltered" id="fltPLN" style="display:none;"></div>
    </div>
  </div>

  <!-- ============ FILTERED / PERIOD KPI STRIP ============ -->
  <div class="card" style="margin-top:18px;">
    <div class="sectionHead">
      <h2 class="section">Current selection</h2>
      <span class="pill" id="selectionPill">All time &middot; All jurisdictions</span>
    </div>
    <div class="kpis">
      <div class="kpi"><div class="label">Submitted</div><div class="value" id="kpiSubmitted">&mdash;</div></div>
      <div class="kpi"><div class="label">Completed</div><div class="value" id="kpiCompleted">&mdash;</div></div>
      <div class="kpi"><div class="label">Completion rate</div><div class="value" id="kpiRate">&mdash;</div></div>
      <div class="kpi"><div class="label">Avg days to close</div><div class="value" id="kpiClose">&mdash;</div></div>
      <div class="kpi"><div class="label">Interactions</div><div class="value" id="kpiInteractions">&mdash;</div></div>
      <div class="kpi"><div class="label">Deliveries</div><div class="value" id="kpiDeliveries">&mdash;</div></div>
      <div class="kpi"><div class="label">ITAs</div><div class="value" id="kpiITA">&mdash;</div></div>
      <div class="kpi"><div class="label">PLNs</div><div class="value" id="kpiPLN">&mdash;</div></div>
    </div>
  </div>

  <!-- ============ CHARTS ============ -->
  <div class="eyebrow">Trends</div>
  <div class="charts2">
    <div class="card">
      <div class="sectionHead" style="margin-bottom:2px;"><h2 class="section">Tickets submitted over time</h2></div>
      <div id="trendChart" class="chartTall"></div>
    </div>
    <div class="card">
      <div class="sectionHead" style="margin-bottom:2px;"><h2 class="section">Activity over time</h2></div>
      <div id="activityChart" class="chartTall"></div>
    </div>
  </div>

  <div class="charts2">
    <div class="card">
      <div class="sectionHead" style="margin-bottom:2px;"><h2 class="section">Tickets by status and jurisdiction</h2></div>
      <div id="statusJurisdictionChart" style="height: 620px;"></div>
    </div>
    <div class="card">
      <div class="sectionHead" style="margin-bottom:2px;"><h2 class="section">Tickets by focus area and status</h2></div>
      <div id="focusAreaStatusChart" style="height: 620px;"></div>
    </div>
  </div>

  <div style="display: none;">
  __PLOTLY_SNIPPET__
  </div>

  <!-- ============ ENGAGEMENT BY JURISDICTION ============ -->
  <div class="eyebrow" id="secJurisdictions">Engagement by jurisdiction</div>
  <div class="card">
    <div class="sectionHead" style="margin-bottom:2px;">
      <h2 class="section">Interactions &amp; deliveries by jurisdiction</h2>
    </div>
    <div id="jurEngagementChart" style="height: 640px;"></div>
  </div>

  <div class="card">
    <div class="sectionHead">
      <h2 class="section">Jurisdiction detail <span class="pill count" id="jurCountPill">0</span></h2>
    </div>
    <div class="split">
      <div class="tableWrap mid">
        <table>
          <thead>
            <tr>
              <th>Jurisdiction</th>
              <th class="num">Tickets</th>
              <th class="num">Interactions</th>
              <th class="num">Deliveries</th>
              <th class="num">Total</th>
            </tr>
          </thead>
          <tbody id="jurTbody"></tbody>
        </table>
      </div>
      <div class="detailBox">
        <div class="detailTitle">
          <b>Jurisdiction records</b>
          <button class="btn" id="clearJurDetail">Clear</button>
        </div>
        <div id="jurDetailBody" class="muted" style="margin-top:10px;">
          Click a jurisdiction row to list its interactions and deliveries.
        </div>
      </div>
    </div>
  </div>

  <!-- ============ TICKETS ============ -->
  <div class="eyebrow" id="secTickets">Tickets</div>
  <div class="card">
    <div class="sectionHead">
      <h2 class="section">TA tickets <span class="pill count" id="ticketCountPill">0</span></h2>
    </div>

    <div class="split">
      <div class="tableWrap">
        <table>
          <thead>
            <tr>
              <th>Ticket ID</th>
              <th>Submitted</th>
              <th>Due</th>
              <th>Jurisdiction</th>
              <th>Status</th>
              <th>Focus area</th>
              <th class="num">Int.</th>
              <th class="num">Del.</th>
            </tr>
          </thead>
          <tbody id="ticketTbody"></tbody>
        </table>
      </div>

      <div class="detailBox">
        <div class="detailTitle">
          <b>Ticket Details</b>
          <button class="btn" id="clearDetail">Clear</button>
        </div>
        <div id="detailBody" class="muted" style="margin-top:10px;">
          Click a ticket row to load Interactions and Deliveries.
        </div>
      </div>
    </div>
  </div>

  <!-- ============ ITA / PLN ============ -->
  <div class="eyebrow" id="activitySection">Intensive TAs &amp; Peer Learning Networks</div>
  <div class="card">
    <div class="sectionHead">
      <h2 class="section">ITA &amp; PLN sessions</h2>
    </div>

    <div class="activity2">
      <div>
        <div class="actHead" style="border-left:4px solid var(--c-ita);">
          <div>
            <div class="name" style="color:var(--c-ita);">Intensive TAs</div>
            <div class="meta" id="itaMeta">&mdash;</div>
          </div>
          <span class="pill count" id="itaCountPill">0</span>
        </div>
        <div class="tableWrap short">
          <table>
            <thead><tr><th>#</th><th>Date</th><th>Location</th><th class="num">Jurisdictions</th><th>Status</th></tr></thead>
            <tbody id="itaTbody"></tbody>
          </table>
        </div>
      </div>

      <div>
        <div class="actHead" style="border-left:4px solid var(--c-pln);">
          <div>
            <div class="name" style="color:var(--c-pln);">Peer Learning Networks</div>
            <div class="meta" id="plnMeta">&mdash;</div>
          </div>
          <span class="pill count" id="plnCountPill">0</span>
        </div>
        <div class="tableWrap short">
          <table>
            <thead><tr><th>#</th><th>Date</th><th>Focus area</th><th class="num">Jurisdictions</th><th>Status</th></tr></thead>
            <tbody id="plnTbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="detailBox" style="margin-top:14px;">
      <div class="detailTitle">
        <b>ITA / PLN Details</b>
        <button class="btn" id="clearActivityDetail">Clear</button>
      </div>
      <div id="activityDetailBody" class="muted" style="margin-top:10px;">
        Click an ITA or PLN row above to see its jurisdictions and focus areas.
      </div>
    </div>
  </div>

  <script>
    const PAYLOAD = __DATA_JSON__;

    /* ===================== helpers ===================== */
    const MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    const TODAY = new Date().toISOString().slice(0, 10);

    function fmtNum(x){
      if (x === null || x === undefined || Number.isNaN(x)) return "—";
      return String(Math.round(x * 10) / 10);
    }
    function fmtPct(x){
      if (x === null || x === undefined || Number.isNaN(x)) return "—";
      return (Math.round(x * 1000) / 10).toFixed(1) + "%";
    }
    function esc(s){
      if (s === null || s === undefined || (typeof s === "number" && Number.isNaN(s))) return "";
      return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
    }
    function dash(v){ const t = esc(v); return t ? t : "\u2014"; }
    function parseDate(v){
      if (!v) return null;
      const s = String(v).slice(0, 10);
      const parts = s.split("-");
      if (parts.length !== 3) return null;
      const y = parseInt(parts[0], 10), m = parseInt(parts[1], 10), d = parseInt(parts[2], 10);
      if (!y || !m || !d) return null;
      return { y: y, m: m, d: d };
    }

    /* Period keys at the three granularities. */
    function periodKeyFromDate(v, gran){
      const p = parseDate(v);
      if (!p) return null;
      if (gran === "year")    return String(p.y);
      if (gran === "quarter") return p.y + "-Q" + Math.ceil(p.m / 3);
      return p.y + "-" + String(p.m).padStart(2, "0");
    }
    /* Year and sub-period (month or quarter) are chosen independently in the filter bar. */
    function yearOf(v){ const p = parseDate(v); return p ? String(p.y) : null; }
    function subOf(v, gran){
      const p = parseDate(v);
      if (!p) return null;
      if (gran === "quarter") return "Q" + Math.ceil(p.m / 3);
      return String(p.m).padStart(2, "0");
    }
    function subLabel(sub, gran){
      if (gran === "quarter") return sub;
      return MONTH_NAMES[parseInt(sub, 10) - 1];
    }
    function periodKeyFromMonth(monthStr, gran){
      if (!monthStr) return null;
      const parts = String(monthStr).split("-");
      if (parts.length < 2) return null;
      const y = parseInt(parts[0], 10), m = parseInt(parts[1], 10);
      if (!y || !m) return null;
      if (gran === "year")    return String(y);
      if (gran === "quarter") return y + "-Q" + Math.ceil(m / 3);
      return y + "-" + String(m).padStart(2, "0");
    }
    function periodLabel(key, gran){
      if (!key) return "";
      if (gran === "year") return key;
      if (gran === "quarter") {
        const bits = key.split("-Q");
        return "Q" + bits[1] + " " + bits[0];
      }
      const bits = key.split("-");
      return MONTH_NAMES[parseInt(bits[1], 10) - 1] + " " + bits[0];
    }

    /* Every date in the dataset, so the period dropdown covers tickets,
       interactions, deliveries, ITAs and PLNs alike. */
    function allDates(){
      const out = [];
      (PAYLOAD.tickets || []).forEach(t => { if (t["Submit Date"]) out.push(t["Submit Date"]); });
      (PAYLOAD.interactions || []).forEach(x => { if (x["Date of Interaction"]) out.push(x["Date of Interaction"]); });
      (PAYLOAD.deliveries || []).forEach(x => { if (x["Date of Delivery"]) out.push(x["Date of Delivery"]); });
      (PAYLOAD.itas || []).forEach(x => { if (x.date) out.push(x.date); });
      (PAYLOAD.plns || []).forEach(x => { if (x.date) out.push(x.date); });
      return out;
    }
    const ALL_DATES = allDates();

    function periodsFor(gran){
      const set = new Set();
      ALL_DATES.forEach(d => { const k = periodKeyFromDate(d, gran); if (k) set.add(k); });
      return Array.from(set).sort();
    }

    /* ===================== elements ===================== */
    const granSel = document.getElementById("granSel");
    const yearSel = document.getElementById("yearSel");
    const periodSel = document.getElementById("periodSel");
    const periodGroup = document.getElementById("periodGroup");
    const periodLabelEl = document.getElementById("periodLabel");
    const jurSel = document.getElementById("jurSel");
    const jurSelectBox = document.getElementById("jurSelectBox");
    const jurSelectText = document.getElementById("jurSelectText");
    const jurDropdown = document.getElementById("jurDropdown");
    const jurSearch = document.getElementById("jurSearch");
    const jurOptions = document.getElementById("jurOptions");
    const ticketSearch = document.getElementById("ticketSearch");
    const resetFilters = document.getElementById("resetFilters");
    const filterNote = document.getElementById("filterNote");
    const selectionPill = document.getElementById("selectionPill");

    const ticketTbody = document.getElementById("ticketTbody");
    const ticketCountPill = document.getElementById("ticketCountPill");
    const detailBody = document.getElementById("detailBody");

    const itaTbody = document.getElementById("itaTbody");
    const plnTbody = document.getElementById("plnTbody");
    const itaCountPill = document.getElementById("itaCountPill");
    const plnCountPill = document.getElementById("plnCountPill");
    const activityDetailBody = document.getElementById("activityDetailBody");

    let selectedJurisdictions = new Set(["All jurisdictions"]);
    const ALL_PERIODS = "__ALL__";

    /* ===================== filter plumbing ===================== */
    function initYearOptions(){
      const keep = yearSel.value;
      const years = Array.from(new Set(ALL_DATES.map(yearOf).filter(Boolean))).sort().reverse();
      yearSel.innerHTML = "";
      const all = document.createElement("option");
      all.value = ALL_PERIODS; all.textContent = "All years";
      yearSel.appendChild(all);
      years.forEach(y => {
        const o = document.createElement("option");
        o.value = y; o.textContent = y;
        yearSel.appendChild(o);
      });
      yearSel.value = years.includes(keep) ? keep : ALL_PERIODS;
    }

    function initPeriodOptions(){
      const gran = granSel.value;
      const keep = periodSel.value;

      /* Yearly grouping has no sub-period to pick. */
      periodGroup.style.display = (gran === "year") ? "none" : "flex";
      periodLabelEl.textContent = (gran === "quarter") ? "Quarter" : "Month";

      /* Only offer months / quarters that actually have data in the chosen year. */
      const year = yearSel.value;
      const dates = (year === ALL_PERIODS) ? ALL_DATES : ALL_DATES.filter(d => yearOf(d) === year);
      const subs = Array.from(new Set(dates.map(d => subOf(d, gran)).filter(Boolean))).sort();
      periodSel.innerHTML = "";
      const all = document.createElement("option");
      all.value = ALL_PERIODS;
      all.textContent = (gran === "quarter") ? "All quarters" : "All months";
      periodSel.appendChild(all);
      subs.forEach(k => {
        const o = document.createElement("option");
        o.value = k; o.textContent = subLabel(k, gran);
        periodSel.appendChild(o);
      });
      periodSel.value = subs.includes(keep) ? keep : ALL_PERIODS;
    }

    function getFilters(){
      const jurValue = jurSel.value;
      const jurArray = jurValue === "All jurisdictions" ? ["All jurisdictions"] : jurValue.split(",");
      const gran = granSel.value;
      const year = yearSel.value;
      const sub = (gran === "year") ? ALL_PERIODS : periodSel.value;
      return {
        gran: gran,
        year: year, sub: sub,
        allYears: year === ALL_PERIODS,
        allSubs: sub === ALL_PERIODS,
        allPeriods: (year === ALL_PERIODS && sub === ALL_PERIODS),
        jurArray: jurArray,
        allJur: jurArray.includes("All jurisdictions"),
        q: (ticketSearch.value || "").trim().toLowerCase()
      };
    }
    function anyFilterActive(){
      const f = getFilters();
      return !f.allPeriods || !f.allJur || !!f.q;
    }
    function inPeriod(dateVal, f){
      if (f.allPeriods) return true;
      if (!f.allYears && yearOf(dateVal) !== f.year) return false;
      if (!f.allSubs && subOf(dateVal, f.gran) !== f.sub) return false;
      return true;
    }
    /* The timeline charts show the whole range and enlarge the selected periods. */
    function isSelectedPeriodKey(key, f){
      if (f.allPeriods) return false;
      if (f.gran === "year") return key === f.year;
      const bits = String(key).split("-");
      if (!f.allYears && bits[0] !== f.year) return false;
      if (!f.allSubs && bits[1] !== f.sub) return false;
      return true;
    }
    const JUR = "Jurisdiction (Standardized)";
    function jurMatches(jur, f){
      if (f.allJur) return true;
      return f.jurArray.includes(String(jur || "Unknown"));
    }
    function anyJurMatches(jurList, f){
      if (f.allJur) return true;
      return (jurList || []).some(j => f.jurArray.includes(j));
    }

    /* ===================== filtered slices ===================== */
    /* ---- search: one term, applied to every record type ---- */
    function blobHas(parts, q){
      return parts.map(x => String(x === null || x === undefined ? "" : x)).join(" ").toLowerCase().includes(q);
    }
    function ticketMatchesSearch(r, q){
      return blobHas([r["Ticket ID"], r["Organization"], r["Focus Area"], r["Focus Area (Standardized)"],
                      r["TA Type"], r["Status"], r["Jurisdiction"], r[JUR], r["TA Description"]], q);
    }
    /* Tickets whose own text matches, so their interactions and deliveries come along. */
    function searchTicketIds(f){
      if (!f.q) return null;
      const ids = new Set();
      (PAYLOAD.tickets || []).forEach(r => {
        if (ticketMatchesSearch(r, f.q)) ids.add(String(r["Ticket ID"] || "").trim());
      });
      return ids;
    }
    function interactionMatchesSearch(x, f, ids){
      if (!f.q) return true;
      const tid = String(x["Ticket ID"] || "").trim();
      if (ids && ids.has(tid)) return true;
      return blobHas([tid, x["Type of Interaction"], x["Short Summary"], x["Document"],
                      x["Jurisdiction"], x[JUR]], f.q);
    }
    function deliveryMatchesSearch(x, f, ids){
      if (!f.q) return true;
      const tid = String(x["Ticket ID"] || "").trim();
      if (ids && ids.has(tid)) return true;
      return blobHas([tid, x["Type of Delivery"], x["Short Summary"], x["Document"], x[JUR]], f.q);
    }
    function activityMatchesSearch(a, f){
      if (!f.q) return true;
      return blobHas([a.id, a.location, a.focus_area, a.status, (a.jurisdictions || []).join(" ")], f.q);
    }

    function ticketsJurOnly(f){
      /* jurisdiction + text search, no period filter (used by the timeline charts) */
      let rows = (PAYLOAD.tickets || []).slice();
      if (!f.allJur) rows = rows.filter(r => jurMatches(r[JUR], f));
      if (f.q) rows = rows.filter(r => ticketMatchesSearch(r, f.q));
      return rows;
    }
    function filteredTickets(){
      const f = getFilters();
      let rows = ticketsJurOnly(f);
      if (!f.allPeriods) rows = rows.filter(r => inPeriod(r["Submit Date"], f));
      rows.sort((a,b) => String(b["Submit Date"]||"").localeCompare(String(a["Submit Date"]||"")));
      return rows;
    }
    function ticketIdSet(rows){
      return new Set(rows.map(r => String(r["Ticket ID"] || "").trim()));
    }
    /* Every interaction and delivery already carries a resolved jurisdiction (from its
       ticket when it has one, otherwise from the record itself), so both filter directly. */
    function filteredInteractions(f){
      const ids = searchTicketIds(f);
      return (PAYLOAD.interactions || []).filter(x =>
        jurMatches(x[JUR], f) && inPeriod(x["Date of Interaction"], f) && interactionMatchesSearch(x, f, ids));
    }
    function filteredDeliveries(f){
      const ids = searchTicketIds(f);
      return (PAYLOAD.deliveries || []).filter(x =>
        jurMatches(x[JUR], f) && inPeriod(x["Date of Delivery"], f) && deliveryMatchesSearch(x, f, ids));
    }
    function filteredActivities(list, f){
      return (list || []).filter(a =>
        inPeriod(a.date, f) && anyJurMatches(a.jurisdictions, f) && activityMatchesSearch(a, f));
    }
    /* KPI counts track delivered sessions; scheduled ones stay visible in the tables. */
    function completedOnly(list){ return (list || []).filter(a => a.complete); }

    /* ===================== totals + KPIs ===================== */
    function setTotals(){
      const s = PAYLOAD.summary || {};
      document.getElementById("totTickets").textContent = fmtNum(s.total_tickets || 0);
      document.getElementById("totCompleted").textContent = fmtNum(s.completed_tickets || 0);
      document.getElementById("totProgress").textContent = fmtNum(s.in_progress_tickets || 0);
      document.getElementById("totInteractions").textContent = fmtNum(s.total_interactions || 0);
      document.getElementById("totDeliveries").textContent = fmtNum(s.total_deliveries || 0);
      document.getElementById("totJurisdictions").textContent = fmtNum(s.total_jurisdictions || 0);
      document.getElementById("totITA").textContent = fmtNum(s.total_itas || 0);
      document.getElementById("totPLN").textContent = fmtNum(s.total_plns || 0);
    }

    function setFilteredLine(id, value){
      const el = document.getElementById(id);
      if (!anyFilterActive()) { el.style.display = "none"; return; }
      el.style.display = "block";
      el.innerHTML = "Filtered: <b>" + fmtNum(value) + "</b>";
    }

    function describeSelection(){
      const f = getFilters();
      let periodTxt;
      if (f.allPeriods) periodTxt = "All time";
      else if (f.gran === "year") periodTxt = f.year;
      else if (f.allYears) periodTxt = subLabel(f.sub, f.gran) + " (every year)";
      else if (f.allSubs) periodTxt = f.year;
      else periodTxt = subLabel(f.sub, f.gran) + " " + f.year;
      let jurTxt;
      if (f.allJur) jurTxt = "All jurisdictions";
      else if (f.jurArray.length === 1) jurTxt = f.jurArray[0];
      else jurTxt = f.jurArray.length + " jurisdictions";
      return { periodTxt, jurTxt, q: f.q };
    }

    function setKPIs(){
      const f = getFilters();
      const rows = filteredTickets();

      const submitted = rows.length;
      const completed = rows.filter(r => (r["Close Date"] && String(r["Close Date"]).length > 0)).length;
      const inProgress = submitted - completed;
      const rate = submitted ? (completed / submitted) : NaN;

      const closeVals = rows.map(r => r["Days to Close"]).filter(v => v !== null && v !== undefined && !Number.isNaN(v));
      const avgClose = closeVals.length ? (closeVals.reduce((a,b)=>a+b,0) / closeVals.length) : NaN;

      const inter = filteredInteractions(f).length;
      const deliv = filteredDeliveries(f).length;
      const itasAll = filteredActivities(PAYLOAD.itas, f);
      const plnsAll = filteredActivities(PAYLOAD.plns, f);
      const itas = completedOnly(itasAll);
      const plns = completedOnly(plnsAll);

      const jurSet = new Set();
      rows.forEach(r => { const j = String(r[JUR] || "Unknown"); if (j !== "Unknown") jurSet.add(j); });
      filteredInteractions(f).forEach(x => { const j = String(x[JUR] || "Unknown"); if (j !== "Unknown") jurSet.add(j); });
      filteredDeliveries(f).forEach(x => { const j = String(x[JUR] || "Unknown"); if (j !== "Unknown") jurSet.add(j); });
      itas.forEach(a => a.jurisdictions.forEach(j => jurSet.add(j)));
      plns.forEach(a => a.jurisdictions.forEach(j => jurSet.add(j)));

      /* current-selection strip */
      document.getElementById("kpiSubmitted").textContent = fmtNum(submitted);
      document.getElementById("kpiCompleted").textContent = fmtNum(completed);
      document.getElementById("kpiRate").textContent = fmtPct(rate);
      document.getElementById("kpiClose").textContent = fmtNum(avgClose);
      document.getElementById("kpiInteractions").textContent = fmtNum(inter);
      document.getElementById("kpiDeliveries").textContent = fmtNum(deliv);
      document.getElementById("kpiITA").textContent = fmtNum(itas.length);
      document.getElementById("kpiPLN").textContent = fmtNum(plns.length);

      /* "Filtered:" lines on the total cards */
      setFilteredLine("fltTickets", submitted);
      setFilteredLine("fltCompleted", completed);
      setFilteredLine("fltProgress", inProgress);
      setFilteredLine("fltInteractions", inter);
      setFilteredLine("fltDeliveries", deliv);
      setFilteredLine("fltJurisdictions", jurSet.size);
      setFilteredLine("fltITA", itas.length);
      setFilteredLine("fltPLN", plns.length);

      /* filter banner */
      const d = describeSelection();
      selectionPill.textContent = d.periodTxt + " · " + d.jurTxt;
      const spacer = document.getElementById("filterSpacer");
      if (anyFilterActive()) {
        filterNote.style.display = "block";
        if (spacer) spacer.style.display = "none";
        filterNote.innerHTML = "Showing <b>" + esc(d.periodTxt) + "</b> &middot; <b>" + esc(d.jurTxt) + "</b>"
          + (d.q ? " &middot; search <b>" + esc(d.q) + "</b>" : "");
      } else {
        filterNote.style.display = "none";
        if (spacer) spacer.style.display = "block";
      }
    }

    /* ===================== ticket table ===================== */
    function renderTicketTable(){
      const rows = filteredTickets();
      ticketCountPill.textContent = String(rows.length);
      ticketTbody.innerHTML = "";
      const frag = document.createDocumentFragment();
      rows.slice(0, 800).forEach(r => {
        const tr = document.createElement("tr");
        tr.className = "clickRow";
        tr.addEventListener("click", () => {
          Array.from(ticketTbody.querySelectorAll("tr")).forEach(x => x.classList.remove("active"));
          tr.classList.add("active");
          showTicketDetail(r);
        });
        function td(val){
          const cell = document.createElement("td");
          cell.textContent = (val === null || val === undefined) ? "" : String(val);
          return cell;
        }
        function numTd(val){ const c = td(val); c.className = "num"; return c; }
        /* Targeted due date, flagged when it has passed and the ticket is still open. */
        function dueTd(row){
          const due = row["Targeted Due Date"];
          const c = td(due ? due : "\u2014");
          c.className = "nowrap";
          if (due && !row["Close Date"] && String(due) < TODAY) {
            c.classList.add("overdue");
            c.title = "Past the targeted due date and still open";
          }
          return c;
        }
        tr.appendChild(td(r["Ticket ID"]));
        const sub = td(r["Submit Date"]); sub.className = "nowrap";
        tr.appendChild(sub);
        tr.appendChild(dueTd(r));
        const jur = td(r["Jurisdiction"]);       /* raw wording, on purpose */
        jur.className = "clip jur";
        jur.title = String(r["Jurisdiction"] || "") + "  \u2192  " + String(r[JUR] || "");
        tr.appendChild(jur);
        const stc = td(r["Status"]); stc.className = "nowrap";
        tr.appendChild(stc);
        const fa = td(r["Focus Area"]);          /* raw wording, on purpose */
        fa.className = "clip";
        fa.title = String(r["Focus Area"] || "") + (r["TA Type"] ? "  ·  " + r["TA Type"] : "");
        tr.appendChild(fa);
        tr.appendChild(numTd(r["interactions_count"]));
        tr.appendChild(numTd(r["deliveries_count"]));
        frag.appendChild(tr);
      });
      ticketTbody.appendChild(frag);
    }

    function showTicketDetail(r){
      const tid = String(r["Ticket ID"] || "").trim();
      const inter = (PAYLOAD.interactions || []).filter(x => {
        const xTid = String(x["Ticket ID"] || "").trim();
        return xTid === tid && xTid !== "No Ticket ID";
      });
      const deliv = (PAYLOAD.deliveries || []).filter(x => String(x["Ticket ID"] || "").trim() === tid);

      const interHtml = recordCards(inter, "Date of Interaction", "Type of Interaction",
                                    "No interaction records found for this ticket.");
      const delHtml = recordCards(deliv, "Date of Delivery", "Type of Delivery",
                                  "No delivery records found for this ticket.");

      detailBody.innerHTML =
        '<div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;">' +
          '<span class="pill"><b>Ticket:</b> ' + esc(r["Ticket ID"]) + '</span>' +
          '<span class="pill"><b>Jurisdiction:</b> ' + esc(r[JUR]) + '</span>' +
          '<span class="pill"><b>Status:</b> ' + esc(r["Status"]) + '</span>' +
          '<span class="pill"><b>Interactions:</b> ' + esc(r["interactions_count"]) + '</span>' +
          '<span class="pill"><b>Deliveries:</b> ' + esc(r["deliveries_count"]) + '</span>' +
        '</div>' +
        '<div class="muted" style="font-size:12px;"><b>Organization:</b> ' + esc(r["Organization"]) + '</div>' +
        '<div class="muted" style="font-size:12px;"><b>Jurisdiction (as submitted):</b> ' + esc(r["Jurisdiction"]) + '</div>' +
        '<div class="muted" style="font-size:12px;"><b>Focus Area (as submitted):</b> ' + esc(r["Focus Area"]) + '</div>' +
        '<div class="muted" style="font-size:12px;"><b>Standardized Focus Area:</b> ' + esc(r["Focus Area (Standardized)"]) + '</div>' +
        '<div class="muted" style="font-size:12px;"><b>TA Type:</b> ' + esc(r["TA Type"]) + '</div>' +
        '<div class="muted" style="font-size:12px;"><b>Submit:</b> ' + dash(r["Submit Date"]) +
          ' &bull; <b>Targeted due:</b> ' + dash(r["Targeted Due Date"]) +
          ' &bull; <b>Assigned:</b> ' + dash(r["Assigned Date"]) + ' &bull; <b>Closed:</b> ' + dash(r["Close Date"]) + '</div>' +
        (r["TA Description"]
          ? '<div style="margin-top:12px; padding:10px; background:#f9fafb; border-radius:8px; border-left:3px solid #1e3a8a;">' +
            '<div style="font-size:12px; color:#6b7280; margin-bottom:4px; font-weight:600;">Description:</div>' +
            '<div style="font-size:13px; color:#333; line-height:1.5;">' + esc(r["TA Description"]) + '</div></div>'
          : "") +
        recSection("Interactions", inter.length) + interHtml +
        recSection("Deliveries", deliv.length) + delHtml;
    }

    document.getElementById("clearDetail").addEventListener("click", () => {
      Array.from(ticketTbody.querySelectorAll("tr")).forEach(x => x.classList.remove("active"));
      detailBody.innerHTML = "<div class='muted'>Click a ticket row to load Interactions and Deliveries.</div>";
    });

    /* ===================== engagement by jurisdiction ===================== */
    const jurTbody = document.getElementById("jurTbody");
    const jurCountPill = document.getElementById("jurCountPill");
    const jurDetailBody = document.getElementById("jurDetailBody");
    let activeJurisdiction = null;

    /* One row per jurisdiction: tickets submitted, interactions and deliveries placed
       there, honouring the current period / jurisdiction / search filters. */
    function jurisdictionRollup(){
      const f = getFilters();
      const rollup = {};
      function bucket(j){
        const key = String(j || "Unknown");
        if (!rollup[key]) rollup[key] = { jurisdiction: key, tickets: 0, interactions: 0, deliveries: 0 };
        return rollup[key];
      }
      filteredTickets().forEach(r => bucket(r[JUR]).tickets += 1);
      filteredInteractions(f).forEach(x => bucket(x[JUR]).interactions += 1);
      filteredDeliveries(f).forEach(x => bucket(x[JUR]).deliveries += 1);

      const rows = Object.values(rollup);
      rows.forEach(r => r.total = r.interactions + r.deliveries);
      rows.sort((a, b) => b.total - a.total || b.tickets - a.tickets || a.jurisdiction.localeCompare(b.jurisdiction));
      return rows;
    }

    function renderJurEngagementChart(rows){
      const div = "jurEngagementChart";
      const withActivity = rows.filter(r => r.total > 0);
      if (!withActivity.length) {
        fitBarHeight(div, 0, 96);
        Plotly.react(div, [], {
          title: { text: "No interactions or deliveries match the current filters", font: { size: 13, color: INK_3 } },
          plot_bgcolor: "#fff", paper_bgcolor: "#fff"
        }, CHART_CONFIG);
        return;
      }
      /* Ascending so the largest bar sits at the top of a horizontal chart. */
      const ordered = withActivity.slice().sort((a, b) => a.total - b.total);
      const names = ordered.map(r => r.jurisdiction);
      const traces = [
        { x: ordered.map(r => r.interactions), y: names, name: "Interactions", type: "bar", orientation: "h",
          marker: { color: SERIES[0], line: { color: "#fff", width: 2 } },
          hovertemplate: "<b>%{y}</b><br>Interactions: %{x}<extra></extra>" },
        { x: ordered.map(r => r.deliveries), y: names, name: "Deliveries", type: "bar", orientation: "h",
          marker: { color: SERIES[1], line: { color: "#fff", width: 2 } },
          hovertemplate: "<b>%{y}</b><br>Deliveries: %{x}<extra></extra>" }
      ];
      const totals = ordered.map(r => r.total);
      const chartH = fitBarHeight(div, ordered.length, 96);
      const layout = {
        height: chartH,
        annotations: totalAnnotations(names, totals),
        xaxis: { title: { text: "Engagement events", font: { size: 12, color: INK_2 } },
                 gridcolor: GRID, zeroline: false, tickfont: AXIS_FONT, range: axisHeadroom(totals) },
        yaxis: { tickfont: { family: AXIS_FONT.family, size: 10, color: INK_2 }, automargin: true,
                 ticklen: 4, tickcolor: "rgba(0,0,0,0)" },
        plot_bgcolor: "#fff", paper_bgcolor: "#fff",
        margin: { t: 38, r: 46, b: 52, l: 20 },
        barmode: "stack", bargap: 0.34,
        hoverlabel: HOVER, font: { family: AXIS_FONT.family, color: INK_2 },
        legend: { orientation: "h", yref: "container", y: 1, yanchor: "top", x: 0, xanchor: "left",
                  traceorder: "normal", font: { size: 11, color: INK_2 } }
      };
      Plotly.react(div, traces, layout, CHART_CONFIG);
    }

    function renderJurisdictionSection(){
      const rows = jurisdictionRollup();
      jurCountPill.textContent = String(rows.length);
      renderJurEngagementChart(rows);

      jurTbody.innerHTML = "";
      if (!rows.length) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 5; td.className = "emptyMsg";
        td.textContent = "No jurisdictions match the current filters.";
        tr.appendChild(td); jurTbody.appendChild(tr);
        return;
      }
      const frag = document.createDocumentFragment();
      rows.forEach(r => {
        const tr = document.createElement("tr");
        tr.className = "clickRow" + (r.jurisdiction === activeJurisdiction ? " active" : "");
        tr.addEventListener("click", () => {
          activeJurisdiction = r.jurisdiction;
          Array.from(jurTbody.querySelectorAll("tr")).forEach(x => x.classList.remove("active"));
          tr.classList.add("active");
          showJurisdictionDetail(r);
        });
        function cell(v, cls){
          const c = document.createElement("td");
          c.textContent = (v === null || v === undefined) ? "" : String(v);
          if (cls) c.className = cls;
          return c;
        }
        tr.appendChild(cell(r.jurisdiction));
        tr.appendChild(cell(r.tickets, "num"));
        tr.appendChild(cell(r.interactions, "num"));
        tr.appendChild(cell(r.deliveries, "num"));
        tr.appendChild(cell(r.total, "num"));
        frag.appendChild(tr);
      });
      jurTbody.appendChild(frag);

      /* Keep an open detail panel in sync with the filters. */
      if (activeJurisdiction) {
        const still = rows.find(r => r.jurisdiction === activeJurisdiction);
        if (still) showJurisdictionDetail(still); else clearJurisdictionDetail();
      }
    }

    /* Document cells hold one or more URLs, comma or whitespace separated. */
    function docLinks(value){
      const raw = String(value || "").trim();
      if (!raw) return "";
      const urls = raw.split(/[,;\s]+/).filter(u => /^https?:\/\//i.test(u));
      if (!urls.length) return '<span class="recTag">' + esc(raw) + "</span>";
      return urls.map((u, i) =>
        '<a class="recDoc" href="' + esc(u) + '" target="_blank" rel="noopener">Document' +
        (urls.length > 1 ? " " + (i + 1) : "") + "</a>").join(" ");
    }

    /* Full record, full summary. Used by both the ticket panel and the jurisdiction panel. */
    function recordCards(list, dateKey, typeKey, emptyMsg, opts){
      opts = opts || {};
      if (!list.length) return "<div class='muted' style='font-size:12.5px;'>" + emptyMsg + "</div>";
      const cards = list.map(x => {
        const tid = String(x["Ticket ID"] || "").trim();
        const showTicket = opts.showTicket && tid && tid !== "No Ticket ID";
        const noTicket = opts.showTicket && (!tid || tid === "No Ticket ID");
        const summary = String(x["Short Summary"] || "").trim();
        return '<div class="rec">' +
          '<div class="recHead">' +
            '<span class="recDate">' + esc(x[dateKey] || "—") + '</span>' +
            (x[typeKey] ? '<span class="recTag">' + esc(x[typeKey]) + '</span>' : "") +
            (showTicket ? '<span class="recTag">' + esc(tid) + '</span>' : "") +
            (noTicket ? '<span class="recTag">No ticket ID</span>' : "") +
            (opts.showJurisdiction && x[JUR] ? '<span class="recTag">' + esc(x[JUR]) + '</span>' : "") +
            docLinks(x["Document"]) +
          '</div>' +
          '<div class="recBody">' + (summary ? esc(summary) : '<span class="muted">No summary recorded.</span>') + '</div>' +
        '</div>';
      }).join("");
      return '<div class="recList">' + cards + "</div>";
    }
    function recSection(label, count){
      return '<div class="recSection"><b>' + label + '</b><span class="recCount">' + count + '</span></div>';
    }

    function showJurisdictionDetail(r){
      const f = getFilters();
      const byDate = (k) => (a, b) => String(b[k] || "").localeCompare(String(a[k] || ""));
      const inter = filteredInteractions(f).filter(x => String(x[JUR]) === r.jurisdiction).sort(byDate("Date of Interaction"));
      const deliv = filteredDeliveries(f).filter(x => String(x[JUR]) === r.jurisdiction).sort(byDate("Date of Delivery"));
      const noTicket = inter.filter(x => {
        const t = String(x["Ticket ID"] || "").trim();
        return t === "" || t === "No Ticket ID";
      }).length;

      jurDetailBody.innerHTML =
        '<div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;">' +
          '<span class="pill"><b>' + esc(r.jurisdiction) + '</b></span>' +
          '<span class="pill"><b>Tickets:</b> ' + esc(r.tickets) + '</span>' +
          '<span class="pill"><b>Interactions:</b> ' + esc(r.interactions) + '</span>' +
          '<span class="pill"><b>Deliveries:</b> ' + esc(r.deliveries) + '</span>' +
          (noTicket ? '<span class="pill"><b>Without a ticket ID:</b> ' + esc(noTicket) + '</span>' : "") +
        '</div>' +
        recSection("Interactions", inter.length) +
        recordCards(inter, "Date of Interaction", "Type of Interaction",
                    "No interactions in this selection.", { showTicket: true }) +
        recSection("Deliveries", deliv.length) +
        recordCards(deliv, "Date of Delivery", "Type of Delivery",
                    "No deliveries in this selection.", { showTicket: true });
    }

    function clearJurisdictionDetail(){
      activeJurisdiction = null;
      Array.from(jurTbody.querySelectorAll("tr")).forEach(x => x.classList.remove("active"));
      jurDetailBody.innerHTML = "<div class='muted'>Click a jurisdiction row to list its interactions and deliveries.</div>";
    }
    document.getElementById("clearJurDetail").addEventListener("click", clearJurisdictionDetail);

    /* ===================== ITA / PLN tables ===================== */
    function renderActivityTables(){
      const f = getFilters();
      const itas = filteredActivities(PAYLOAD.itas, f);
      const plns = filteredActivities(PAYLOAD.plns, f);

      function summarize(list, pill, meta, noun){
        const done = completedOnly(list).length;
        const sched = list.length - done;
        pill.textContent = String(done);
        meta.textContent = done + " completed" + (sched ? " \u00b7 " + sched + " scheduled" : "") +
                           (list.length ? "" : " \u2014 none in this selection");
      }
      summarize(itas, itaCountPill, document.getElementById("itaMeta"), "ITA");
      summarize(plns, plnCountPill, document.getElementById("plnMeta"), "PLN");

      function fill(tbody, list, kind){
        tbody.innerHTML = "";
        if (!list.length) {
          const tr = document.createElement("tr");
          const td = document.createElement("td");
          td.colSpan = 5; td.className = "muted";
          td.textContent = "No " + kind + " match the current filters.";
          tr.appendChild(td); tbody.appendChild(tr);
          return;
        }
        list.forEach(a => {
          const tr = document.createElement("tr");
          tr.className = "clickRow";
          tr.addEventListener("click", () => {
            Array.from(itaTbody.querySelectorAll("tr")).forEach(x => x.classList.remove("active"));
            Array.from(plnTbody.querySelectorAll("tr")).forEach(x => x.classList.remove("active"));
            tr.classList.add("active");
            showActivityDetail(a, kind);
          });
          function td(v){ const c = document.createElement("td"); c.textContent = (v === null || v === undefined) ? "" : String(v); return c; }
          tr.appendChild(td(a.id));
          tr.appendChild(td(a.date));
          tr.appendChild(td(kind === "ITA" ? a.location : a.focus_area));
          const jc = td(a.jurisdiction_count); jc.className = "num";
          tr.appendChild(jc);
          const st = document.createElement("td");
          const badge = document.createElement("span");
          badge.className = "badge " + (a.complete ? "done" : "sched");
          badge.textContent = a.complete ? "Complete" : (a.status || "Scheduled");
          st.appendChild(badge);
          tr.appendChild(st);
          tbody.appendChild(tr);
        });
      }
      fill(itaTbody, itas, "ITA");
      fill(plnTbody, plns, "PLN");
    }

    function showActivityDetail(a, kind){
      const accent = kind === "ITA" ? "var(--c-ita)" : "var(--c-pln)";
      const label = kind === "ITA" ? "Intensive TA" : "Peer Learning Network";
      const tags = (a.jurisdictions || []).map(j => '<span class="jtag">' + esc(j) + '</span>').join("") ||
                   '<span class="muted">No jurisdictions recorded.</span>';
      activityDetailBody.innerHTML =
        '<div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px; align-items:center;">' +
          '<span class="pill accent" style="background:' + accent + ';"><b>' + esc(label) + ' #' + esc(a.id) + '</b></span>' +
          '<span class="pill"><b>Date:</b> ' + esc(a.date) + '</span>' +
          (a.location ? '<span class="pill"><b>Location:</b> ' + esc(a.location) + '</span>' : "") +
          '<span class="badge ' + (a.complete ? "done" : "sched") + '">' + esc(a.complete ? "Complete" : (a.status || "Scheduled")) + '</span>' +
          '<span class="pill"><b>Jurisdictions:</b> ' + esc(a.jurisdiction_count) + '</span>' +
        '</div>' +
        '<div style="font-size:13px; margin-bottom:8px;"><b>Focus area(s):</b> ' + esc(a.focus_area || "—") + '</div>' +
        (a.notes ? '<div style="font-size:13px; margin-bottom:8px;"><b>Meeting notes:</b> <a href="' + esc(a.notes) + '" target="_blank" rel="noopener">' + esc(a.notes) + '</a></div>' : "") +
        '<div style="font-size:13px;"><b>Jurisdictions participating:</b><div style="margin-top:6px;">' + tags + '</div></div>';
      document.getElementById("activitySection").scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    document.getElementById("clearActivityDetail").addEventListener("click", () => {
      Array.from(itaTbody.querySelectorAll("tr")).forEach(x => x.classList.remove("active"));
      Array.from(plnTbody.querySelectorAll("tr")).forEach(x => x.classList.remove("active"));
      activityDetailBody.innerHTML = "<div class='muted'>Click an ITA or PLN row above to see its jurisdictions and focus areas.</div>";
    });

    document.getElementById("cardITA").addEventListener("click", () => {
      document.getElementById("activitySection").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    document.getElementById("cardPLN").addEventListener("click", () => {
      document.getElementById("activitySection").scrollIntoView({ behavior: "smooth", block: "start" });
    });

    /* ===================== charts ===================== */
    /* Validated categorical slots, assigned in fixed order and never cycled. */
    const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
    const INK = "#101828", INK_2 = "#475467", INK_3 = "#98a2b3", GRID = "#eceff3";

    /* Status keeps a stable slot so filtering never repaints the survivors. */
    const STATUS_ORDER = ["Completed", "In Progress", "Open", "Assigned", "Pending", "On Hold", "Closed", "Cancelled", "Unknown"];
    function statusColor(status){
      const i = STATUS_ORDER.indexOf(String(status));
      return SERIES[(i < 0 ? STATUS_ORDER.length : i) % SERIES.length];
    }
    function sortStatuses(list){
      return list.slice().sort((a, b) => {
        const ia = STATUS_ORDER.indexOf(a), ib = STATUS_ORDER.indexOf(b);
        return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
      });
    }
    const AXIS_FONT = { family: "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif", size: 11, color: INK_2 };
    const HOVER = { bgcolor: "#ffffff", bordercolor: "#e4e7ec", font: { size: 12, color: INK } };

    /* Timeline axis: every period in the data at the current granularity. */
    function timelinePeriods(gran){
      return periodsFor(gran);
    }

    function granWord(gran){
      return gran === "year" ? "by year" : (gran === "quarter" ? "by quarter" : "by month");
    }

    function baseLayout(xTitle, yTitle, gran){
      return {
        xaxis: {
          title: { text: xTitle, font: { size: 12, color: INK_2 } },
          tickangle: gran === "year" ? 0 : -45,
          gridcolor: GRID, zeroline: false, linecolor: "#e4e7ec",
          tickfont: AXIS_FONT, type: "category"
        },
        yaxis: {
          title: { text: yTitle, font: { size: 12, color: INK_2 } },
          gridcolor: GRID, zeroline: false, tickfont: AXIS_FONT, rangemode: "tozero"
        },
        plot_bgcolor: "#fff", paper_bgcolor: "#fff",
        margin: { t: 30, r: 20, b: 78, l: 62 },
        hovermode: "x unified", hoverlabel: HOVER,
        font: { family: AXIS_FONT.family, color: INK_2 },
        legend: { orientation: "h", x: 0, y: 1.16, xanchor: "left", font: { size: 11, color: INK_2 } }
      };
    }
    const CHART_CONFIG = { responsive: true, displayModeBar: false };

    /* Horizontal bar charts grow with their category count so every label stays readable. */
    /* Horizontal bar charts grow with their category count so every label stays readable.
       The height goes on the container AND in the layout, otherwise Plotly keeps the old
       canvas size when a filter shrinks the category list and the plot overruns its card. */
    function fitBarHeight(divId, n, chrome){
      const h = Math.max(300, n * 22 + (chrome || 96));
      const el = document.getElementById(divId);
      if (el) el.style.height = h + "px";
      return h;
    }

    /* Total at the end of each stacked bar. */
    function totalAnnotations(categories, totals){
      return categories.map((c, i) => ({
        x: totals[i], y: c, xref: "x", yref: "y",
        text: String(totals[i]), showarrow: false,
        xanchor: "left", yanchor: "middle", xshift: 7,
        font: { family: AXIS_FONT.family, size: 11, color: INK }
      }));
    }
    function axisHeadroom(totals){
      const max = Math.max(1, ...totals);
      return [0, max * 1.10 + 0.6];
    }

    function highlightSizes(keys, f, base){
      if (f.allPeriods) return keys.map(() => base);
      return keys.map(k => isSelectedPeriodKey(k, f) ? base + 7 : base);
    }

    function renderTrendChart(){
      const f = getFilters();
      const keys = timelinePeriods(f.gran);
      const labels = keys.map(k => periodLabel(k, f.gran));
      const rows = ticketsJurOnly(f);

      const counts = {};
      keys.forEach(k => counts[k] = 0);
      rows.forEach(r => {
        const k = periodKeyFromDate(r["Submit Date"], f.gran);
        if (k !== null && k in counts) counts[k] += 1;
      });

      const trace = {
        x: labels, y: keys.map(k => counts[k]),
        type: "scatter", mode: "lines+markers", name: "Tickets submitted",
        line: { color: SERIES[0], width: 2, shape: "linear" },
        marker: { color: SERIES[0], size: highlightSizes(keys, f, 8), line: { color: "#fff", width: 2 } },
        fill: "tozeroy", fillcolor: "rgba(42, 120, 214, 0.08)",
        hovertemplate: "Tickets: %{y}<extra></extra>"
      };
      const layout = baseLayout("", "Number of tickets", f.gran);
      layout.showlegend = false;
      layout.margin.t = 16;
      layout.height = document.getElementById("trendChart").clientHeight;
      Plotly.react("trendChart", [trace], layout, CHART_CONFIG);
    }

    function renderActivityChart(){
      const f = getFilters();
      const keys = timelinePeriods(f.gran);
      const labels = keys.map(k => periodLabel(k, f.gran));
      const jurRows = ticketsJurOnly(f);
      const ids = ticketIdSet(jurRows);

      const zero = () => { const o = {}; keys.forEach(k => o[k] = 0); return o; };
      const cInter = zero(), cDeliv = zero();

      (PAYLOAD.interactions || []).forEach(x => {
        const tid = String(x["Ticket ID"] || "").trim();
        const isOrphan = (tid === "" || tid === "No Ticket ID");
        if (!isOrphan && !ids.has(tid)) return;
        if (isOrphan && !f.allJur && !jurMatches(x["Jurisdiction"], f)) return;
        const k = periodKeyFromDate(x["Date of Interaction"], f.gran);
        if (k !== null && k in cInter) cInter[k] += 1;
      });
      (PAYLOAD.deliveries || []).forEach(x => {
        const tid = String(x["Ticket ID"] || "").trim();
        if (!f.allJur && !ids.has(tid)) return;
        const k = periodKeyFromDate(x["Date of Delivery"], f.gran);
        if (k !== null && k in cDeliv) cDeliv[k] += 1;
      });
      /* Two stacked panels sharing one x-axis. Interactions run an order of magnitude
         above deliveries, so each gets its own scale instead of a second y-axis.
         ITAs and PLNs are deliberately left out of this chart. */
      const PANELS = [
        { key: "y",  name: "Interactions", store: cInter, color: SERIES[0] },
        { key: "y2", name: "Deliveries",   store: cDeliv, color: SERIES[1] }
      ];

      const traces = PANELS.map(sr => ({
        x: labels, y: keys.map(k => sr.store[k]),
        type: "scatter", mode: "lines+markers", name: sr.name,
        yaxis: sr.key, xaxis: "x", showlegend: true,
        line: { color: sr.color, width: 2 },
        marker: { color: sr.color, size: highlightSizes(keys, f, 7), line: { color: "#fff", width: 2 } },
        hovertemplate: sr.name + ": %{y}<extra></extra>"
      }));


      const DOM = [[0.56, 1.0], [0.0, 0.44]];
      function panelAxis(domain, titleText, color){
        return {
          domain: domain, gridcolor: GRID, zeroline: false, rangemode: "tozero",
          tickfont: { family: AXIS_FONT.family, size: 10, color: INK_3 }, nticks: 4,
          title: { text: titleText, font: { size: 11, color: color } }
        };
      }
      const layout = {
        xaxis: {
          anchor: "y2", tickangle: f.gran === "year" ? 0 : -45, gridcolor: GRID,
          zeroline: false, linecolor: "#e4e7ec", tickfont: AXIS_FONT, type: "category"
        },
        yaxis:  panelAxis(DOM[0], "Interactions", SERIES[0]),
        yaxis2: panelAxis(DOM[1], "Deliveries", SERIES[1]),
        plot_bgcolor: "#fff", paper_bgcolor: "#fff",
        margin: { t: 30, r: 20, b: 78, l: 66 },
        hovermode: "x unified", hoverlabel: HOVER,
        font: { family: AXIS_FONT.family, color: INK_2 },
        legend: { orientation: "h", x: 0, y: 1.16, xanchor: "left", font: { size: 11, color: INK_2 } }
      };
      layout.height = document.getElementById("activityChart").clientHeight;
      Plotly.react("activityChart", traces, layout, CHART_CONFIG);
    }

    function stackedBar(divId, rows, groupKey, tickSize, forcedN){
      const dataMap = {};
      rows.forEach(t => {
        const g = String(t[groupKey] || "Unknown");
        const s = String(t["Status"] || "Unknown");
        if (!dataMap[g]) dataMap[g] = {};
        dataMap[g][s] = (dataMap[g][s] || 0) + 1;
      });
      const groups = Object.keys(dataMap).sort((a, b) => {
        const ta = Object.values(dataMap[a]).reduce((s, v) => s + v, 0);
        const tb = Object.values(dataMap[b]).reduce((s, v) => s + v, 0);
        return ta - tb;   /* ascending -> biggest bar lands on top in a horizontal chart */
      });
      const statuses = sortStatuses(Array.from(new Set(rows.map(t => String(t["Status"] || "Unknown")))));

      if (!groups.length) {
        fitBarHeight(divId, forcedN || 0, 96);
        Plotly.react(divId, [], {
          title: { text: "No tickets match the current filters", font: { size: 13, color: INK_3 } },
          plot_bgcolor: "#fff", paper_bgcolor: "#fff"
        }, CHART_CONFIG);
        return;
      }

      const traces = statuses.map(st => ({
        x: groups.map(g => dataMap[g][st] || 0),
        y: groups, name: st, type: "bar", orientation: "h",
        marker: { color: statusColor(st), line: { color: "#fff", width: 2 } },
        hovertemplate: "<b>%{y}</b><br>" + st + ": %{x}<extra></extra>"
      }));

      const totals = groups.map(g => Object.values(dataMap[g]).reduce((a, v) => a + v, 0));
      const chartH = fitBarHeight(divId, Math.max(groups.length, forcedN || 0), 96);
      const layout = {
        height: chartH,
        annotations: totalAnnotations(groups, totals),
        xaxis: { title: { text: "Number of tickets", font: { size: 12, color: INK_2 } },
                 gridcolor: GRID, zeroline: false, tickfont: AXIS_FONT, range: axisHeadroom(totals) },
        yaxis: { tickfont: { family: AXIS_FONT.family, size: tickSize, color: INK_2 }, automargin: true, ticklen: 4, tickcolor: "rgba(0,0,0,0)" },
        plot_bgcolor: "#fff", paper_bgcolor: "#fff",
        margin: { t: 38, r: 40, b: 52, l: 20 },
        barmode: "stack", bargap: 0.35,
        hoverlabel: HOVER, font: { family: AXIS_FONT.family, color: INK_2 },
        legend: { orientation: "h", yref: "container", y: 1, yanchor: "top", x: 0, xanchor: "left",
                  traceorder: "normal", font: { size: 11, color: INK_2 } }
      };
      Plotly.react(divId, traces, layout, CHART_CONFIG);
    }

    /* The two ticket bar charts sit side by side, so they share one height. */
    function distinctCount(rows, key){
      return new Set(rows.map(t => String(t[key] || "Unknown"))).size;
    }
    function renderTicketBarCharts(){
      const rows = filteredTickets();
      const n = Math.max(distinctCount(rows, JUR), distinctCount(rows, "Focus Area (Standardized)"));
      stackedBar("statusJurisdictionChart", rows, JUR, 10, n);
      stackedBar("focusAreaStatusChart", rows, "Focus Area (Standardized)", 11, n);
    }

    /* ===================== jurisdiction dropdown ===================== */
    function jurisdictionList(){
      return (PAYLOAD.filter_jurisdictions || PAYLOAD.jurisdictions || []).slice().sort();
    }

    function renderJurOptions(filterText){
      const jurs = jurisdictionList();
      const filterLower = String(filterText || "").toLowerCase();
      jurOptions.innerHTML = "";

      if (!filterLower || "all jurisdictions".includes(filterLower)) {
        const allDiv = document.createElement("div");
        allDiv.className = "jur-checkbox";
        const allCb = document.createElement("input");
        allCb.type = "checkbox"; allCb.value = "All jurisdictions";
        allCb.checked = selectedJurisdictions.has("All jurisdictions");
        allCb.onchange = function(){
          selectedJurisdictions.clear();
          selectedJurisdictions.add("All jurisdictions");
          renderJurOptions(jurSearch.value);
          updateJurSelection();
        };
        allDiv.appendChild(allCb);
        allDiv.appendChild(document.createTextNode("All jurisdictions"));
        jurOptions.appendChild(allDiv);
      }

      jurs.forEach(j => {
        if (filterLower && !j.toLowerCase().includes(filterLower)) return;
        const div = document.createElement("div");
        div.className = "jur-checkbox";
        const cb = document.createElement("input");
        cb.type = "checkbox"; cb.value = j;
        cb.checked = selectedJurisdictions.has(j);
        cb.onchange = function(){
          if (this.checked) {
            selectedJurisdictions.delete("All jurisdictions");
            selectedJurisdictions.add(j);
          } else {
            selectedJurisdictions.delete(j);
            if (selectedJurisdictions.size === 0) selectedJurisdictions.add("All jurisdictions");
          }
          renderJurOptions(jurSearch.value);
          updateJurSelection();
        };
        div.appendChild(cb);
        div.appendChild(document.createTextNode(j));
        jurOptions.appendChild(div);
      });
    }

    function updateJurSelection(){
      const selected = Array.from(selectedJurisdictions);
      if (selected.length === 0 || (selected.length === 1 && selected[0] === "All jurisdictions")) {
        jurSelectText.textContent = "All jurisdictions";
        jurSel.value = "All jurisdictions";
      } else if (selected.length === 1) {
        jurSelectText.textContent = selected[0];
        jurSel.value = selected[0];
      } else {
        jurSelectText.textContent = selected.length + " selected";
        jurSel.value = selected.join(",");
      }
      rerender();
    }

    jurSelectBox.addEventListener("click", function(e){
      e.stopPropagation();
      jurDropdown.style.display = (jurDropdown.style.display === "none" || !jurDropdown.style.display) ? "block" : "none";
    });
    jurSearch.addEventListener("input", function(){ renderJurOptions(jurSearch.value); });
    jurSearch.addEventListener("click", function(e){ e.stopPropagation(); });
    jurDropdown.addEventListener("click", function(e){ e.stopPropagation(); });
    document.addEventListener("click", function(e){
      if (!jurSelectBox.contains(e.target) && !jurDropdown.contains(e.target)) jurDropdown.style.display = "none";
    });

    /* ===================== wiring ===================== */
    function rerender(){
      setKPIs();
      renderTicketTable();
      renderJurisdictionSection();
      renderActivityTables();
      renderTrendChart();
      renderActivityChart();
      renderTicketBarCharts();
    }

    const granSeg = document.getElementById("granSeg");
    function setGranularity(value){
      granSel.value = value;
      Array.from(granSeg.querySelectorAll("button")).forEach(b =>
        b.setAttribute("aria-pressed", String(b.dataset.gran === value)));
      initPeriodOptions();
      rerender();
    }
    yearSel.addEventListener("change", function(){ initPeriodOptions(); rerender(); });
    granSeg.addEventListener("click", function(e){
      const btn = e.target.closest("button[data-gran]");
      if (btn) setGranularity(btn.dataset.gran);
    });
    granSel.addEventListener("change", function(){ setGranularity(granSel.value); });
    periodSel.addEventListener("change", rerender);
    ticketSearch.addEventListener("input", rerender);
    resetFilters.addEventListener("click", function(){
      granSel.value = "month";
      Array.from(granSeg.querySelectorAll("button")).forEach(b =>
        b.setAttribute("aria-pressed", String(b.dataset.gran === "month")));
      clearJurisdictionDetail();
      initYearOptions();
      initPeriodOptions();
      yearSel.value = ALL_PERIODS;
      initPeriodOptions();
      periodSel.value = ALL_PERIODS;
      selectedJurisdictions = new Set(["All jurisdictions"]);
      jurSearch.value = "";
      renderJurOptions("");
      ticketSearch.value = "";
      updateJurSelection();
    });

    initYearOptions();
    initPeriodOptions();
    renderJurOptions("");
    updateJurSelection();
    setTotals();
    rerender();
  </script>
  </div>
</body>
</html>
"""


def _json_safe(value: Any) -> Any:
    """
    Replace NaN / NaT with null so the embedded JSON is valid and empty dates
    render as blanks rather than the string "NaN".
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def build_html(payload: dict, title: str, logo_path: str = "logo.png") -> str:
    data_json = json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    current_date = datetime.now().strftime("%b %d, %Y")

    logo_block = ""
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                logo_data = f.read()
            ext = Path(logo_path).suffix.lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            b64 = base64.b64encode(logo_data).decode("utf-8")
            logo_block = f'<img src="data:{mime};base64,{b64}" alt="Logo" class="header-logo">'
        except Exception:
            logo_block = ""

    # Plotly.js, inlined so the file works with no network.
    plotly_snippet = pio.to_html(
        {"data": [], "layout": {}},
        include_plotlyjs="inline",
        full_html=False,
    )

    html = HTML_TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__CURRENT_DATE__", current_date)
    html = html.replace("__LOGO_BLOCK__", logo_block)
    html = html.replace("__PLOTLY_SNIPPET__", plotly_snippet)
    html = html.replace("__DATA_JSON__", data_json)
    return html


# -----------------------------
# D) Main
# -----------------------------
DEFAULT_EXTRA_XLSX = "HRSA064 Additional TA activities.xlsx"


def build_dashboard(xlsx_path: str, out_html: str, extra_xlsx: str | None = None) -> str:
    sheets = read_hrsa_workbook(xlsx_path)

    df_main = normalize_main(sheets.get("Main", pd.DataFrame()))
    df_inter = normalize_interactions(sheets.get("Interaction", pd.DataFrame()))
    df_del = normalize_deliveries(sheets.get("Delivery", pd.DataFrame()))

    # ITA / PLN workbook: default to a file of that name sitting next to the TA request file.
    if extra_xlsx is None:
        extra_xlsx = os.path.join(os.path.dirname(os.path.abspath(xlsx_path)), DEFAULT_EXTRA_XLSX)
    extra = read_extra_activities(extra_xlsx)

    payload = compute_payload(
        df_main, df_inter, df_del,
        df_ita=extra["ita"], df_pln=extra["pln"],
        id_col="Ticket ID",
    )
    # Look for logo.png in the same directory as the Excel file
    logo_path = os.path.join(os.path.dirname(xlsx_path), "logo.png")
    if not os.path.exists(logo_path):
        logo_path = "logo.png"  # Fallback to current directory
    html = build_html(payload, title="HRSA Monthly Report Dashboard", logo_path=logo_path)

    Path(out_html).write_text(html, encoding="utf-8")
    return out_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate HRSA Monthly Report Dashboard from Excel file"
    )
    parser.add_argument(
        "--xlsx",
        default="HRSA64_TA_Request.xlsx",
        help="Path to HRSA64_TA_Request.xlsx (default: HRSA64_TA_Request.xlsx)"
    )
    parser.add_argument(
        "--extra",
        default=None,
        help=(
            "Path to 'HRSA064 Additional TA activities.xlsx' (ITA + PLN data). "
            "Defaults to a file of that name next to --xlsx."
        ),
    )
    parser.add_argument(
        "--out",
        default="HRSA_monthly_dashboard.html",
        help="Output HTML filename (default: HRSA_monthly_dashboard.html)"
    )
    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.xlsx):
        print(f"✗ Error: Excel file not found: {args.xlsx}", file=sys.stderr)
        print(f"\nUsage: {sys.argv[0]} --xlsx <path_to_excel_file> [--out <output_file>]", file=sys.stderr)
        sys.exit(1)

    # Append date suffix to output filename if not already present
    out_html = args.out
    if "__" not in Path(out_html).stem:
        # Get current date in yyyymmdd format
        date_suffix = datetime.now().strftime("__%Y%m%d")
        # Split filename into name and extension
        path = Path(out_html)
        out_html = str(path.parent / f"{path.stem}{date_suffix}{path.suffix}")

    try:
        out = build_dashboard(args.xlsx, out_html, extra_xlsx=args.extra)
        print("✓ Successfully wrote:", out)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)