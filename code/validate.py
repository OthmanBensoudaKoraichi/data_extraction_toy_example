"""
validate.py — Compare pipeline output (results.csv) against hand-coded Qualtrics CSV.

Usage:
    python validate.py --pipeline results.csv --qualtrics Eviction_Coding_May_14__2026_15_31.csv

Checks Tier 1 fields (service_date, service_method, case_number) and
Tier 2 fields (num_defendants, process_server_name, service_company_name,
mailing_postmark_date) for each case.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize_date(d):
    """Normalize date strings to MM/DD/YYYY for comparison."""
    if pd.isna(d) or d is None or str(d).strip() == "":
        return None
    d = str(d).strip()
    # Handle M/D/YY, M/D/YYYY, MM/DD/YY, MM/DD/YYYY
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", d)
    if m:
        month, day, year = m.groups()
        if len(year) == 2:
            year = "20" + year
        return f"{int(month):02d}/{int(day):02d}/{year}"
    return d


def normalize_method(m):
    """Normalize service method strings."""
    if pd.isna(m) or m is None:
        return None
    m = str(m).strip().lower()
    mapping = {
        "personal service": "personal",
        "personal": "personal",
        "substitute service": "substitute",
        "substitute": "substitute",
        "substituted": "substitute",
        "service by posting": "posting",
        "posting": "posting",
        "mail": "posting",
        "safe at home program": "safe_at_home",
        "safe_at_home": "safe_at_home",
    }
    return mapping.get(m, m)


def normalize_str(s):
    """Normalize a string for fuzzy comparison."""
    if pd.isna(s) or s is None:
        return None
    return str(s).strip().lower()


def normalize_case_number(cn):
    """Normalize case numbers to handle typos like '25AVUD001171' vs '25AVUD01171'."""
    if cn is None or pd.isna(cn) or str(cn).strip() in ("", "nan"):
        return None
    cn = str(cn).strip().upper()
    # Format: digits + letters + digits (e.g., 25AVUD01171)
    # Strip leading zeros from the trailing numeric part
    m = re.match(r"(\d+[A-Z]+)(\d+)", cn)
    if m:
        return m.group(1) + str(int(m.group(2)))
    return cn


# ── Parse Qualtrics CSV ─────────────────────────────────────────────────────

def parse_qualtrics(path):
    """Parse the Qualtrics export into a list of case dicts."""
    df = pd.read_csv(path, skiprows=[1, 2])  # skip header description rows
    cases = []
    for _, row in df.iterrows():
        case_num = str(row.get("Q1.2", "")).strip()
        if not case_num or case_num.startswith("{"):
            continue

        # Defendants (from Q9 columns — up to 5)
        defendants = []
        for j in range(1, 6):
            method = row.get(f"{j}_Q9.1")
            if pd.isna(method):
                break
            defendants.append({
                "service_method": normalize_method(method),
                "service_address": str(row.get(f"{j}_Q9.2", "")).strip() if not pd.isna(row.get(f"{j}_Q9.2")) else None,
                "service_date": normalize_date(row.get(f"{j}_Q9.3")),
                "service_time": str(row.get(f"{j}_Q9.4", "")).strip() if not pd.isna(row.get(f"{j}_Q9.4")) else None,
                "co_occupant_description": str(row.get(f"{j}_Q9.5", "")).strip() if not pd.isna(row.get(f"{j}_Q9.5")) else None,
                "mailing_postmark_date": normalize_date(row.get(f"{j}_Q9.6")),
            })

        # Service attempts (from Q5 columns — application for posting)
        attempts_app = []
        for j in range(1, 11):
            date = row.get(f"{j}_Q5.1")
            if pd.isna(date):
                break
            attempts_app.append({
                "attempt_date": normalize_date(date),
                "attempt_time": str(row.get(f"{j}_Q5.2", "")).strip() if not pd.isna(row.get(f"{j}_Q5.2")) else None,
                "description": str(row.get(f"{j}_Q5.3", "")).strip() if not pd.isna(row.get(f"{j}_Q5.3")) else None,
            })

        # Service attempts (from Q12 columns — proof of service)
        attempts_pos = []
        for j in range(1, 11):
            date = row.get(f"{j}_Q12.1")
            if pd.isna(date):
                break
            attempts_pos.append({
                "attempt_date": normalize_date(date),
                "attempt_time": str(row.get(f"{j}_Q12.2", "")).strip() if not pd.isna(row.get(f"{j}_Q12.2")) else None,
                "description": str(row.get(f"{j}_Q12.3", "")).strip() if not pd.isna(row.get(f"{j}_Q12.3")) else None,
            })

        cases.append({
            "case_number": case_num,
            "has_application_for_posting": str(row.get("Q2.1", "")).strip().lower() == "yes",
            "application_filed_date": normalize_date(row.get("Q3.2")),
            "application_granted_date": normalize_date(row.get("Q3.3")),
            "judge_name": str(row.get("Q3.4", "")).strip() if not pd.isna(row.get("Q3.4")) else None,
            "judge_department": str(row.get("Q3.5", "")).strip() if not pd.isna(row.get("Q3.5")) else None,
            "process_server_name_app": str(row.get("Q6.2", "")).strip() if not pd.isna(row.get("Q6.2")) else None,
            "service_company_name_app": str(row.get("Q6.3", "")).strip() if not pd.isna(row.get("Q6.3")) else None,
            "process_server_registration_app": str(row.get("Q6.4", "")).strip() if not pd.isna(row.get("Q6.4")) else None,
            "service_charge_app": str(row.get("Q6.5", "")).strip() if not pd.isna(row.get("Q6.5")) else None,
            "num_named_defendants": int(row["Q7.1"]) if not pd.isna(row.get("Q7.1")) else None,
            "has_proof_of_service": str(row.get("Q8.1", "")).strip().lower() == "yes",
            "defendants": defendants,
            "process_server_name_pos": str(row.get("Q10.2", "")).strip() if not pd.isna(row.get("Q10.2")) else None,
            "service_company_name_pos": str(row.get("Q10.3", "")).strip() if not pd.isna(row.get("Q10.3")) else None,
            "process_server_registration_pos": str(row.get("Q10.4", "")).strip() if not pd.isna(row.get("Q10.4")) else None,
            "service_charge_pos": str(row.get("Q10.5", "")).strip() if not pd.isna(row.get("Q10.5")) else None,
            "attempts_app": attempts_app,
            "attempts_pos": attempts_pos,
            "has_unnamed_defendants": str(row.get("Q13.1", "")).strip().lower() == "yes",
            "unnamed_service_method": normalize_method(row.get("Q14.1")),
            "unnamed_mailing_postmark_date": normalize_date(row.get("Q14.2")),
            "comments": str(row.get("Q15.1", "")).strip() if not pd.isna(row.get("Q15.1")) else None,
        })

    return cases


# ── Parse pipeline CSV ──────────────────────────────────────────────────────

def parse_pipeline(path):
    """Parse the pipeline results.csv into a list of case dicts."""
    df = pd.read_csv(path)
    cases = []
    for _, row in df.iterrows():
        defendants = json.loads(row.get("defendants", "[]")) if not pd.isna(row.get("defendants")) else []
        for d in defendants:
            d["service_method"] = normalize_method(d.get("service_method"))
            d["service_date"] = normalize_date(d.get("service_date"))
            d["mailing_postmark_date"] = normalize_date(d.get("mailing_postmark_date"))

        attempts = json.loads(row.get("attempts", "[]")) if not pd.isna(row.get("attempts")) else []
        for a in attempts:
            a["attempt_date"] = normalize_date(a.get("attempt_date"))

        cases.append({
            "case_number": str(row.get("case_number", "")).strip(),
            "file": str(row.get("_file", "")),
            "pos_type": normalize_method(row.get("pos_type")),
            "has_application_for_posting": row.get("has_application_for_posting", False),
            "has_proof_of_service": row.get("has_proof_of_service", False),
            "application_filed_date": normalize_date(row.get("application_filed_date")),
            "application_granted_date": normalize_date(row.get("application_granted_date")),
            "judge_name": str(row.get("judge_name", "")).strip() if not pd.isna(row.get("judge_name")) else None,
            "judge_department": str(row.get("judge_department", "")).strip() if not pd.isna(row.get("judge_department")) else None,
            "process_server_name": str(row.get("process_server_name", "")).strip() if not pd.isna(row.get("process_server_name")) else None,
            "service_company_name": str(row.get("service_company_name", "")).strip() if not pd.isna(row.get("service_company_name")) else None,
            "process_server_registration": str(row.get("process_server_registration", "")).strip() if not pd.isna(row.get("process_server_registration")) else None,
            "service_charge": str(row.get("service_charge", "")).strip() if not pd.isna(row.get("service_charge")) else None,
            "num_named_defendants": int(row["num_named_defendants"]) if not pd.isna(row.get("num_named_defendants")) else None,
            "defendants": defendants,
            "has_unnamed_defendants": row.get("has_unnamed_defendants", False),
            "unnamed_service_method": normalize_method(row.get("unnamed_service_method")),
            "unnamed_mailing_postmark_date": normalize_date(row.get("unnamed_mailing_postmark_date")),
            "num_attempts": int(row["num_attempts"]) if not pd.isna(row.get("num_attempts")) else 0,
            "attempts": attempts,
        })
    return cases


# ── Compare ─────────────────────────────────────────────────────────────────

def compare_field(label, human_val, pipeline_val, tier, results):
    """Compare a single field and record the result."""
    match = False
    if human_val is None and pipeline_val is None:
        match = True
    elif human_val is not None and pipeline_val is not None:
        if isinstance(human_val, str) and isinstance(pipeline_val, str):
            match = human_val.lower().strip() == pipeline_val.lower().strip()
        else:
            match = human_val == pipeline_val
    results.append((label, tier, human_val, pipeline_val, match))


def merge_pipeline_rows(p_rows):
    """Merge multiple pipeline rows for the same case into one case-level record.
    Combines defendants and attempts from all documents, takes non-null values."""
    merged = dict(p_rows[0])
    merged["_files"] = [p.get("file", "?") for p in p_rows]
    all_defendants = []
    all_attempts = []
    for p in p_rows:
        all_defendants.extend(p.get("defendants", []))
        all_attempts.extend(p.get("attempts", []))
        # Take first non-null for scalar fields
        for key in ["process_server_name", "service_company_name",
                     "process_server_registration", "service_charge"]:
            if merged.get(key) is None and p.get(key) is not None:
                merged[key] = p[key]
        # OR together boolean fields
        if p.get("has_unnamed_defendants"):
            merged["has_unnamed_defendants"] = True
            merged["unnamed_service_method"] = p.get("unnamed_service_method") or merged.get("unnamed_service_method")
            merged["unnamed_mailing_postmark_date"] = p.get("unnamed_mailing_postmark_date") or merged.get("unnamed_mailing_postmark_date")
    # Deduplicate defendants by name
    seen = set()
    unique_defs = []
    for d in all_defendants:
        name = d.get("defendant_name", "").strip().upper()
        if name and name not in seen and name != "ALL OTHER UNNAMED OCCUPANTS":
            seen.add(name)
            unique_defs.append(d)
    merged["defendants"] = unique_defs
    merged["num_named_defendants"] = len(unique_defs)
    merged["attempts"] = all_attempts
    merged["num_attempts"] = len(all_attempts)
    return merged


def run_comparison(qualtrics_cases, pipeline_cases):
    """Compare pipeline output against hand-coded data, case by case."""

    # Group pipeline rows by normalized case number (may have multiple docs per case)
    pipeline_by_case = {}
    for p in pipeline_cases:
        cn = normalize_case_number(p["case_number"])
        if cn is None:
            continue
        if cn not in pipeline_by_case:
            pipeline_by_case[cn] = []
        pipeline_by_case[cn].append(p)

    total_checks = 0
    total_matches = 0
    tier1_checks = 0
    tier1_matches = 0

    for q in qualtrics_cases:
        cn = q["case_number"]
        cn_norm = normalize_case_number(cn)
        print(f"\n{'='*70}")
        print(f"  CASE: {cn}")
        print(f"{'='*70}")

        p_rows = pipeline_by_case.get(cn_norm, [])
        if not p_rows:
            print(f"  ⚠ No pipeline output found for this case!")
            continue

        # Merge all pipeline rows for this case into one record
        p = merge_pipeline_rows(p_rows)

        print(f"  Pipeline files: {p.get('_files', ['?'])}")
        print(f"  Human coder noted: app_for_posting={q['has_application_for_posting']}, has_pos={q['has_proof_of_service']}")
        print()

        results = []

        # --- TIER 1: Critical for regression ---
        print("  TIER 1 — Critical (determines outcome variable / treatment)")
        print("  " + "-"*66)

        # Service method & date per defendant
        q_defs = q["defendants"]
        p_defs = p["defendants"]

        if len(q_defs) == 0 and len(p_defs) == 0:
            print("  (No defendants in either source)")
        else:
            for j in range(max(len(q_defs), len(p_defs))):
                qd = q_defs[j] if j < len(q_defs) else {}
                pd_ = p_defs[j] if j < len(p_defs) else {}
                prefix = f"  Defendant {j+1}"
                compare_field(f"{prefix} service_method", qd.get("service_method"), pd_.get("service_method"), 1, results)
                # Qualtrics only asks for service_date on personal service (display logic).
                # If the human coded non-personal, they were never shown the date field,
                # so we can't compare. Only check date if human method is personal or
                # if human has no method (defendant missing from human data).
                q_method = qd.get("service_method")
                if q_method == "personal" or q_method is None:
                    compare_field(f"{prefix} service_date", qd.get("service_date"), pd_.get("service_date"), 1, results)
                else:
                    print(f"  ⊘  {prefix} service_date — skipped (Qualtrics only asks for personal service)")

        for label, tier, hv, pv, match in results:
            if tier == 1:
                symbol = "✓" if match else "✗"
                print(f"  {symbol}  {label}")
                if not match:
                    print(f"       Human:    {hv}")
                    print(f"       Pipeline: {pv}")

        # --- TIER 2: Important ---
        print()
        print("  TIER 2 — Important (robustness / heterogeneity)")
        print("  " + "-"*66)

        compare_field("num_named_defendants", q["num_named_defendants"], p["num_named_defendants"], 2, results)

        # Process server: compare against whichever source the qualtrics has
        q_server = q["process_server_name_pos"] or q["process_server_name_app"]
        compare_field("process_server_name", normalize_str(q_server), normalize_str(p.get("process_server_name")), 2, results)

        q_company = q["service_company_name_pos"] or q["service_company_name_app"]
        compare_field("service_company_name", normalize_str(q_company), normalize_str(p.get("service_company_name")), 2, results)

        # Mailing postmark per defendant
        for j in range(max(len(q_defs), len(p_defs))):
            qd = q_defs[j] if j < len(q_defs) else {}
            pd_ = p_defs[j] if j < len(p_defs) else {}
            compare_field(f"Defendant {j+1} mailing_postmark_date", qd.get("mailing_postmark_date"), pd_.get("mailing_postmark_date"), 2, results)

        compare_field("has_unnamed_defendants", q["has_unnamed_defendants"], p["has_unnamed_defendants"], 2, results)
        compare_field("unnamed_service_method", q["unnamed_service_method"], normalize_method(p.get("unnamed_service_method")), 2, results)
        compare_field("unnamed_mailing_postmark_date", q["unnamed_mailing_postmark_date"], p.get("unnamed_mailing_postmark_date"), 2, results)

        for label, tier, hv, pv, match in results:
            if tier == 2:
                symbol = "✓" if match else "✗"
                print(f"  {symbol}  {label}")
                if not match:
                    print(f"       Human:    {hv}")
                    print(f"       Pipeline: {pv}")

        # Tally
        for _, tier, _, _, match in results:
            total_checks += 1
            total_matches += 1 if match else 0
            if tier == 1:
                tier1_checks += 1
                tier1_matches += 1 if match else 0

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Tier 1 accuracy: {tier1_matches}/{tier1_checks} ({100*tier1_matches/tier1_checks:.0f}%)" if tier1_checks else "  Tier 1: no checks")
    print(f"  Overall accuracy: {total_matches}/{total_checks} ({100*total_matches/total_checks:.0f}%)" if total_checks else "  Overall: no checks")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate pipeline output against hand-coded Qualtrics data")
    parser.add_argument("--pipeline", type=Path, required=True, help="Pipeline results.csv")
    parser.add_argument("--qualtrics", type=Path, required=True, help="Qualtrics export CSV")
    args = parser.parse_args()

    print("Loading Qualtrics (hand-coded) data...")
    q_cases = parse_qualtrics(args.qualtrics)
    print(f"  Found {len(q_cases)} cases: {[c['case_number'] for c in q_cases]}")

    print("Loading pipeline output...")
    p_cases = parse_pipeline(args.pipeline)
    print(f"  Found {len(p_cases)} rows: {[c['case_number'] for c in p_cases]}")

    run_comparison(q_cases, p_cases)


if __name__ == "__main__":
    main()