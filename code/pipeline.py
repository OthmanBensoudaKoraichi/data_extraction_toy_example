"""
Othman's pipeline.ipynb converted to .py with new nodes added.
Original: 2 nodes (ask_pos_type, ask_attorney_fees)
Added:    6 nodes (ask_case_number, ask_application_info, ask_process_server,
                   ask_defendants, ask_unnamed_defendants, ask_service_attempts)
"""

# ── Cell 2: Imports + PDF → Markdown (UNCHANGED) ────────────────────────────
from __future__ import annotations
import os
import base64
from pathlib import Path
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path.cwd()

def pdf_to_markdown_pages(
    pdf_path: str | Path,
    *,
    endpoint: str | None = None,
    key: str | None = None,
) -> list[str]:
    """Return one Markdown string per PDF page using Azure Document Intelligence."""
    endpoint = endpoint or os.getenv("AZURE_ENDPOINT")
    key = key or os.getenv("AZURE_KEY")
    if not endpoint or not key:
        raise ValueError("Missing AZURE_ENDPOINT or AZURE_KEY.")

    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

    pdf_bytes = Path(pdf_path).read_bytes()
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body={"base64Source": base64.b64encode(pdf_bytes).decode()},
        output_content_format=DocumentContentFormat.MARKDOWN,
    )

    result = poller.result()

    full_md = result.content or ""
    pages_md: list[str] = []
    for page in sorted(result.pages, key=lambda p: p.page_number):
        if not page.spans:
            pages_md.append("")
            continue
        start = min(s.offset for s in page.spans)
        end = max(s.offset + s.length for s in page.spans)
        pages_md.append(full_md[start:end])

    return pages_md


# ── Cell 6: Schemas, chains, state, nodes, graph ────────────────────────────

from typing import Optional, Any, TypedDict
import time
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph


def build_graph(md_content: str):

    # --- LLM (UNCHANGED)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # --- Prompt template (UNCHANGED)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a legal document analyst. Answer based only on the document provided."),
        ("human", "Document:\n{document}\n\nQuestion: {question}"),
    ])

    # --- Structured output schemas (ORIGINAL) ---
    class PosType(BaseModel):
        pos_type: str = Field(description="Type of proof of service (personal, substituted, mail, etc.)")

    class AttorneyFees(BaseModel):
        attorney_fees: int = Field(description="Attorney fees amount in dollars as an integer")

    # --- Structured output schemas (NEW) ---

    class CaseNumber(BaseModel):
        case_number: str = Field(description="Case number as printed on the document (e.g., '25AVUD01285').")

    class ApplicationInfo(BaseModel):
        has_application_for_posting: bool = Field(default=False, description="True if this document is or contains an 'Application for Service by Posting'.")
        application_filed_date: Optional[str] = Field(default=None, description="Date the application was filed, MM/DD/YYYY.")
        application_granted_date: Optional[str] = Field(default=None, description="Date the application was granted, MM/DD/YYYY.")
        judge_name: Optional[str] = Field(default=None, description="Name of the judge who granted the application.")
        judge_department: Optional[str] = Field(default=None, description="Division or department of the judge.")
        has_proof_of_service: bool = Field(default=False, description="True if this document is or contains a proof of service.")

    class ProcessServer(BaseModel):
        process_server_name: Optional[str] = Field(default=None, description="Full name of the process server / declarant.")
        service_company_name: Optional[str] = Field(default=None, description="Name of the process-serving company.")
        process_server_registration: Optional[str] = Field(default=None, description="Registration number of the process server.")
        service_charge: Optional[str] = Field(default=None, description="Amount charged for service, as written (e.g., '210').")

    class DefendantService(BaseModel):
        defendant_name: str = Field(description="Full name of the defendant. 'REDACTED' if redacted.")
        service_method: str = Field(description="One of: 'personal', 'substitute', 'posting', 'safe_at_home'.")
        service_address: str = Field(description="Full address where served.")
        service_date: Optional[str] = Field(default=None, description="Date served, MM/DD/YYYY.")
        service_time: Optional[str] = Field(default=None, description="Time served, HH:MM am/pm.")
        co_occupant_description: Optional[str] = Field(default=None, description="For substitute service: co-occupant name and description.")
        mailing_postmark_date: Optional[str] = Field(default=None, description="For substitute/posting: mailed copy postmark date, MM/DD/YYYY.")

    class DefendantsList(BaseModel):
        num_named_defendants: int = Field(description="Total number of named defendants.")
        defendants: list[DefendantService] = Field(description="Service details for each named defendant.")

    class UnnamedDefendants(BaseModel):
        has_unnamed_defendants: bool = Field(default=False, description="True if document lists unnamed defendants.")
        unnamed_service_method: Optional[str] = Field(default=None, description="'substitute' or 'posting'.")
        unnamed_mailing_postmark_date: Optional[str] = Field(default=None, description="Postmark date, MM/DD/YYYY.")

    class ServiceAttempt(BaseModel):
        attempt_date: str = Field(description="Date of the attempt, MM/DD/YYYY.")
        attempt_time: Optional[str] = Field(default=None, description="Time, HH:MM am/pm.")
        description: str = Field(description="What happened. Replace defendant names with REDACTED.")

    class ServiceAttemptsList(BaseModel):
        num_attempts: int = Field(default=0, description="Number of prior/failed service attempts. 0 if none.")
        attempts: list[ServiceAttempt] = Field(default_factory=list, description="Details of each attempt.")

    # --- Structured chains (ORIGINAL) ---
    pos_chain = prompt | llm.with_structured_output(PosType)
    fees_chain = prompt | llm.with_structured_output(AttorneyFees)

    # --- Structured chains (NEW) ---
    case_chain = prompt | llm.with_structured_output(CaseNumber)
    app_chain = prompt | llm.with_structured_output(ApplicationInfo)
    server_chain = prompt | llm.with_structured_output(ProcessServer)
    defendants_chain = prompt | llm.with_structured_output(DefendantsList)
    unnamed_chain = prompt | llm.with_structured_output(UnnamedDefendants)
    attempts_chain = prompt | llm.with_structured_output(ServiceAttemptsList)

    # --- State (ORIGINAL + NEW fields) ---
    class State(TypedDict, total=False):
        # ORIGINAL
        pos_type: Optional[str]
        attorney_fees: Optional[int]
        # NEW
        case_number: Optional[str]
        has_application_for_posting: Optional[bool]
        application_filed_date: Optional[str]
        application_granted_date: Optional[str]
        judge_name: Optional[str]
        judge_department: Optional[str]
        has_proof_of_service: Optional[bool]
        process_server_name: Optional[str]
        service_company_name: Optional[str]
        process_server_registration: Optional[str]
        service_charge: Optional[str]
        num_named_defendants: Optional[int]
        defendants: Optional[list]
        has_unnamed_defendants: Optional[bool]
        unnamed_service_method: Optional[str]
        unnamed_mailing_postmark_date: Optional[str]
        num_attempts: Optional[int]
        attempts: Optional[list]

    # --- Async nodes (ORIGINAL) ---

    async def start(state: State) -> State:
        return state

    async def ask_pos_type(state: State) -> State:
        t0 = time.perf_counter()
        res = await pos_chain.ainvoke({
            "document": md_content,
            "question": "What type of proof of service is this? (personal, substituted, mail, etc.)",
        })
        print(f"  ⏱ ask_pos_type: {time.perf_counter() - t0:.2f}s")
        return {"pos_type": res.pos_type}

    async def ask_attorney_fees(state: State) -> State:
        t0 = time.perf_counter()
        res = await fees_chain.ainvoke({
            "document": md_content,
            "question": "What are the attorney fees? Return the amount in dollars.",
        })
        print(f"  ⏱ ask_attorney_fees: {time.perf_counter() - t0:.2f}s")
        return {"attorney_fees": res.attorney_fees}

    # --- Async nodes (NEW) ---

    async def ask_case_number(state: State) -> State:
        t0 = time.perf_counter()
        res = await case_chain.ainvoke({
            "document": md_content,
            "question": "What is the case number on this document?",
        })
        print(f"  ⏱ ask_case_number: {time.perf_counter() - t0:.2f}s")
        return {"case_number": res.case_number}

    async def ask_application_info(state: State) -> State:
        t0 = time.perf_counter()
        res = await app_chain.ainvoke({
            "document": md_content,
            "question": (
                "Answer the following:\n"
                "1. Is this document (or does it contain) an 'Application for Service by Posting'?\n"
                "2. If yes: what date was the application filed with the court? Look for the "
                "court's 'FILED' stamp (not the date the declarant signed). (MM/DD/YYYY)\n"
                "3. If yes: what date was the application granted? This is the date on the "
                "court's order, near the judge's signature. (MM/DD/YYYY)\n"
                "4. If yes: what is the name of the judge? Look near the judicial officer "
                "signature at the bottom of the order section.\n"
                "5. If yes: what division or department NUMBER is the judge in? Return the "
                "actual number or code (e.g., 'F44', 'A9'), NOT the label 'DIV. / DEPT.'\n"
                "6. Is this document (or does it contain) a proof of service (POS-010 form "
                "documenting completed service)? An application for posting is NOT a proof of service."
            ),
        })
        print(f"  ⏱ ask_application_info: {time.perf_counter() - t0:.2f}s")
        return {
            "has_application_for_posting": res.has_application_for_posting,
            "application_filed_date": res.application_filed_date,
            "application_granted_date": res.application_granted_date,
            "judge_name": res.judge_name,
            "judge_department": res.judge_department,
            "has_proof_of_service": res.has_proof_of_service,
        }

    async def ask_process_server(state: State) -> State:
        t0 = time.perf_counter()
        res = await server_chain.ainvoke({
            "document": md_content,
            "question": (
                "Extract the PROCESS SERVER information — the person who physically "
                "served or attempted to serve the court documents. This is typically the "
                "declarant who signs at the bottom of the document, often identified by "
                "a registration number and/or a service company name.\n\n"
                "The process server is NOT the attorney representing the plaintiff. "
                "Attorneys are typically listed in the header with 'Attorney for Plaintiff' "
                "or 'Esq.' The process server is listed separately, usually near the end.\n\n"
                "1. Full name of the process server\n"
                "2. Name of the service company (if any)\n"
                "3. Registration number of the process server (if any)\n"
                "4. How much did the server charge for service? (dollar amount, if stated)"
            ),
        })
        print(f"  ⏱ ask_process_server: {time.perf_counter() - t0:.2f}s")
        return {
            "process_server_name": res.process_server_name,
            "service_company_name": res.service_company_name,
            "process_server_registration": res.process_server_registration,
            "service_charge": res.service_charge,
        }

    async def ask_defendants(state: State) -> State:
        t0 = time.perf_counter()
        res = await defendants_chain.ainvoke({
            "document": md_content,
            "question": (
                "Extract information about every named defendant who was ACTUALLY SERVED "
                "in this document. Look for language indicating completed service such as "
                "'served by personally delivering', 'served by leaving copies with', "
                "'posted on the door', or similar.\n\n"
                "Do NOT extract defendants who are merely listed in the case caption, "
                "complaint header, or in a request/application but were not actually served.\n\n"
                "IMPORTANT: A court order stating defendants 'may be served by posting' "
                "is granting PERMISSION for future service — it does NOT mean service "
                "has been completed. Do not extract defendants from such orders.\n\n"
                "Scan the ENTIRE document from start to finish — there may be MULTIPLE "
                "proof of service sections, one per defendant. Each defendant may appear "
                "on a separate page. Count ALL of them.\n\n"
                "For each defendant actually served, provide:\n"
                "- defendant_name\n"
                "- service_method: 'personal', 'substitute', 'posting', or 'safe_at_home'\n"
                "- service_address: full address where served\n"
                "- service_date: date served (MM/DD/YYYY), null if not stated\n"
                "- service_time: time served (HH:MM am/pm), null if not stated\n"
                "- co_occupant_description: for substitute service, name and description of "
                "co-occupant who accepted service; null otherwise\n"
                "- mailing_postmark_date: for substitute or posting, postmark date of "
                "mailed copy (MM/DD/YYYY); null otherwise"
            ),
        })
        print(f"  ⏱ ask_defendants: {time.perf_counter() - t0:.2f}s")
        return {
            "num_named_defendants": res.num_named_defendants,
            "defendants": [d.model_dump() for d in res.defendants],
        }

    async def ask_unnamed_defendants(state: State) -> State:
        t0 = time.perf_counter()
        res = await unnamed_chain.ainvoke({
            "document": md_content,
            "question": (
                "Does this document list any unnamed defendants as a party that was "
                "actually served? Look for language like 'ALL OTHER UNNAMED OCCUPANTS', "
                "'DOES 1-10', or similar under 'Party Served' or in a service section.\n\n"
                "Only return true if unnamed defendants were actually served (not merely "
                "listed in the case caption or complaint).\n\n"
                "If yes, how were they served? ('substitute' or 'posting')\n"
                "If served by posting or substitute, what is the mailing postmark date?"
            ),
        })
        print(f"  ⏱ ask_unnamed_defendants: {time.perf_counter() - t0:.2f}s")
        return {
            "has_unnamed_defendants": res.has_unnamed_defendants,
            "unnamed_service_method": res.unnamed_service_method,
            "unnamed_mailing_postmark_date": res.unnamed_mailing_postmark_date,
        }

    async def ask_service_attempts(state: State) -> State:
        t0 = time.perf_counter()
        res = await attempts_chain.ainvoke({
            "document": md_content,
            "question": (
                "Does this document contain a 'Declaration of Diligence' or any record "
                "of PRIOR FAILED attempts to serve the defendant?\n\n"
                "These are attempts that did NOT result in successful service — e.g., "
                "'knocked on the door, no one answered', 'no one home', etc. They are "
                "typically listed in a Declaration of Diligence section.\n\n"
                "Do NOT count the actual successful service event (where documents were "
                "delivered or left with someone) as an attempt. That is the completed "
                "service, not a prior attempt.\n\n"
                "For each prior failed attempt: date (MM/DD/YYYY), time (HH:MM am/pm), "
                "and description.\n"
                "If no prior attempts are documented, return 0 and an empty list."
            ),
        })
        print(f"  ⏱ ask_service_attempts: {time.perf_counter() - t0:.2f}s")
        return {
            "num_attempts": res.num_attempts,
            "attempts": [a.model_dump() for a in res.attempts],
        }

    # --- Parallel graph (ORIGINAL edges + NEW edges) ---
    builder = StateGraph(state_schema=State)
    builder.add_node("start", start)
    builder.add_node("ask_pos_type", ask_pos_type)                        # ORIGINAL
    builder.add_node("ask_attorney_fees", ask_attorney_fees)              # ORIGINAL
    builder.add_node("ask_case_number", ask_case_number)                  # NEW
    builder.add_node("ask_application_info", ask_application_info)        # NEW
    builder.add_node("ask_process_server", ask_process_server)            # NEW
    builder.add_node("ask_defendants", ask_defendants)                    # NEW
    builder.add_node("ask_unnamed_defendants", ask_unnamed_defendants)    # NEW
    builder.add_node("ask_service_attempts", ask_service_attempts)        # NEW

    builder.set_entry_point("start")
    builder.add_edge("start", "ask_pos_type")             # ORIGINAL
    builder.add_edge("start", "ask_attorney_fees")        # ORIGINAL
    builder.add_edge("start", "ask_case_number")          # NEW
    builder.add_edge("start", "ask_application_info")     # NEW
    builder.add_edge("start", "ask_process_server")       # NEW
    builder.add_edge("start", "ask_defendants")           # NEW
    builder.add_edge("start", "ask_unnamed_defendants")   # NEW
    builder.add_edge("start", "ask_service_attempts")     # NEW

    return builder.compile()


# ── Cell 11: Supabase (ORIGINAL pattern, more columns) ───────────────────────

def upload_to_supabase(out):
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("  ⚠ Supabase credentials not set — skipping upload.")
        return

    supabase = create_client(url, key)

    response = supabase.table("proof_of_service").insert({
        # ORIGINAL
        "pos_type":         out.get("pos_type"),
        "attorney_fees":    out.get("attorney_fees"),
        # NEW
        "case_number":                  out.get("case_number"),
        "has_application_for_posting":  out.get("has_application_for_posting", False),
        "application_filed_date":       out.get("application_filed_date"),
        "application_granted_date":     out.get("application_granted_date"),
        "judge_name":                   out.get("judge_name"),
        "judge_department":             out.get("judge_department"),
        "has_proof_of_service":         out.get("has_proof_of_service", False),
        "process_server_name":          out.get("process_server_name"),
        "service_company_name":         out.get("service_company_name"),
        "process_server_registration":  out.get("process_server_registration"),
        "service_charge":               out.get("service_charge"),
        "num_named_defendants":         out.get("num_named_defendants", 0),
        "defendants":                   json.dumps(out.get("defendants", [])),
        "has_unnamed_defendants":       out.get("has_unnamed_defendants", False),
        "unnamed_service_method":       out.get("unnamed_service_method"),
        "unnamed_mailing_postmark_date": out.get("unnamed_mailing_postmark_date"),
        "num_attempts":                 out.get("num_attempts", 0),
        "attempts":                     json.dumps(out.get("attempts", [])),
    }).execute()

    print(f"  ☁️ Inserted: {response.data}")


# ── CLI wrapper (for batch runs) ─────────────────────────────────────────────

import asyncio
import argparse
import csv


async def process_one(pdf_path):
    print(f"📄 Converting PDF → Markdown: {pdf_path}")
    pages = pdf_to_markdown_pages(pdf_path)
    md_content = "\n\n".join(pages)

    md_path = Path(pdf_path).with_suffix(".md")
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  → Saved markdown to {md_path}")

    print("🔍 Extracting fields (8 parallel LLM calls)...")
    t0 = time.perf_counter()
    graph = build_graph(md_content)
    out = await graph.ainvoke({})
    print(f"  ⏱ TOTAL extraction: {time.perf_counter() - t0:.2f}s")
    return out


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    if args.pdf:
        pdfs = [args.pdf]
    elif args.pdf_dir:
        pdfs = sorted(args.pdf_dir.glob("*.pdf"))
        print(f"📂 Processing {len(pdfs)} PDFs from {args.pdf_dir}\n")
    else:
        parser.print_help()
        return

    results = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(pdfs)}] {pdf.name}")
        print(f"{'='*60}")
        try:
            out = await process_one(pdf)
            out["_file"] = pdf.name
            results.append(out)
            if args.upload:
                upload_to_supabase(out)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append({"_file": pdf.name, "_error": str(e)})

    if results:
        rows = []
        for r in results:
            row = {k: v for k, v in r.items() if k not in ("defendants", "attempts")}
            row["defendants"] = json.dumps(r.get("defendants", []))
            row["attempts"] = json.dumps(r.get("attempts", []))
            rows.append(row)

        fieldnames = list(rows[0].keys())
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✅ Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())