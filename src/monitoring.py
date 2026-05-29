"""
Obiettivo: valutare la qualità dei dati sintetici generati da inference.py.

Il monitoring è generico e adattivo: legge lo schema prodotto dall'agente analizzatore
e valuta i record rispetto a quello schema, senza assumere un tipo specifico di documento.

- Input:
    --run_id = identificativo della run (coincide con quello di inference.py)

- Metriche:
    1. Completezza strutturale (vs schema)
    2. Coerenza interna (constraints dello schema)
    3. Diversità dei record
    4. Validità formale dei campi
    5. Qualità semantica (LLM)
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from rich.console import Console
from rich.table import Table

load_dotenv()
OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")
OUTPUT_DIR = "output"

console = Console()


def load_run_data(run_id: str) -> tuple:
    data_path = os.path.join(OUTPUT_DIR, run_id, "synthetic_data.json")
    meta_path = os.path.join(OUTPUT_DIR, run_id, "metadata.json")

    records = []
    schema = []

    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            records = json.load(f)

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            raw_schema = meta.get("schema", "[]")
            if isinstance(raw_schema, str):
                try:
                    schema = json.loads(raw_schema)
                except json.JSONDecodeError:
                    schema = []
            elif isinstance(raw_schema, list):
                schema = raw_schema

    return records, schema


def check_structural_completeness(records: List[dict], schema: List[dict]) -> Dict[str, Any]:
    logger.info("[Metrica 1] Completezza strutturale vs schema")
    total = len(records)
    if total == 0 or not schema:
        return {"score": 0.0, "details": "Nessun record o schema disponibile"}

    field_names = [s["field"] for s in schema]
    field_presence = {f: 0 for f in field_names}

    for rec in records:
        for field in field_names:
            val = rec.get(field)
            if val is not None and val != "" and val != []:
                field_presence[field] += 1

    field_scores = {f: count / total for f, count in field_presence.items()}
    overall = sum(field_scores.values()) / len(field_scores) if field_scores else 0

    return {
        "score": round(overall, 4),
        "field_completeness": field_scores,
        "total_records": total,
        "schema_fields": len(field_names),
    }


def check_internal_coherence(records: List[dict], schema: List[dict]) -> Dict[str, Any]:
    logger.info("[Metrica 2] Coerenza interna (constraints)")
    total = len(records)
    coherent = 0
    errors = []

    if total == 0:
        return {"score": 0.0, "details": "Nessun record"}

    for i, rec in enumerate(records):
        rec_ok = True

        for field_def in schema:
            constraints = field_def.get("constraints", "")
            field_name = field_def["field"]
            field_type = field_def.get("type", "string")

            if field_type == "number" and "somma" in str(constraints).lower():
                list_ref_match = re.search(r"(\w+)\[\]\.(\w+)", str(constraints))
                if list_ref_match:
                    list_name = list_ref_match.group(1)
                    item_field = list_ref_match.group(2)
                    items = rec.get(list_name, [])
                    if isinstance(items, list):
                        expected = sum(
                            float(it.get(item_field, 0))
                            for it in items
                            if isinstance(it, dict)
                        )
                        actual = float(rec.get(field_name, 0))
                        if abs(actual - expected) > 0.05:
                            errors.append(
                                f"Record {i}: {field_name}={actual}, somma {list_name}[].{item_field}={expected:.2f}"
                            )
                            rec_ok = False

            if "corrispond" in str(constraints).lower() or "deriva" in str(constraints).lower():
                mul_match = re.search(r"(\w+)\s*\*\s*(\w+)", str(constraints))
                if mul_match:
                    a = float(rec.get(mul_match.group(1), 0))
                    b = float(rec.get(mul_match.group(2), 0))
                    actual = float(rec.get(field_name, 0))
                    expected = a * b
                    if abs(actual - expected) > 0.03:
                        errors.append(
                            f"Record {i}: {field_name}={actual}, atteso {mul_match.group(1)}*{mul_match.group(2)}={expected:.2f}"
                        )
                        rec_ok = False

        if rec_ok:
            coherent += 1

    score = coherent / total if total > 0 else 0
    return {
        "score": round(score, 4),
        "coherent_records": coherent,
        "total_records": total,
        "errors": errors[:20],
    }


def check_diversity(records: List[dict], schema: List[dict]) -> Dict[str, Any]:
    logger.info("[Metrica 3] Diversità dei record")
    total = len(records)
    if total <= 1:
        return {"score": 0.0, "details": "Troppi pochi record"}

    string_fields = [s["field"] for s in schema if s.get("type") == "string"]
    if not string_fields:
        string_fields = list(records[0].keys()) if records else []

    uniqueness_ratios = {}
    for field in string_fields:
        values = [str(rec.get(field, "")) for rec in records]
        unique = len(set(values))
        uniqueness_ratios[field] = unique / total

    overall = sum(uniqueness_ratios.values()) / len(uniqueness_ratios) if uniqueness_ratios else 0

    return {
        "score": round(overall, 4),
        "uniqueness_by_field": {k: round(v, 2) for k, v in uniqueness_ratios.items()},
        "total_records": total,
    }


def check_field_validity(records: List[dict], schema: List[dict]) -> Dict[str, Any]:
    logger.info("[Metrica 4] Validità formale dei campi")

    patterns = {
        "date": re.compile(r"^\d{2}/\d{2}/\d{4}$"),
        "tax_id": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
        "iban": re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$"),
        "email": re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
        "phone": re.compile(r"^[\d\s\-\+\(\)]{7,20}$"),
    }

    total = len(records)
    if total == 0:
        return {"score": 0.0, "details": "Nessun record"}

    validity_scores = {}

    for field_def in schema:
        field_name = field_def["field"]
        constraints = str(field_def.get("constraints", "")).lower()
        field_type = field_def.get("type", "")

        pattern = None
        if field_type == "date" or "data" in field_name.lower() or "date" in field_name.lower():
            pattern = patterns["date"]
        elif "iban" in field_name.lower():
            pattern = patterns["iban"]
        elif "tax" in field_name.lower() or "fiscale" in field_name.lower():
            pattern = patterns["tax_id"]
        elif "email" in field_name.lower() or "mail" in field_name.lower():
            pattern = patterns["email"]
        elif "phone" in field_name.lower() or "telefono" in field_name.lower():
            pattern = patterns["phone"]
        elif "formato" in constraints:
            if "dd/mm" in constraints or "mm/dd" in constraints:
                pattern = patterns["date"]

        if pattern:
            valid = 0
            present = 0
            for rec in records:
                val = str(rec.get(field_name, ""))
                if val:
                    present += 1
                    if pattern.match(val):
                        valid += 1
            validity_scores[field_name] = valid / present if present > 0 else 1.0

    overall = sum(validity_scores.values()) / len(validity_scores) if validity_scores else 1.0

    return {
        "score": round(overall, 4),
        "field_validity": validity_scores,
    }


def semantic_quality_check(records: List[dict], schema: List[dict]) -> Dict[str, Any]:
    logger.info("[Metrica 5] Valutazione semantica (LLM)")

    if not records:
        return {"score": 0.0, "feedback": "Nessun record da valutare"}

    sample = records[: min(5, len(records))]
    payload = json.dumps({"schema": schema, "sample_records": sample}, indent=2, ensure_ascii=False)

    llm = ChatOpenAI(
        model="google/gemma-4-31b-it",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPEN_ROUTER_KEY,
        temperature=0.2,
    )

    evaluator = create_agent(
        model=llm,
        tools=[],
        system_prompt="""Sei un valutatore di qualità per dati sintetici generati da documenti.

Valuta su scala 1-10:
1. Realismo: i dati generati sono credibili e verosimili?
2. Varietà: i record sono sufficientemente diversi tra loro?
3. Coerenza semantica: i valori hanno senso nel contesto (es. un prodotto ha un prezzo ragionevole)?
4. Usabilità: i dati sono pronti per essere usati in applicazioni downstream?

Output JSON:
{
  "realism_score": 0,
  "variety_score": 0,
  "coherence_score": 0,
  "usability_score": 0,
  "overall_score": 0,
  "feedback": "breve commento in italiano"
}

Output SOLO JSON, senza markdown.
""",
    )

    result = evaluator.invoke({"messages": [HumanMessage(content=payload)]})
    raw = result["messages"][-1].content

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    try:
        scores = json.loads(raw)
        scores["overall_score"] = round(scores.get("overall_score", 0) / 10, 4)
    except json.JSONDecodeError:
        scores = {"overall_score": 0.5, "feedback": raw[:200]}

    return scores


def compute_final_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    weights = {
        "structural_completeness": 0.25,
        "internal_coherence": 0.30,
        "diversity": 0.15,
        "field_validity": 0.15,
        "semantic_quality": 0.15,
    }

    weighted_sum = 0.0
    for key, weight in weights.items():
        val = metrics.get(key, {}).get("score", 0)
        weighted_sum += val * weight

    return {
        "overall_score": round(weighted_sum, 4),
        "weights": weights,
        "grade": (
            "A" if weighted_sum >= 0.9
            else "B" if weighted_sum >= 0.75
            else "C" if weighted_sum >= 0.6
            else "D" if weighted_sum >= 0.4
            else "F"
        ),
    }


def print_report(metrics: Dict[str, Any], final: Dict[str, Any], run_id: str):
    console.print(f"\n[bold cyan]══════════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]  REPORT MONITORING - Run: {run_id}[/bold cyan]")
    console.print(f"[bold cyan]══════════════════════════════════════════[/bold cyan]\n")

    table = Table(title="Metriche di Qualità")
    table.add_column("Metrica", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Dettaglio", style="white")

    sc = metrics.get("structural_completeness", {})
    table.add_row(
        "Completezza strutturale",
        f"{sc.get('score', 0):.2%}",
        f"{sc.get('total_records', 0)} record, {sc.get('schema_fields', 0)} campi",
    )

    ic = metrics.get("internal_coherence", {})
    table.add_row(
        "Coerenza interna",
        f"{ic.get('score', 0):.2%}",
        f"{ic.get('coherent_records', 0)}/{ic.get('total_records', 0)} coerenti",
    )

    div = metrics.get("diversity", {})
    table.add_row(
        "Diversità",
        f"{div.get('score', 0):.2%}",
        f"{div.get('total_records', 0)} record",
    )

    fv = metrics.get("field_validity", {})
    table.add_row(
        "Validità formale",
        f"{fv.get('score', 0):.2%}",
        f"{len(fv.get('field_validity', {}))} campi validati",
    )

    sq = metrics.get("semantic_quality", {})
    table.add_row(
        "Qualità semantica",
        f"{sq.get('overall_score', 0):.2%}",
        str(sq.get("feedback", ""))[:60],
    )

    console.print(table)
    console.print(f"\n[bold yellow]SCORE FINALE: {final['overall_score']:.2%} - GRADO: {final['grade']}[/bold yellow]")

    ic_errors = metrics.get("internal_coherence", {}).get("errors", [])
    if ic_errors:
        console.print("\n[bold red]Errori di coerenza:[/bold red]")
        for err in ic_errors[:10]:
            console.print(f"  • {err}")

    console.print(f"\n[dim]Report salvato in: {OUTPUT_DIR}/{run_id}/monitoring_report.json[/dim]\n")


def main():
    parser = argparse.ArgumentParser(description="Script di monitoring - valutazione qualità dati sintetici")
    parser.add_argument("--run_id", type=str, required=True, help="ID della run (es. 20260519-1650)")
    args = parser.parse_args()

    logger.info(f"Avvio monitoring per run: {args.run_id}")
    records, schema = load_run_data(args.run_id)

    if not records:
        logger.error("Nessun record trovato. Eseguire prima inference.py")
        return

    logger.info(f"Caricati {len(records)} record e {len(schema)} campi schema")

    metrics = {
        "structural_completeness": check_structural_completeness(records, schema),
        "internal_coherence": check_internal_coherence(records, schema),
        "diversity": check_diversity(records, schema),
        "field_validity": check_field_validity(records, schema),
        "semantic_quality": semantic_quality_check(records, schema),
    }

    final = compute_final_score(metrics)
    metrics["final_score"] = final

    run_dir = os.path.join(OUTPUT_DIR, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    report_path = os.path.join(run_dir, "monitoring_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    print_report(metrics, final, args.run_id)


if __name__ == "__main__":
    main()