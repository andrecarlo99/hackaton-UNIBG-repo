"""
Obiettivo: definire e eseguire il workflow agentico che genera n dati sintetici
a partire da un documento PDF generico (fattura, contratto, referto medico, ecc.)
e produce un PDF sintetico con la stessa struttura dell'originale.

- Input da linea di comando:
    --file = nome del file presente nella cartella /dataset
    --n_file = n dati sintetici da generare
    --run_id = stringa yyyymmdd-hhmm, identificativo della run
- Output:
    n dati sintetici generati e validati, salvati in output/<run_id>/
    + PDF sintetico con la stessa struttura dell'originale
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, TypedDict
import pytesseract
from dotenv import load_dotenv
from fpdf import FPDF
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pdf2image import convert_from_path
from pydantic import BaseModel, Field

load_dotenv()
OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")
OUTPUT_DIR = "output"


class AgentState(TypedDict):
    messages: List[BaseMessage]
    raw_text: str
    document_schema: str
    source_data: str
    generated_data: str
    validated_data: str
    final_data: List[dict]
    errors: List[str]
    pdf_path: str


def extract_text_from_pdf(pdf_path: str) -> str:
    logger.info(f"Estrazione OCR da: {pdf_path}")
    pages = convert_from_path(pdf_path, 300)
    full_text = ""
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang="eng")
        full_text += f"--- Page {i+1} ---\n{text}\n"
    return full_text


def build_llm(temperature: float = 0.3):
    return ChatOpenAI(
        model="anthropic/claude-opus-4.7",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPEN_ROUTER_KEY,
        temperature=temperature,
    )


class FieldDefinition(BaseModel):
    field: str = Field(description="Nome del campo")
    type: str = Field(description="Tipo: string, number, date, list, object")
    sensitive: bool = Field(description="True se contiene dati personali/sensibili")
    constraints: str = Field(description="Vincoli del campo (es. 'somma di X', 'formato MM/DD/YYYY')")


class DocumentExtraction(BaseModel):
    schema_def: List[FieldDefinition] = Field(description="Schema del documento: array di definizioni dei campi")
    data: Dict[str, Any] = Field(description="Dati estratti dal documento sorgente")


class ValidatedRecords(BaseModel):
    records: List[Dict[str, Any]] = Field(description="Record validati e corretti")


class LayoutInfo(BaseModel):
    title: str = Field(description="Titolo del documento (es. 'INVOICE', 'CONTRACT')")
    sections: List[str] = Field(description="Sezioni del documento in ordine (es. header, items, summary)")
    table_columns: List[str] = Field(description="Colonne della tabella principale, se presente")
    field_order: List[str] = Field(description="Ordine dei campi come appaiono nel documento")


def analyze_node(state: AgentState) -> dict:
    logger.info("[Agente 1] Analisi del documento - deduzione struttura e estrazione dati")

    llm = build_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(DocumentExtraction)

    prompt = f"""Analizza il seguente testo OCR estratto da un PDF.

TESTO OCR:
{state["raw_text"]}

Il tuo compito:
1. Deduci la struttura del documento (NON fare assunzioni sul tipo).
2. Per ogni campo, indica nome, tipo, se è sensibile, e eventuali vincoli.
3. Estrai i dati dal documento nel formato dedotto.

Marca come sensitive: true TUTTI i campi con dati personali (nomi, indirizzi, codici fiscali, IBAN, numeri documento, ecc.).
Identifica relazioni tra campi (es. totali = somma di dettagli) e scrivile nei constraints."""

    extraction: DocumentExtraction = structured_llm.invoke(prompt)

    schema_list = [fd.model_dump() for fd in extraction.schema_def]
    schema = json.dumps(schema_list, indent=2)
    source = json.dumps(extraction.data, indent=2)

    logger.info(f"Schema dedotto: {len(extraction.schema_def)} campi")
    return {"document_schema": schema, "source_data": source}


def generate_node(state: AgentState) -> dict:
    n_files = state["messages"][0].content
    try:
        n = int(n_files)
    except ValueError:
        n = 5

    logger.info(f"[Agente 2] Generazione di {n} record sintetici")

    llm = build_llm(temperature=0.7)
    generator = create_agent(
        model=llm,
        tools=[],
        system_prompt=f"""Sei un anonimizzatore di dati. Devi produrre ESATTAMENTE {n} record basati sui dati forniti.

REGOLE FONDAMENTALI:
1. Produci ESATTAMENTE {n} record in un array JSON
2. I campi NON sensibili DEVONO rimanere IDENTICI all'originale: NON modificarli MAI
   - Quantità, prezzi, totali, percentuali IVA, date, numeri fattura: COPIATI ESATTAMENTE
   - Il numero di elementi nelle liste (es. items) DEVE rimanere lo stesso
   - I valori numerici NON vanno alterati in alcun modo
3. I campi marcati "sensitive": true DEVONO essere sostituiti con valori fittizi ma realistici:
   - Nomi di persone/aziende: nomi credibili ma inventati
   - Indirizzi: indirizzi verosimili ma fittizi
   - Codici fiscali / Tax ID: stesso formato, valori inventati
   - IBAN: formato corretto, valori inventati
4. NON alterare la struttura del documento: stesso numero di items, stessi campi

Output: array JSON di {n} record. SOLO JSON, senza markdown.
""",
    )

    prompt = f"""SCHEMA del documento:
{state['document_schema']}

DATI SORGENTE (da usare come riferimento per formato e struttura):
{state['source_data']}

Genera {n} record sintetici."""
    result = generator.invoke({"messages": [HumanMessage(content=prompt)]})
    generated = result["messages"][-1].content
    logger.info(f"Generati {n} record sintetici")
    return {"generated_data": generated}


def validate_node(state: AgentState) -> dict:
    logger.info("[Agente 3] Validazione e correzione dei record generati")

    llm = build_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(ValidatedRecords)

    prompt = f"""Valida i seguenti record sintetici contro lo schema fornito.

SCHEMA e constraints:
{state['document_schema']}

RECORD DA VALIDARE:
{state['generated_data']}

Per OGNI record:
1. Verifica che TUTTI i campi dello schema siano presenti.
2. Verifica che i tipi siano corretti.
3. Per ogni constraint, verifica che sia rispettato.
4. Se trovi errori di calcolo, CORREGGI ricalcolando.
5. Se un record è completamente invalido, rimuovilo."""

    validation: ValidatedRecords = structured_llm.invoke(prompt)
    validated = json.dumps(validation.records, indent=2)

    logger.info(f"Validazione completata: {len(validation.records)} record")
    return {"validated_data": validated}


def analyze_layout_node(state: AgentState) -> dict:
    """
    Agente 4 - Analizzatore di Layout:
    Deduce la struttura visuale del documento per replicarla nel PDF sintetico.
    """
    logger.info("[Agente 4] Analisi del layout visuale del documento")

    llm = build_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(LayoutInfo)

    prompt = f"""Analizza il seguente testo OCR e deduci il LAYOUT visuale del documento originale.

TESTO OCR:
{state["raw_text"]}

Determina:
1. Il titolo del documento (es. "INVOICE", "CONTRACT", "MEDICAL REPORT")
2. Le sezioni in ordine di apparizione (es. ["header", "items_table", "summary"])
3. Se c'è una tabella, quali sono le sue colonne
4. L'ordine in cui appaiono i campi"""

    layout: LayoutInfo = structured_llm.invoke(prompt)
    layout_json = json.dumps(layout.model_dump(), indent=2)

    logger.info(f"Layout dedotto: {layout.title}, {len(layout.sections)} sezioni")
    return {}


def generate_pdf_node(state: AgentState) -> dict:
    logger.info("[Agente 5] Generazione PDF sintetico Dinamico")

    run_id = ""
    for msg in state["messages"]:
        if hasattr(msg, "content") and "run_id:" in str(msg.content):
            run_id = str(msg.content).replace("run_id:", "").strip()
            break

    records = json.loads(state["validated_data"])
    if not records:
        logger.error("Nessun record da inserire nel PDF")
        return {"pdf_path": ""}

    record = records[0]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Titolo ---
    doc_title = str(record.get("document_type", record.get("doc_type", "SYNTHETIC DOCUMENT"))).upper()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 12, doc_title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "DOCUMENT DATA", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    lists_to_render = []

    # Iterazione strutturata direttamente sulle chiavi estratte dal record validato
    for name, val in record.items():
        if not val or str(val).strip() == "None" or val == [] or val == {}:
            continue
        if name in ["document_type", "doc_type"]:
            continue

        if isinstance(val, list):
            lists_to_render.append((name, val))
            continue

        if isinstance(val, dict):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, f"{str(name).replace('_', ' ').upper()}:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            for k, v in val.items():
                if v and str(v) != "None":
                    pdf.cell(0, 6, f"   - {str(k).replace('_', ' ').title()}: {str(v)[:80]}", new_x="LMARGIN", new_y="NEXT")
        else:
            formatted_name = str(name).replace('_', ' ').title()
            pdf.cell(0, 6, f"{formatted_name}: {str(val)[:80]}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # --- Generazione Tabelle di liste ---
    for list_name, list_data in lists_to_render:
        if not list_data or not isinstance(list_data[0], dict):
            continue 
            
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, str(list_name).replace('_', ' ').upper(), new_x="LMARGIN", new_y="NEXT")
        
        headers = list(list_data[0].keys())
        num_cols = len(headers)
        col_width = 190 / max(num_cols, 1) 
        
        pdf.set_fill_color(30, 60, 120)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        for h in headers:
            pdf.cell(col_width, 8, str(h).replace('_', ' ').title()[:15], border=1, fill=True, align="C", new_x="RIGHT", new_y="LAST")
        pdf.ln()
        
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", "", 9)
        for item in list_data:
            for h in headers:
                val = str(item.get(h, ""))[:20]
                pdf.cell(col_width, 7, val, border=1, align="C", new_x="RIGHT", new_y="LAST")
            pdf.ln()
        pdf.ln(6)

    # --- Footer ---
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "This is a synthetic document generated for testing purposes.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"Run ID: {run_id}", align="C", new_x="LMARGIN", new_y="NEXT")

    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    pdf_path = os.path.join(run_dir, "synthetic_document.pdf")
    pdf.output(pdf_path)
    logger.info(f"PDF sintetico salvato nel workflow di produzione: {pdf_path}")

    return {"pdf_path": pdf_path}
    logger.info("[Agente 5] Generazione PDF sintetico Dinamico")

    run_id = ""
    for msg in state["messages"]:
        if hasattr(msg, "content") and "run_id:" in str(msg.content):
            run_id = str(msg.content).replace("run_id:", "").strip()
            break

    records = json.loads(state["validated_data"])
    schema = json.loads(state["document_schema"])
    source = json.loads(state["source_data"])

    if not records:
        logger.error("Nessun record da inserire nel PDF")
        return {"pdf_path": ""}

    record = records[0]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Titolo ---
    doc_title = str(record.get("document_type", record.get("doc_type", "SYNTHETIC DOCUMENT"))).upper()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 12, doc_title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.set_text_color(40, 40, 40)

    # 2. ITERAZIONE DINAMICA SUI CAMPI SEMPLICI
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "DOCUMENT DATA", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    
    lists_to_render = []

    for f in schema:
        name = f["field"]
        val = record.get(name)
        
        # Saltiamo dati vuoti
        if not val or val == "None":
            continue

        # Se è una lista, la salviamo per renderizzarla come tabella dopo
        if f["type"] == "list" or isinstance(val, list):
            lists_to_render.append((name, val))
            continue

        # Stampiamo i dati semplici (string, number, date)
        if isinstance(val, dict):
            # Gestione sottomenù/oggetti
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, f"{name.replace('_', ' ').upper()}:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            for k, v in val.items():
                pdf.cell(0, 6, f"   - {k.replace('_', ' ').title()}: {str(v)[:80]}", new_x="LMARGIN", new_y="NEXT")
        else:
            # Campi testuali standard
            formatted_name = name.replace('_', ' ').title()
            pdf.cell(0, 6, f"{formatted_name}: {str(val)[:80]}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # 3. RENDERIZZAZIONE DINAMICA DELLE TABELLE (Liste)
    for list_name, list_data in lists_to_render:
        if not list_data or not isinstance(list_data[0], dict):
            continue # Salta liste non di oggetti
            
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, list_name.replace('_', ' ').upper(), new_x="LMARGIN", new_y="NEXT")
        
        # Estrai dinamicamente le chiavi dal primo elemento per fare gli header
        headers = list(list_data[0].keys())
        num_cols = len(headers)
        col_width = 190 / max(num_cols, 1) # Distribuzione equa della larghezza
        
        # Disegna Header
        pdf.set_fill_color(30, 60, 120)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        for h in headers:
            pdf.cell(col_width, 8, str(h).title()[:15], border=1, fill=True, align="C", new_x="RIGHT", new_y="LAST")
        pdf.ln()
        
        # Disegna Righe
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", "", 9)
        for item in list_data:
            for h in headers:
                val = str(item.get(h, ""))[:20]
                pdf.cell(col_width, 7, val, border=1, align="C", new_x="RIGHT", new_y="LAST")
            pdf.ln()
        pdf.ln(6)

        # --- Footer ---
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "This is a synthetic document generated for testing purposes.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"Run ID: {run_id}", align="C", new_x="LMARGIN", new_y="NEXT")

    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Salviamo con un nome generico
    pdf_path = os.path.join(run_dir, "synthetic_document.pdf")
    pdf.output(pdf_path)
    logger.info(f"PDF sintetico salvato: {pdf_path}")

    return {"pdf_path": pdf_path}

def save_node(state: AgentState) -> dict:
    logger.info("[Salvataggio] Scrittura dei risultati")

    run_id = ""
    for msg in state["messages"]:
        if hasattr(msg, "content") and "run_id:" in str(msg.content):
            run_id = str(msg.content).replace("run_id:", "").strip()
            break

    records = json.loads(state["validated_data"])

    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "synthetic_data.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    metadata = {
        "run_id": run_id,
        "schema": state["document_schema"],
        "source_data": state["source_data"],
        "records_generated": len(records),
        "pdf_path": state.get("pdf_path", ""),
    }
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Salvati {len(records)} record in {run_dir}/")
    return {"final_data": records}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("analyze", analyze_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("generate_pdf", generate_pdf_node)
    graph.add_node("save", save_node)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "generate_pdf")
    graph.add_edge("generate_pdf", "save")
    graph.add_edge("save", END)
    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description="Workflow agentico di generazione dati sintetici + PDF")
    parser.add_argument("--file", type=str, required=True, help="Nome file nella cartella dataset")
    parser.add_argument("--n_file", type=int, required=True, help="Numero di record sintetici da generare")
    parser.add_argument("--run_id", type=str, required=True, help="ID della run (es. 20260519-1650)")

    args = parser.parse_args()
    path_file = os.path.join("dataset", args.file)

    if not os.path.exists(path_file):
        logger.error(f"File non trovato: {path_file}")
        return

    logger.info(f"Avvio workflow: {args.file} -> {args.n_file} record sintetici + PDF (run: {args.run_id})")

    raw_text = extract_text_from_pdf(path_file)
    app = build_graph()

    result = app.invoke({
        "messages": [
            HumanMessage(content=str(args.n_file)),
            HumanMessage(content=f"run_id:{args.run_id}"),
        ],
        "raw_text": raw_text,
        "document_schema": "",
        "source_data": "",
        "generated_data": "",
        "validated_data": "",
        "final_data": [],
        "errors": [],
        "pdf_path": "",
    })

    n_generated = len(result.get("final_data", []))
    pdf_path = result.get("pdf_path", "")
    logger.info(f"Workflow completato: {n_generated} record + PDF: {pdf_path}")


if __name__ == "__main__":
    main()
