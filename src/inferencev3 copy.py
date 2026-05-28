"""
Obiettivo: definire e eseguire il workflow agentico che genera n dati sintetici
a partire da un documento PDF generico (fattura, contratto, referto medico, ecc.).

Il workflow è completamente adattivo: gli agenti deducono autonomamente la struttura
del documento, senza fare assunzioni sul tipo di dato.

- Input da linea di comando:
    --file = nome del file presente nella cartella /dataset
    --n_file = n dati sintetici da generare
    --run_id = stringa yyyymmdd-hhmm, identificativo della run
- Output:
    n dati sintetici generati e validati, salvati in output/<run_id>/
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, TypedDict

import pytesseract
from dotenv import load_dotenv
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
        model="google/gemma-4-31b-it",
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


def analyze_node(state: AgentState) -> dict:
    """
    Agente 1 - Analizzatore:
    Esamina il testo OCR e produce DUE output:
    1. Uno schema che descrive la struttura del documento (campi, tipi, relazioni)
    2. I dati estratti dal documento sorgente

    Usa with_structured_output() per output Pydantic garantito.
    """
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
    """
    Agente 2 - Generatore:
    Riceve lo schema e i dati sorgente, produce N record sintetici.
    I dati sensibili vengono sostituiti con valori realistici ma fittizi.
    Le relazioni matematiche vengono preservate.
    """
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
        system_prompt=f"""Sei un generatore di dati sintetici. Devi generare ESATTAMENTE {n} record basati su schema e dati forniti.

REGOLE:
1. Genera ESATTAMENTE {n} record in un array JSON
2. Ogni record DEVE rispettare lo schema fornito (stessi campi, stessi tipi)
3. I campi marcati "sensitive": true DEVONO essere sostituiti con valori fittizi ma realistici:
   - Nomi di persone: usa nomi credibili ma inventati
   - Indirizzi: genera indirizzi verosimili (città reali, vie inventate)
   - Codici identificativi: genera nuovi codici con lo stesso formato
   - IBAN / conti bancari: genera codici con formato corretto ma inventati
4. I campi NON sensibili possono variare entro range ragionevoli
5. RISPETTA TUTTI i constraints dello schema (es. totali = somma dei dettagli)
6. Se lo schema contiene liste di oggetti, varia il numero di elementi (±30%)
7. Mantieni la coerenza interna: se un campo deriva da altri, ricalcolalo correttamente

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
    """
    Agente 3 - Validatore:
    Verifica ogni record generato contro lo schema e i constraints.
    Corregge errori di calcolo, rimuove record invalidi.

    Usa with_structured_output() per output Pydantic garantito.
    """
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


def save_node(state: AgentState) -> dict:
    """
    Salva i dati finali (record validati) e i metadati (schema, dati sorgente).
    """
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
    graph.add_node("save", save_node)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "save")
    graph.add_edge("save", END)
    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description="Workflow agentico di generazione dati sintetici")
    parser.add_argument("--file", type=str, required=True, help="Nome file nella cartella dataset")
    parser.add_argument("--n_file", type=int, required=True, help="Numero di record sintetici da generare")
    parser.add_argument("--run_id", type=str, required=True, help="ID della run (es. 20260519-1650)")

    args = parser.parse_args()
    path_file = os.path.join("dataset", args.file)

    if not os.path.exists(path_file):
        logger.error(f"File non trovato: {path_file}")
        return

    logger.info(f"Avvio workflow: {args.file} → {args.n_file} record sintetici (run: {args.run_id})")

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
    })

    n_generated = len(result.get("final_data", []))
    logger.info(f"Workflow completato: {n_generated} record sintetici generati e salvati")


if __name__ == "__main__":
    main()