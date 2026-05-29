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
from typing import List, TypedDict

import pytesseract
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pdf2image import convert_from_path

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


def analyze_node(state: AgentState) -> dict:
    """
    Agente 1 - Analizzatore:
    Esamina il testo OCR e produce DUE output:
    1. Un JSON Schema che descrive la struttura del documento (campi, tipi, relazioni)
    2. I dati estratti dal documento sorgente, nel formato definito dallo schema
    """
    logger.info("[Agente 1] Analisi del documento - deduzione struttura e estrazione dati")

    llm = build_llm(temperature=0.2)
    analyzer = create_agent(
        model=llm,
        tools=[],
        system_prompt="""Sei un analizzatore di documenti. Ricevi il testo OCR di un PDF generico.

Il tuo compito è produrre un JSON con DUE sezioni:

1. "schema": descrivi la struttura del documento. Per ogni campo indica:
   - "field": nome del campo
   - "type": uno tra "string", "number", "date", "list", "object"
   - "sensitive": true se il campo contiene dati personali/sensibili (nomi, indirizzi, codici fiscali, IBAN, numeri documento, ecc.)
   - "constraints": eventuali vincoli (es. "deve essere uguale alla somma di X", "formato MM/DD/YYYY", "deve corrispondere a Y * Z")

2. "data": i dati estratti dal documento, strutturati secondo lo schema.

REGOLE FONDAMENTALI:
- NON fare assunzioni sul tipo di documento. Deduci la struttura dai dati presenti.
- Identifica automaticamente le relazioni tra campi (es. totali = somma di dettagli).
- Marca come "sensitive: true" TUTTI i campi che contengono dati personali.
- Per i campi numerici, verifica le relazioni aritmetiche e documentale nei constraints.
- Output SOLO JSON valido, senza markdown, senza testo aggiuntivo.

Esempio di output:
{
  "schema": [
    {"field": "doc_id", "type": "string", "sensitive": true, "constraints": "identificativo univoco"},
    {"field": "date", "type": "date", "sensitive": false, "constraints": "formato MM/DD/YYYY"},
    {"field": "total", "type": "number", "sensitive": false, "constraints": "deve essere uguale alla somma di items[].amount"}
  ],
  "data": { ... }
}
""",
    )

    result = analyzer.invoke({"messages": [HumanMessage(content=state["raw_text"])]})
    raw_output = result["messages"][-1].content

    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if json_match:
        raw_output = json_match.group(0)

    try:
        parsed = json.loads(raw_output)
        schema = json.dumps(parsed.get("schema", []), indent=2)
        source = json.dumps(parsed.get("data", {}), indent=2)
    except json.JSONDecodeError:
        logger.warning("Analizzatore non ha prodotto JSON valido, uso output raw")
        schema = raw_output
        source = raw_output

    logger.info("Schema e dati sorgente estratti")
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
    """
    logger.info("[Agente 3] Validazione e correzione dei record generati")

    llm = build_llm(temperature=0.1)
    validator = create_agent(
        model=llm,
        tools=[],
        system_prompt="""Sei un validatore di dati sintetici. Ricevi uno schema, dei constraints e dei record generati.

Per OGNI record:
1. Verifica che TUTTI i campi dello schema siano presenti
2. Verifica che i tipi siano corretti (string, number, date, list, object)
3. Per ogni constraint nello schema, verifica che sia rispettato
4. Se trovi errori di calcolo (es. totale != somma), CORREGGI ricalcolando
5. Se un campo obbligatorio manca, NON inventarlo - rimuovi il record
6. Se un record è completamente invalido, rimuovilo

Output: array JSON con SOLO i record validi (corretti dove necessario).
NON aggiungere markdown, SOLO JSON.
""",
    )

    prompt = f"""SCHEMA e constraints:
{state['document_schema']}

RECORD DA VALIDARE:
{state['generated_data']}

Valida, correggi, e restituisci l'array JSON dei record validi."""
    result = validator.invoke({"messages": [HumanMessage(content=prompt)]})
    validated = result["messages"][-1].content
    logger.info("Validazione completata")
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

    def extract_json_array(raw: str) -> List[dict]:
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []

    records = extract_json_array(state["validated_data"])
    if not records:
        logger.warning("Validatore non ha prodotto JSON valido, uso dati generati")
        records = extract_json_array(state["generated_data"])
    if not records:
        logger.error("Nessun dato parsabile, salvo output raw")
        records = [{"raw_output": state["validated_data"]}]

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
