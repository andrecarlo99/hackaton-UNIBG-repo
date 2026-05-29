"""
Obiettivo: Monitorare e valutare la qualità dei dati sintetici generati dal workflow.
Analizza l'output logico (JSON) e fisico (PDF) di una specifica run e calcola i KPI.
"""

import argparse
import json
import os
from loguru import logger

# Importiamo la funzione OCR dal file di inference.
# Supportiamo inferencev4, inferencev3 e fallback locale.
try:
    from inferencev4 import extract_text_from_pdf
except ImportError:
    try:
        from inferencev3 import extract_text_from_pdf
    except ImportError:
        logger.warning("Impossibile importare 'extract_text_from_pdf'. Uso fallback locale.")
        import pytesseract
        from pdf2image import convert_from_path

        def extract_text_from_pdf(pdf_path: str) -> str:
            pages = convert_from_path(pdf_path, 300)
            full_text = ""
            for i, page in enumerate(pages):
                text = pytesseract.image_to_string(page, lang="eng")
                full_text += f"--- Page {i+1} ---\n{text}\n"
            return full_text

#directory di output 
OUTPUT_DIR = "output/notebook_v4"

def load_run_data():
    """Carica i dati generati e i metadati dalla cartella di output."""
    if not os.path.exists(OUTPUT_DIR):
        logger.error(f"Cartella output non trovata: {OUTPUT_DIR}")
        return None, None
    
    try:
        with open(os.path.join(OUTPUT_DIR, "synthetic_data.json"), "r", encoding="utf-8") as f:
            synthetic_records = json.load(f)
            
        with open(os.path.join(OUTPUT_DIR, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
            metadata["schema"] = json.loads(metadata["schema"])
            metadata["source_data"] = json.loads(metadata["source_data"])
            
        return synthetic_records, metadata
    except Exception as e:
        logger.error(f"Errore nella lettura dei file: {e}")
        return None, None

def check_pdf_physical_quality(pdf_path: str, expected_record: dict, ocr_text: str) -> dict:
    """
    Valuta la salute fisica del PDF e verifica che i dati attesi 
    siano effettivamente leggibili sul documento renderizzato.
    """
    if not os.path.exists(pdf_path):
        return {"status": "Fallito", "error": "File non trovato", "visual_match_rate": 0.0}
    
    file_size_kb = os.path.getsize(pdf_path) / 1024
    if file_size_kb < 2:
        return {"status": "Fallito", "error": "File corrotto o vuoto", "visual_match_rate": 0.0}

    testo_estratto_lower = ocr_text.lower()
    valori_attesi = 0
    valori_trovati = 0
    errori_visivi = []

    for key, value in expected_record.items():
        if value is None or isinstance(value, bool):
            continue
            
        valore_str = str(value).strip().lower()
        if valore_str == "":
            continue
            
        valori_attesi += 1
        if valore_str in testo_estratto_lower:
            valori_trovati += 1
        else:
            errori_visivi.append({"chiave": key, "valore_illeggibile": valore_str})

    visual_match_rate = (valori_trovati / valori_attesi) * 100 if valori_attesi > 0 else 0.0

    return {
        "status": "Passato" if visual_match_rate > 90.0 else "Allerta Rendering",
        "file_size_kb": round(file_size_kb, 1),
        "visual_match_rate_percent": round(visual_match_rate, 2),
        "rendering_errors": errori_visivi
    }

def calculate_kpis(synthetic_records: list, metadata: dict, run_dir: str) -> dict:
    """Calcola i KPI logici (JSON) e fisici (PDF) sui record sintetici."""
    schema = metadata["schema"]
    source_data = metadata["source_data"]
    total_records = len(synthetic_records)
    
    if total_records == 0:
        return {"error": "Nessun record sintetico trovato."}

    expected_keys = [field["field"] for field in schema]
    sensitive_keys = [field["field"] for field in schema if field.get("sensitive", False)]
    
    total_expected_keys_checked = total_records * len(expected_keys)
    matched_keys_count = 0
    pii_leakage_count = 0
    total_sensitive_checks = total_records * len(sensitive_keys)
    
    # Metriche fisiche
    pdf_analizzati = 0
    somma_visual_match_rate = 0.0
    errori_rendering_totali = []

    for i, record in enumerate(synthetic_records):
        # 1. KEY-VALUE MATCH RATE (Logico)
        for key in expected_keys:
            if key in record and record[key] is not None and str(record[key]).strip() != "":
                matched_keys_count += 1
                
        # 2. PII LEAKAGE RATE (Logico)
        for key in sensitive_keys:
            if key in record and key in source_data:
                synth_val = str(record[key]).strip().lower()
                source_val = str(source_data[key]).strip().lower()
                if synth_val == source_val and synth_val != "":
                    pii_leakage_count += 1

        # 3. VISUAL MATCH RATE (Fisico - Round Trip OCR) --> qualità del rendering visivo
        # Supporta sia PDF multipli (synthetic_invoice_1.pdf, ...) sia singolo (synthetic_invoice.pdf)
        pdf_path = os.path.join(run_dir, f"synthetic_invoice_{i+1}.pdf")
        if not os.path.exists(pdf_path) and i == 0:
            pdf_path = os.path.join(run_dir, "synthetic_invoice.pdf")
        if os.path.exists(pdf_path):
            logger.info(f"Analisi visiva su: {pdf_path}")
            ocr_text = extract_text_from_pdf(pdf_path)
            physical_report = check_pdf_physical_quality(pdf_path, record, ocr_text)
            
            somma_visual_match_rate += physical_report["visual_match_rate_percent"]
            pdf_analizzati += 1
            if physical_report["rendering_errors"]:
                errori_rendering_totali.extend(physical_report["rendering_errors"])

    # Calcolo Medie e Percentuali
    key_value_match_rate = (matched_keys_count / total_expected_keys_checked) * 100 if total_expected_keys_checked > 0 else 0.0
    pii_leakage_rate = (pii_leakage_count / total_sensitive_checks) * 100 if total_sensitive_checks > 0 else 0.0
    avg_visual_match_rate = (somma_visual_match_rate / pdf_analizzati) if pdf_analizzati > 0 else 0.0
    schema_compliance = "Passato" if key_value_match_rate == 100.0 else "Fallito"

    return {
        "run_id": metadata.get("run_id"),
        "total_records_evaluated": total_records,
        "pdf_physically_evaluated": pdf_analizzati,
        "kpi_metrics": {
            "key_value_match_rate_percent": round(key_value_match_rate, 2),
            "pii_leakage_rate_percent": round(pii_leakage_rate, 2),
            "visual_match_rate_percent": round(avg_visual_match_rate, 2),
            "schema_compliance": schema_compliance
        },
        "details": {
            "rendering_issues_found": len(errori_rendering_totali),
            "sample_rendering_errors": errori_rendering_totali[:5] # Mostriamo solo i primi 5 per non intasare il log
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Monitoring KPI Logici e Fisici per Dati Sintetici")
    parser.add_argument("--run_id", type=str, required=False, default=None, help="(opzionale) ID run per il report")
    
    args = parser.parse_args()
    logger.info(f"Avvio monitoring su: {OUTPUT_DIR}")
    
    synthetic_records, metadata = load_run_data()
    
    if not synthetic_records or not metadata:
        logger.error("Impossibile procedere con il monitoring. Dati mancanti.")
        return

    # Calcolo KPI
    report = calculate_kpis(synthetic_records, metadata, OUTPUT_DIR)
    
    # Stampa a schermo stile Dashboard
    print("\n" + "="*55)
    print(f"  📊 REPORT MONITORING DATI SINTETICI (END-TO-END)  ")
    print("="*55)
    print(f" Run ID:              {report['run_id']}")
    print(f" Dati JSON Generati:  {report['total_records_evaluated']}")
    print(f" PDF Fisici Testati:  {report['pdf_physically_evaluated']}")
    print("-" * 55)
    print(f" 🧠 Qualità Logica (Agenti JSON):")
    print(f"    - Key-Value Match:    {report['kpi_metrics']['key_value_match_rate_percent']}%")
    
    leakage = report['kpi_metrics']['pii_leakage_rate_percent']
    leakage_str = f"    - PII Leakage Rate:   🚨 {leakage}%" if leakage > 0 else f"    - PII Leakage Rate:   🛡️ {leakage}% (Sicuro)"
    print(leakage_str)
    
    print("-" * 55)
    print(f" 🖨️  Qualità Fisica (Rendering PDF):")
    if report['pdf_physically_evaluated'] > 0:
        vmr = report['kpi_metrics']['visual_match_rate_percent']
        vmr_icon = "✅" if vmr > 95.0 else ("⚠️" if vmr > 80.0 else "❌")
        print(f"    - Visual Match Rate:  {vmr_icon} {vmr}%")
        if report['details']['rendering_issues_found'] > 0:
            print(f"    - Errori di layout:   Trovati {report['details']['rendering_issues_found']} problemi di testo tagliato/sovrapposto.")
    else:
        print("    - Visual Match Rate:  N/A (Nessun file PDF trovato nella cartella)")

    print("="*55 + "\n")
    
    # Salvataggio del report
    report_path = os.path.join(OUTPUT_DIR, "monitoring_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Report dettagliato salvato in: {report_path}")

if __name__ == "__main__":
    main()
