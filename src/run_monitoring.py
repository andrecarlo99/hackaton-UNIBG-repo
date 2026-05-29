"""
Script di esecuzione per monitoringv1.py
Chiama tutte le funzioni principali esposte dal modulo.
"""

import os
import json
from loguru import logger

from monitoringv1 import (
    load_run_data,
    check_pdf_physical_quality,
    calculate_kpis,
    OUTPUT_DIR,
)

# Importa extract_text_from_pdf dalla stessa sorgente usata da monitoringv1
# per evitare definizioni duplicate e comportamenti incoerenti.
import monitoringv1
extract_text_from_pdf = monitoringv1.extract_text_from_pdf


if __name__ == "__main__":

    # ── 1. load_run_data ─────────────────────────────────────────────
    logger.info(">>> Chiamata a load_run_data()")
    synthetic_records, metadata = load_run_data()

    if not synthetic_records or not metadata:
        logger.error("Dati mancanti – impossibile proseguire con le altre funzioni.")
    else:
        logger.success(f"Caricati {len(synthetic_records)} record sintetici.")

        # ── 2. check_pdf_physical_quality ────────────────────────────
        logger.info(">>> Chiamata a check_pdf_physical_quality() per ogni record")
        for i, record in enumerate(synthetic_records):
            pdf_path = os.path.join(OUTPUT_DIR, f"synthetic_invoice_{i+1}.pdf")
            if not os.path.exists(pdf_path) and i == 0:
                pdf_path = os.path.join(OUTPUT_DIR, "synthetic_invoice.pdf")

            if os.path.exists(pdf_path):
                ocr_text = extract_text_from_pdf(pdf_path)
                result = check_pdf_physical_quality(pdf_path, record, ocr_text)
                logger.info(f"  Record {i+1} – status: {result['status']}, "
                            f"visual_match_rate: {result.get('visual_match_rate_percent', 'N/A')}%")
            else:
                logger.warning(f"  Record {i+1} – PDF non trovato: {pdf_path}")

        # ── 3. calculate_kpis ────────────────────────────────────────
        logger.info(">>> Chiamata a calculate_kpis()")
        report = calculate_kpis(synthetic_records, metadata, OUTPUT_DIR)
        print(json.dumps(report, indent=2, ensure_ascii=False))

    # ── 4. Salvataggio report (evita di richiamare main() che usa argparse
    #        e rieseguirebbe l'intera pipeline in modo ridondante)
    report_path = os.path.join(OUTPUT_DIR, "monitoring_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.success(f"Report salvato in: {report_path}")
