"""
Obiettivo: valutare la qualità dei dati generati da inference.py. 

- Input: 
    --run_id = definisce l'id della run in esecuzione. Coincide con quella dell'inference.py. 

- Definizione di metriche con cui valutare i dati generati. 
- Garantire che alla fine del flusso i dati finali siano validati, coerenti e utilizzabili.

"""



# inserire qui librerie eventuali da importare
import argparse
import os 



def main(): 

    parser = argparse.ArgumentParser(description="Script di monitoring")

    parser.add_argument("--run_id", type=int, help="ID della run attuale")

    # Lettura argomenti
    args = parser.parse_args()

    # Uso argomenti
    print("run id: ", args.run_id)


    ################## TO DO ################### 



if __name__ == "__main__":
    main()