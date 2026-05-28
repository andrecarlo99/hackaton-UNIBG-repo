"""
Obiettivo: definire e eseguire il workflow agentico che genera n dati a partire da un dato iniziale (nella cartella /dataset) 

- Input da linea di comando: 
    --file = nome del file presente nella cartella /dataset su cui generare nuovi dati sintetici
    --n_file =  n dati che bisogna generare
    --run_id : è una stringa del tipo yyyymmdd-hhmm e definisce l'id della run in esecuzione. 
- Output: 
    n dati generati dal processo agentico.

"""


# inserire qui librerie eventuali da importare
import argparse
import os 




def main(): 
    parser = argparse.ArgumentParser(description="Script di inference")

    parser.add_argument("--file", type=str, help="Nome file dato")
    parser.add_argument("--n_file", type=int, help="Numero file da generare")
    parser.add_argument("--run_id", type=int, help="ID della run attuale")

    # Lettura argomenti
    args = parser.parse_args()

    # Definizione del path in cui è presente il file 
    path_file = os.path.join("dataset", args.file)

    # Uso argomenti
    print(path_file)
    print("File: ", args.file)
    print("Numero file: ", args.n_file)
    print("run id: ", args.run_id)


     ################## TO DO ################### 


if __name__ == "__main__":
    main()