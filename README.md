# Hackathon UNIBG - AI Agentic Synthetic Data Generation

Questa sfida si concentra sulla **generazione di dati sintetici di utilizzando un approccio multi-agentico**. 

L’obiettivo principale è **sviluppare un workflow multi-agentico** in grado di:

* **Generare** automaticamente dati coerenti e utili partendo da un input iniziale.
* Valutare la **qualità e la coerenza** dei dati generati attraverso metriche definite.
* **Produrre** un insieme finale di dati validati, pronti per essere utilizzati in applicazioni successive.

La tipologia di dato scelto da cui partire è un documentale. 


Nella cartella /dataset 
 
## Setup dell'ambiente di sviluppo 

Per lavorare con il progetto è consigliato configurare un ambiente Dev Container usando VS Code e Docker. Segui questi passi: 

1. **Scaricare [Visual Studio Code](https://code.visualstudio.com/)**
2. (FACOLTATIVO) **Scaricare [Git per Desktop](https://git-scm.com/install/windows)**, che può essere utilizzato in modo facoltativo per aprire una nuova repository e versionare il codice. 
3. **Scaricare [Docker](https://www.docker.com/products/docker-desktop/)** 
4. **Avviare Docker e tenere il daemon sempre attivo**, per poter avviare il Dev Container.  
5. **Installare l’estensione Dev Containers in VS Code**: da Extensions cercare e installare "Dev Containers" (Microsoft)
6. **Aprire la cartella di lavoro nel Dev Container**: premere Ctrl+Shift+P → selezionare "Dev Containers: Reopen in Container" e confermare l'attuale cartella di lavoro.
7. **Eseguire task di sincronizzazione**: Premere Ctrl+Shift+P → Tasks: Run Tasks → selezionare "uv sync".


Per maggiori dettagli su questi tool, ulteriori indicazioni sono state indicate nel file [./docs/tutorial.md]



## Struttura della repository


📁 **/dataset** = contiene il dato su cui si lavorerà, di tipo documentale. In particolare, saranno presenti due fatture in formato pdf, appartenenti a due settori diversi. <br>
📁 **/docker** = nel caso si vogliano utilizzare ulteriori immagini docker, possono essere inserite qui <br>
📁 **/docs** = contiene un breve documento di spiegazione dei tool utilizzati e descritti nella sezione precedente <br>
📁 **/notebooks** = contiene due notebook di esempio e tutorial sul caricamento dei dati e sull'utilizzo di OpenRouter e Langgraph <br>
📁 **/src** = contiene i due script che dovranno essere modificati per la sfida: inference.py e monitoring.py <br>



## Getting started

Una volta configurato l'ambiente di setup, occorre ronominare il file `EXAMPLE.env` in `.env`. In questo file verranno inserite tutte le **variabili di configurazione** e dati sensibili (es. **chiavi API**), separandoli dal codice sorgente. Qualora si voglia versionare tramite Git, questo file non verrà caricato.  

Per completare la sfida, occorre modificare i due script presenti nella cartella `src`. Entrambi gli script prendono argomenti da linea di comando, pertanto per lanciarli occorre lanciare i seguenti comandi: 



```bash
python inference.py --file invoice-001.pdf --n_file 30 --run_id 20260519-1650
```

```bash
python monitoring.py --run_id 20260519-1650
```

