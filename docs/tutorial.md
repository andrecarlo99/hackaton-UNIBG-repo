# Mini tutorial dei tool utilizzati 


## 1️⃣ Visual Studio Code (VS Code) 
VS Code è un **editor di codice leggero e potente**, con supporto per molteplici linguaggi e strumenti.  
- **Estensioni**: aggiungi funzionalità come Python, Jupyter, Docker e Git.  
- **Devcontainers**: puoi sviluppare dentro un **container Docker**, isolando le dipendenze e mantenendo il tuo ambiente pulito.  
- **Terminale integrato**: esegui comandi direttamente senza uscire dall’editor.  

🔗 Puoi installarlo qui: [Visual Studio Code](https://code.visualstudio.com/Download) 

---


## 2️⃣ Docker 
Docker è una piattaforma per creare, distribuire e far girare **applicazioni dentro container leggeri e isolati**.

- **Container**: ambiente isolato dove la tua app gira sempre allo stesso modo
- **Immagini**: blueprint per creare container
- **Dockerfile**: definisce come costruire un’immagine 

Comandi base:  
```bash
docker build -t my-app .      # costruisci un’immagine
docker run -it my-app          # avvia un container
docker ps                      # mostra container in esecuzione
docker images                  # mostra immagini disponibili
docker stop <container_id>     # ferma un container
```

🔗 Puoi installarlo qui: [Docker Desktop](https://www.docker.com/products/docker-desktop/). Non è necessario crearsi un account. 

--- 


## 3️⃣ Git
Git è un **sistema di controllo versione** distribuito. Ti permette di:  
- Salvare **snapshot del tuo codice** (commit)  
- Tornare a versioni precedenti  
- Collaborare con altri usando **branch e merge**  

Comandi base:  
```bash
git clone <repo>      # copia un repository
git add <file>        # aggiungi file al prossimo commit
git commit -m "msg"   # salva modifiche
git push              # invia le modifiche al server remoto
git pull              # scarica le ultime modifiche dal server
```

🔗 Puoi installarlo qui: [Git per Desktop](https://git-scm.com/install/windows). L'utilizzo è facoltativo. 

---

## 4️⃣ Open Router 

OpenRouter è un **gateway** che permette di accedere a diversi modelli di AI (LLM) tramite un’unica API. Consente di utilizzare modelli di provider diversi (OpenAI, Anthropic, Mistral, ecc.) senza dover gestire integrazioni separate. Per utilizzarlo, occorre eseguire i seguenti passi: 
1. Accedere o crearsi un account nel [loro sito](https://openrouter.ai)
2. Andare nella sezione Get API Keys
3. Crea la nuova chiave
4. Copia e conservala (non verrà mostrata nuovamente) 

OpenRouter mette a disposizione diversi **modelli gratuiti (free tier)**, ma la disponibilità può variare e alcuni modelli potrebbero essere temporaneamente non accessibili a causa di elevato traffico. Per vedere i modelli disponibili puoi consultare la pagina `Models` e elencare i modelli in ordine di prezzo crescente. Quando utilizzi un modello gratuito, verifica sempre che sia attivo e disponibile al momento della richiesta. Per ogni utente sono disponibili **50 chiamate gratuite al giorno**. Precisiamo che per lo svolgimento della gara **non è obbligatorio** utilizzare necessariamente OpenRouter (es. si dispone della versione premium di altri modelli). 

