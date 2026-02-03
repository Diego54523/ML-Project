# MRI Classification – Machine Learning Project

Progetto di Machine Learning per la **classificazione di immagini MRI**, sviluppato come elaborato universitario.  
Il progetto include pipeline complete di **training**, **feature extraction**, **classificazione** e una **demo interattiva di inferenza con explainability**.

---

## 📁 Struttura del progetto

```text
ML-PROJECT/           
├── docs/               # Relazione
├── notebooks/          # Notebook esplorativi
│   ├── images/         # Immagini per i notebook
├── results/            # Output (pesi, feature, risultati)
├── src/
│   ├── data/           # Utility e dataset helper
│   ├── demo/           # Demo interattiva (Streamlit)
│   ├── explainability/ # Codice per interpretabilità
│   ├── models/         # Definizione modelli
│   ├── training/       # Training ed estrazione feature
│   └── main.py         # Entry point CLI
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt    # Dipendenze
```

---

## ⚙️ Setup ambiente

### 1. Creare un ambiente virtuale (consigliato)

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows
```

### 2. Installare le dipendenze

```bash
pip install -r requirements.txt
```

---

### 🔌 Dipendenza RadNet (esecuzione offline)

Il modello **RadNet** utilizza backbone pre-addestrati provenienti dal repository  
[Warvito/radimagenet-models](https://github.com/Warvito/radimagenet-models).

Per garantire **riproducibilità completa** ed evitare dipendenze da connessione internet
durante l’esecuzione, il repository viene caricato **localmente**.

#### Setup RadNet (obbligatorio solo per RadNet)

Dalla root del progetto:

```bash
mkdir -p src/third_party
git clone https://github.com/Warvito/radimagenet-models src/third_party/radimagenet-models
```

Il codice utilizzerà automaticamente la versione **locale** del repository.
In assenza della cartella, verrà tentato il download tramite `torch.hub`.

> ℹ️ Nota: la cartella `third_party/` è esclusa dal versionamento Git.

---

### 📦 Dataset

Il dataset deve essere organizzato in formato **ImageFolder** (una cartella per classe), ad esempio:
```text
MRI/
├── Mild/
├── Moderate/
├── Non/
└── Very Mild/
```

Creare un file `.env` nella root del progetto e specificare il percorso del dataset:
```env
DATA_PATH=percorso/alla/cartella/MRI
```

---

### 🚀 Esecuzione pipeline (CLI)

Il progetto espone un **entry point unico**:
```bash
python src/main.py --task <task>
```

#### Task disponibili:
- `train_cnn` - training Custom CNN
- `train_radnet` - training RadNet
- `train_svm` - training SVM su feature estratte 
- `train_mlp` - training MLP su feature estratte
- `train_mlp_on_radnet` - training MLP su feature RadNet
- `extract_resnet` - estrazione feature da ResNet18
- `extract_custom` - estrazione feature da Custom CNN
- `extract_radnet` - estrazione feature da RadNet

Esempio:
```bash
python src/main.py --task train_cnn
```

Tutti gli output (pesi, feature, risultati) vengono salvati nella cartella `results/`.

---

### 🖥️ Avvio demo interattiva

È disponibile una **demo interattiva** che permette di testare il modello in **fase di inferenza** su immagini MRI.

La demo consente di:
- caricare un'immagine MRI
- visualizzare l'immagine di input
- visualizzare la **ground truth** (se disponibile)
- visualizzare la **predizione del modello**
- visualizzare la **heatmap di attivazione** (opzionale) per interpretabilità

#### Avvio demo

```bash
streamlit run src/demo/demo.py
```

#### Modalità disponibili
- **Dataset mode**: Selezione di immagini da un dataset ImageFolder, con ground truth reale (derivata dalla cartella).
- **Single image mode**: Caricamento manuale di una singola immagine (ground truth non disponibile).

La demo utilizza **modelli già addestrati** e permette esclusivamente
di testare il comportamento dei classificatori in **fase di inferenza**.

Il training dei modelli e l’estrazione delle feature vengono effettuati
tramite pipeline dedicate eseguibili da CLI e non sono parte della demo.

---

## 📄 Relazione

La relazione è pensata come documento di supporto alla comprensione delle scelte metodologiche adottate durante lo sviluppo del progetto: 

👉 [Relazione.pdf](docs/Relazione.pdf)

---

## 👨‍🏫 Note per il docente

Per testare rapidamente il progetto: 
1. configurare `DATA_PATH` in `.env`
2. installare le dipendenze
3. avviare la demo interattiva (`streamlit run src/demo/demo.py`)
4. oppure eseguire training ed estrazione feature tramite CLI (`python src/main.py --task <task>`)

### Ulteriori
Nel caso il modello RadiMagnet (radnet) non dovesse funzionare nonostante l'installazione dei pacchetti opzionali per PyTorch nei [requirements.txt](requirements.txt) vedere il link di drive ufficiale della [rete](https://drive.google.com/uc?export=download&id=1VOWHgOq0rm7OkE_JxlWXhMAH4CvcXUHT).

---

## 👥 Gruppo di lavoro

Il presente progetto è stato sviluppato come elaborato di gruppo per il corso di Machine Learning.

**Membri del gruppo:**
- Diego Martinez - [Diego54523](https://github.com/Diego54523)
- Emanuele Galiano - [emanuelegaliano](https://github.com/emanuelegaliano)
- Andrea Tirenti - [Dr-Faxzty](https://github.com/Dr-Faxzty)