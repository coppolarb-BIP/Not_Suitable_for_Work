# NOT SUITABLE FOR WORK — Dashboard Streamlit

Dashboard costruita sul master reale `Not_Suitable_for_Work_MASTER_Drive.xlsx`.

## Avvio locale

1. Installa Python 3.10+.
2. Apri il terminale nella cartella.
3. Esegui:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Si aprirà la dashboard nel browser.

## Uso con dati aggiornati

Hai due possibilità:

- sostituire `Not_Suitable_for_Work_MASTER_Drive.xlsx` nella stessa cartella con la versione aggiornata;
- oppure usare **Aggiorna il dataset** nella sidebar e caricare il master aggiornato direttamente dalla dashboard.

La dashboard legge:
- `Survey_Data` per KPI, filtri e grafici;
- `ML_Output` per Accuracy training, 5-Fold Cross Validation, fattore dominante,
  feature importance e matrice di confusione.

Non modifica l'Excel.

## Per la demo GDG Palermo

Workflow consigliato:
1. aggiorna `Survey_Data`;
2. esegui la cella Colab della Random Forest, che aggiorna `ML_Output`;
3. salva/sincronizza l'Excel;
4. ricarica il file dalla sidebar della dashboard.

## Pubblicazione

Per renderla accessibile tramite URL puoi caricare questa cartella su GitHub e pubblicarla
su Streamlit Community Cloud. Se il repository è pubblico, evita di includere dati che non
vuoi rendere pubblici.
