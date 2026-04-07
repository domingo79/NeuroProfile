# ==============================
# IMPORT DELLE LIBRERIE
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans

# ==============================
# FASE 1 – CREAZIONE DATASET
# ==============================

# Genero dati casuali (100 utenti)
np.random.seed(42)

data = {
    "socialita"     : np.random.randint(0, 11, 100),
    "creativita"    : np.random.randint(0, 11, 100),
    "organizzazione": np.random.randint(0, 11, 100),
    "rischio"       : np.random.randint(0, 11, 100),
    "energia"       : np.random.randint(0, 11, 100),
}

df = pd.DataFrame(data)

# ==============================
# FASE 2 – MODELLO KMEANS DA SCIKILEARN 
# ==============================

# KMeans è un algoritmo di clustering: divide i dati in K gruppi (cluster) assegnando ogni punto al centroide più vicino.
# Il centroide è calcolato prendendo la media di tutte le coordinate dei punti del cluster
# e serve come riferimento. I punti vengono assegnati al cluster il cui centroide è più vicino

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df) # Addestramento del modello: # Allena KMeans sui dati trovando i K centroidi e assegnando ogni punto al cluster più vicino
df["cluster"] = kmeans.labels_  # Assegna a ogni riga del dataFrame il numero del cluster trovato da KMeans

# ==============================
# FASE 3 – PERSONALITÀ
# ==============================

personalita = {
    0: ("Leader",      "Sei una persona dominante, energica e organizzata."),
    1: ("Creativo",    "Hai una mente aperta e ami pensare fuori dagli schemi."),
    2: ("Equilibrato", "Sai bilanciare bene emozioni, lavoro e relazioni."),
    3: ("Avventuroso", "Ti piace il rischio e cerchi nuove esperienze."),
    4: ("Analitico",   "Sei razionale, preciso e orientato ai dettagli.")
}

# ==============================
# FASE EXTRA – GENERAZIONE DESCRIZIONE AVANZATA
# ==============================

# Questa funzione crea una descrizione testuale più naturale e coinvolgente
# basata sui valori inseriti dall’utente.
# L'obiettivo è simulare una spiegazione "umana" della personalità combinando
# i diversi tratti (socialità, creatività, ecc.) in una narrazione coerente

def genera_descrizione_avanzata(nome, s, c, o, r, e, tipo):
    
    # Lista che conterrà le frasi descrittive dei tratti principali
    tratti = []

    # ==============================
    # ANALISI DEI SINGOLI TRATTI
    # ==============================

    # Socialità: misura quanto una persona è estroversa o riservata
    if s > 7:
        tratti.append("ami stare al centro delle interazioni sociali")
    elif s < 4:
        tratti.append("tendi a preferire ambienti più tranquilli e selettivi")

    # Creatività: capacità di pensiero originale e immaginazione
    if c > 7:
        tratti.append("possiedi una forte immaginazione e pensiero creativo")
    elif c < 4:
        tratti.append("hai un approccio pratico e concreto alle situazioni")

    # Organizzazione: livello di struttura, disciplina e pianificazione
    if o > 7:
        tratti.append("sei estremamente organizzato e orientato agli obiettivi")
    elif o < 4:
        tratti.append("preferisci flessibilità e spontaneità rispetto alla rigidità")

    # Propensione al rischio: quanto una persona è incline a rischiare
    if r > 7:
        tratti.append("sei attratto dalle sfide e dal rischio")
    elif r < 4:
        tratti.append("valuti attentamente le decisioni evitando rischi inutili")

    # Energia: livello di vitalità e dinamismo
    if e > 7:
        tratti.append("hai un livello di energia molto alto e trascinante")
    elif e < 4:
        tratti.append("gestisci le tue energie in modo più riflessivo e controllato")

    # ==============================
    # COSTRUZIONE DEL TESTO FINALE
    # ==============================

    # Introduzione personalizzata con nome e tipo di personalità
    testo = f"{nome}, il tuo profilo '{tipo}' racconta una personalità unica. "

    # Se abbiamo almeno un tratto rilevante, lo trasformiamo in frase fluida
    if tratti:
        testo += "In particolare, " + ", ".join(tratti[:-1])
        if len(tratti) > 1:
            testo += " e " + tratti[-1]
        else:
            testo += tratti[0]

    # Chiusura narrativa per rendere il testo più naturale e completo
    testo += ". Questo mix di caratteristiche definisce il tuo modo distintivo di affrontare il mondo, le relazioni e le sfide quotidiane."

    return testo

# ==============================
# FASE 4 – INTERFACCIA STREAMLIT
# ==============================

st.title("🧬 NeuroProfile: Il Codice della Tua Personalità")   # Inserisco il titolo dell'interfaccia utente

st.write("""
Questa app utilizza un algoritmo di clustering (KMeans) per determinare
il tuo tipo di personalità in base alle tue caratteristiche.
""")

# Input utente
nome = st.text_input("Inserisci il tuo nome")

socialita      = st.slider("Socialità", 0, 10)
creativita     = st.slider("Creatività", 0, 10)
organizzazione = st.slider("Organizzazione", 0, 10)
rischio        = st.slider("Propensione al rischio", 0, 10)
energia        = st.slider("Energia", 0, 10)

# ==============================
# FASE 5 – PREDIZIONE E GRAFICO RADAR
# ==============================

if st.button("Scopri il tuo tipo"):   # Tutto il blocco viene eseguito solo quando l'utente clicca il pulsante

    # Creo un array con i dati inseriti dall'utente
    # Serve il formato [[...]] perché sklearn si aspetta una matrice (anche per un solo utente)
    user_data = np.array([[socialita, creativita, organizzazione, rischio, energia]])

    # Uso il modello KMeans già addestrato per prevedere a quale cluster appartiene l'utente
    # Il risultato è un array, quindi prendo il primo elemento [0]
    cluster = kmeans.predict(user_data)[0]

    # Associo il numero del cluster a un tipo di personalità (nome + descrizione)
    tipo, descrizione = personalita[cluster]

    # Mostro il risultato all'utente con un messaggio evidenziato e la descrizione della personalità
    st.success(f"{nome}, il tuo tipo di personalità è: {tipo}")
    st.write(descrizione)

    # ==============================
    # ANALISI APPROFONDITA PERSONALIZZATA
    # ==============================

    # Genero una descrizione più dettagliata e narrativa basata sui valori inseriti
    descrizione_avanzata = genera_descrizione_avanzata(
        nome,
        socialita,
        creativita,
        organizzazione,
        rischio,
        energia,
        tipo
    )

    # Mostro una sezione dedicata per distinguere l'analisi avanzata dal risultato base
    st.write("### 🔍 Analisi approfondita")

    # Visualizzo il testo generato, che combina i vari tratti in una spiegazione fluida
    st.write(descrizione_avanzata)

    # ==============================
    # FASE 6 – VISUALIZZAZIONE RADAR
    # ==============================

    # Estraggo i centroidi dei cluster trovati da KMeans
    # Ogni centroide rappresenta il profilo medio di un gruppo
    centroids = kmeans.cluster_centers_

    labels = ["Socialità", "Creatività", "Organizzazione", "Rischio", "Energia"]  # Etichette delle variabili (assi del grafico radar)

    num_vars = len(labels)  # Numero totale di variabili (dimensioni del radar)

    # Calcolo gli angoli per ogni asse del radar (in radianti)
    # endpoint=False evita la duplicazione dell’ultimo punto
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    angles += angles[:1]   # Chiudo il grafico aggiungendo il primo angolo alla fine (necessario per il radar)

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))  # Creo la figura con coordinate polari (grafico radar)

    # ==============================
    # DISEGNO DEI CLUSTER (CENTROIDI)
    # ==============================

    for i, centroid in enumerate(centroids):  # Itero su tutti i cluster trovati

        # Converto il centroide in lista e chiudo la forma
        values = centroid.tolist()
        values += values[:1]

        # Recupero il nome del cluster (es. Leader, Creativo, ecc.)
        nome_cluster = personalita[i][0]

        if i == cluster:
            # Evidenzio il cluster a cui appartiene l’utente
            ax.plot(angles, values, linewidth=2, label=f"Profilo medio {nome_cluster}")
            ax.fill(angles, values, alpha=0.25)
        else:
            # Disegno gli altri cluster in modo meno visibile (tratteggiati e trasparenti)
            ax.plot(angles, values, linewidth=1, linestyle="dashed", alpha=0.2)

    # ==============================
    # PROFILO UTENTE
    # ==============================

    # Creo il vettore dei valori inseriti dall’utente
    user_values = [socialita, creativita, organizzazione, rischio, energia]
    
    user_values += user_values[:1]  # Chiudo la forma per il radar

    # Disegno il profilo utente in rosso per evidenziarlo
    ax.plot(angles, user_values, color="red", linewidth=3, label="Il tuo profilo")
    ax.fill(angles, user_values, color="red", alpha=0.3)

    # ==============================
    # PERSONALIZZAZIONE GRAFICO
    # ==============================

    # Imposto le etichette sugli assi
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_yticks(range(0, 11, 2))  # Imposto la scala dei valori (da 0 a 10 con step 2)

    ax.set_title(f"Confronto con il profilo '{tipo}'")  # Titolo del grafico dinamico in base al tipo di personalità

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))  # Mostro la legenda (spostata leggermente fuori dal grafico)

    st.pyplot(fig)  # Visualizzo il grafico in Streamlit