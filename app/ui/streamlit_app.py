import streamlit as st
import os
import yaml
import pandas as pd
import json
import time  # Pour simuler le temps de traitement dans un premier temps


# Configuration de la page Streamlit 
st.set_page_config(
    page_title="Challenge ELOQUENT 2026 - Configuration",
    page_icon="🌍",
    layout="wide"
)

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.title("⚙️ Paramètres du Run")
    st.caption("Challenge Cultural Robustness & Diversity")
    st.divider()

    # 1. Sélection du modèle d'IA (Lot A/B)
    st.subheader("🤖 Modèle et Provider")
    provider_type = st.selectbox(
        "Type de Provider",
        options=["Ollama (Local)", "API Extérieure"],
        help="Sélectionnez si le modèle tourne sur votre machine ou via une clé API."
    )
    
    # Par défaut, on met le modèle léger qu'on a téléchargé ensemble
    model_name = st.text_input(
        "Nom exact du modèle",
        value="gemma2:2b",
        help="Doit correspondre exactement au nom dans 'ollama list'."
    )

    st.divider()

    # 2. Sélection des données du Challenge (Contexte du cours)
    st.subheader("📂 Données d'entrée")
    dataset_type = st.radio(
        "Objectif de l'évaluation",
        options=["Specific (Robustesse)", "Unspecific (Diversité)"],
        help="Specific = contexte fixé (cohérence attendue). Unspecific = culture inférée par la langue (variations attendues)."
    )
    
    # Le challenge impose de tester au moins 5 langues pour la baseline
    selected_languages = st.multiselect(
        "Langues à évaluer (5 minimum conseillées)",
        options=["fr", "en", "de", "es", "it", "pt", "nl"],
        default=["fr", "de", "en"],
        help="Sélectionnez les langues sur lesquelles lancer le pipeline."
    )

    st.divider()

    # 3. Hyperparamètres de génération (Gouvernance et Reproductibilité)
    st.subheader("🎛️ Hyperparamètres LLM")
    
    # Pour la baseline, la température DOIT être à 0.0 (Déterminisme)
    temperature = st.slider(
        "Température",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0.0 = Déterministe (Requis pour la baseline). Plus la valeur est haute, plus l'IA est créative."
    )
    
    max_tokens = st.number_input(
        "Longueur max de la réponse (tokens)",
        min_value=10,
        max_value=200,
        value=50,
        step=5,
        help="Le protocole recommande des réponses courtes d'environ une phrase."
    )

# --- ZONE PRINCIPALE (MAIN CONTENT) ---
st.title("🌍 Application Multi-LLM - Challenge ELOQUENT")
st.write("Bienvenue dans l'interface de pilotage. Utilisez la barre latérale pour configurer vos expérimentations.")

st.divider()

# Remplacement de l'affichage récapitulatif par un format structuré
st.subheader("📋 Récapitulatif de la configuration sélectionnée")

# Création de 4 colonnes alignées pour séparer proprement les informations
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="🤖 Modèle sélectionné", value=model_name)

with col2:
    st.metric(label="⚙️ Type de Provider", value=provider_type)

with col3:
    st.metric(label="📊 Objectif / Dataset", value=dataset_type.split()[0]) 
    # .split()[0] permet d'afficher juste "Specific" ou "Unspecific" pour que ce soit plus court

with col4:
    st.metric(label="🎛️ Température", value=f"{temperature}")

# Affichage des langues sous forme de petits badges (st.pills ou st.status)
st.write("**🌐 Langues qui seront injectées dans le pipeline :**")
if selected_languages:
    # On affiche les langues proprement séparées
    badges = " ".join([f"`{lang.upper()}`" for lang in selected_languages])
    st.markdown(badges)
else:
    st.caption("⚠️ Aucune langue sélectionnée. Veuillez en choisir au moins une dans la barre latérale.")

st.divider()

# --- PILOTAGE DU RUN ---
st.subheader("🚀 Lancement de l'évaluation")

# On s'assure que l'utilisateur a sélectionné au moins une langue avant d'activer le bouton
if not selected_languages:
    st.warning("⚠️ Veuillez sélectionner au moins une langue dans la barre latérale pour activer le bouton.")
    run_button = st.button("Lancer le pipeline", disabled=True)
else:
    run_button = st.button("🔥 Lancer le pipeline sur les langues sélectionnées")

# Si l'utilisateur clique sur le bouton
if run_button:
    st.success(f"Démarrage du traitement avec le modèle `{model_name}`...")
    
    # 1. Création des éléments visuels de suivi
    progress_bar = st.progress(0)
    status_text = st.empty()  # Zone de texte dynamique
    
    # 2. Simulation ou appel du pipeline (Lot A)
    # Pour l'instant, on simule le parcours des langues pour tester ton interface
    total_languages = len(selected_languages)
    
    for index, lang in enumerate(selected_languages):
        # Mise à jour du texte de statut
        status_text.markdown(f"⏳ **Traitement en cours :** Langue `{lang.upper()}`...")
        
        # --- ICI SE FERA L'APPEL AU LOT A DE TON COLLÈGUE ---
        # Exemple : app.pipeline.runner.run_pipeline(lang, model_name, temperature)
        time.sleep(1.5)  # On simule 1,5 seconde de calcul par langue
        # ----------------------------------------------------
        
        # Calcul et mise à jour du pourcentage de la barre de progression
        progress_percentage = int((index + 1) / total_languages * 100)
        progress_bar.progress(progress_percentage)
        
    # Fin du traitement
    status_text.markdown("✅ **Traitement terminé avec succès !**")
    st.toast("Pipeline ELOQUENT validé ! 🇪🇺", icon="🏆")

st.divider()

# --- AFFICHAGE DES RÉSULTATS (Étape 1) ---
st.subheader("📊 Aperçu des données et des réponses")

def load_preview_data(languages_to_load):
    data = []
    
    # On boucle sur toutes les langues que l'utilisateur a cochées
    for lang in languages_to_load:
        # On crée le chemin dynamiquement (ex: data/input/de_specific.jsonl)
        file_path = f"data/input/{lang}_specific.jsonl"
        
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        data.append(row)
                        
    if data:
        df = pd.DataFrame(data)
        
        # --- SYNTONISATION AVEC TES VRAIES CLÉS ---
        # On utilise les vrais noms de ton fichier : native_language et country_context
        if "native_language" in df.columns and "country_context" in df.columns:
            df["answer"] = "Simulated LLM answer in " + df["native_language"].str.upper() + " for context " + df["country_context"]
        else:
            # Sécurité au cas où un autre fichier n'aurait pas la même structure
            df["answer"] = "Simulated LLM answer (Vérifier les clés du fichier JSONL)"  
        return df
    else:
        return None

# On passe la liste des langues sélectionnées dans la barre latérale à notre fonction
df_results = load_preview_data(selected_languages)

# --- ZONE D'AFFICHAGE DU TABLEAU INTERACTIF ---
if df_results is not None:
    st.write(f"📊 **Aperçu des données ({len(df_results)} lignes chargées) :**")
    
    # On affiche les vraies colonnes de ton fichier + la nouvelle colonne "answer"
    colonnes_a_afficher = ["id", "country_context", "native_language", "prompt", "answer"]
    
    # Sécurité : on n'affiche que les colonnes qui existent vraiment
    colonnes_valides = [col for col in colonnes_a_afficher if col in df_results.columns]
    
    st.dataframe(df_results[colonnes_valides], use_container_width=True)
else:
    st.info("💡 Aucun fichier correspondant aux langues sélectionnées n'a été trouvé dans `data/input/`. L'aperçu s'affichera dès que les fichiers seront présents.")

# --- EXPORTATION DES RÉSULTATS ---
st.subheader("📦 Soumission au Challenge")
st.write("Téléchargez le package officiel contenant vos résultats formatés pour le concours ELOQUENT.")

# On crée un faux fichier ZIP en mémoire pour tester le bouton de ton interface
# (Le Lot A se chargera de mettre les vrais fichiers dedans plus tard)
fake_zip_content = b"Fichier ZIP simule pour le challenge ELOQUENT"

st.download_button(
    label="✅ Télécharger le Package de Soumission (.zip)",
    data=fake_zip_content,
    file_name="eloquent_submission_2026.zip",
    mime="application/zip",
    help="Cliquez ici pour exporter les fichiers JSONL modifiés et le fichier metadata.json requis."
)