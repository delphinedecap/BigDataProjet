from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml


# Permet d'exécuter l'interface même si Streamlit est lancé depuis app/ui
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.runner import run_experiment


CONFIG_DIR = PROJECT_ROOT / "app" / "config"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
RUNS_DIR = PROJECT_ROOT / "runs"


st.set_page_config(
    page_title="BigDataProjet - Interface ELOQUENT",
    page_icon="🧪",
    layout="wide",
)


def list_yaml_configs() -> List[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml")) + sorted(CONFIG_DIR.glob("*.yml"))


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration invalide : {path}")

    config["config_file"] = str(path)
    return config


def write_temp_yaml(config: Dict[str, Any], original_config_path: Path) -> Path:
    temp_dir = PROJECT_ROOT / "runs" / "ui_temp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_path = temp_dir / f"ui_{original_config_path.stem}.yaml"

    config_to_save = deepcopy(config)
    config_to_save.pop("config_file", None)

    with temp_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config_to_save, file, allow_unicode=True, sort_keys=False)

    return temp_path


def apply_overrides(
    config: Dict[str, Any],
    provider_name: Optional[str],
    model_name: Optional[str],
    base_url: Optional[str],
    temperature: Optional[float],
    top_p: Optional[float],
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    updated = deepcopy(config)

    updated.setdefault("provider", {})
    updated.setdefault("generation", {})

    if provider_name:
        updated["provider"]["name"] = provider_name

    if model_name:
        updated["provider"]["model"] = model_name

    if base_url:
        updated["provider"]["base_url"] = base_url

    if temperature is not None:
        updated["generation"]["temperature"] = temperature

    if top_p is not None:
        updated["generation"]["top_p"] = top_p

    if max_tokens is not None:
        updated["generation"]["max_tokens"] = max_tokens

    return updated


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []

    if not path.exists():
        return pd.DataFrame()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return pd.DataFrame(rows)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_project_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None

    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def should_export_file(path: Path) -> bool:
    ignored_parts = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".streamlit",
        "venv",
        ".venv",
    }

    ignored_suffixes = {
        ".pyc",
        ".pyo",
        ".log.lock",
    }

    if any(part in ignored_parts for part in path.parts):
        return False

    if path.suffix in ignored_suffixes:
        return False

    if not path.is_file():
        return False

    return True


def get_existing_file_paths(
    configs: List[Dict[str, Any]],
    selected_config_paths: List[Path],
    include_runs: bool = False,
) -> List[Path]:
    files = []

    for config, config_path in zip(configs, selected_config_paths):
        files.append(config_path)

        config_file = resolve_project_path(config.get("config_file"))
        if config_file and config_file.exists():
            files.append(config_file)

        for key in ["input_path", "output_path", "metadata_path", "log_file"]:
            resolved_path = resolve_project_path(config.get(key))
            if resolved_path and resolved_path.exists():
                files.append(resolved_path)

    if ANALYSIS_DIR.exists():
        files.extend(sorted(path for path in ANALYSIS_DIR.rglob("*") if path.is_file()))

    if include_runs and RUNS_DIR.exists():
        files.extend(sorted(path for path in RUNS_DIR.rglob("*") if path.is_file()))

    unique_files = []
    seen = set()

    for file_path in files:
        if should_export_file(file_path):
            normalized = file_path.resolve()
            if normalized not in seen:
                seen.add(normalized)
                unique_files.append(file_path)

    return unique_files

def build_zip(files: List[Path]) -> bytes:
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in files:
            try:
                arcname = file_path.resolve().relative_to(PROJECT_ROOT)
            except ValueError:
                arcname = file_path.name

            zip_file.write(file_path, arcname=str(arcname))

    buffer.seek(0)
    return buffer.getvalue()


def run_analysis_script() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "app" / "analysis" / "run_analysis.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def show_config_summary(config: Dict[str, Any]) -> None:
    provider = config.get("provider", {})
    generation = config.get("generation", {})
    prompt_variant = config.get("prompt_variant", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Provider", provider.get("name", "N/A"))

    with col2:
        st.metric("Modèle", provider.get("model", "N/A"))

    with col3:
        st.metric("Température", generation.get("temperature", "N/A"))

    with col4:
        st.metric("Variante", prompt_variant.get("name", "vanilla"))


def display_results(configs: List[Dict[str, Any]]) -> None:
    st.subheader("Aperçu des résultats générés")

    result_tabs = []

    for config in configs:
        output_path = resolve_project_path(config.get("output_path"))
        if output_path and output_path.exists():
            result_tabs.append((output_path.name, output_path, config))

    if not result_tabs:
        st.info("Aucun fichier de sortie trouvé pour l’instant. Lance un run ou sélectionne une configuration dont le fichier output existe déjà.")
        return

    tabs = st.tabs([name for name, _, _ in result_tabs])

    for tab, (_, output_path, config) in zip(tabs, result_tabs):
        with tab:
            df = load_jsonl(output_path)

            if df.empty:
                st.warning(f"Le fichier existe mais il est vide : `{output_path}`")
                continue

            st.write(f"Fichier : `{output_path.relative_to(PROJECT_ROOT)}`")
            st.write(f"{len(df)} lignes chargées.")

            preferred_columns = [
                "id",
                "country_context",
                "native_language",
                "prompt",
                "original_prompt",
                "transformed_prompt",
                "prompt_variant",
                "answer",
                "error",
                "_source_file",
            ]
            columns_to_show = [column for column in preferred_columns if column in df.columns]

            if not columns_to_show:
                columns_to_show = list(df.columns)

            st.dataframe(df[columns_to_show], use_container_width=True)

            metadata_path = resolve_project_path(config.get("metadata_path"))
            metadata = load_json(metadata_path) if metadata_path else None

            if metadata:
                with st.expander("Voir les métadonnées du run"):
                    st.json(metadata)


def display_analysis_results() -> None:
    st.subheader("Analyse des résultats")

    report_path = ANALYSIS_DIR / "analysis_report.txt"
    summary_path = ANALYSIS_DIR / "analysis_summary.json"

    if summary_path.exists():
        summary = load_json(summary_path)

        if summary:
            runs = summary.get("runs", {})
            rows = []

            for name, analysis in runs.items():
                stats = analysis.get("stats", {})
                similarity = analysis.get("similarity", {})

                rows.append(
                    {
                        "run": name,
                        "avg_word_count": stats.get("avg_word_count"),
                        "avg_char_length": stats.get("avg_char_length"),
                        "empty_answer_rate": stats.get("empty_answer_rate"),
                        "avg_similarity": similarity.get("avg_similarity"),
                    }
                )

            if rows:
                st.write("Résumé des analyses disponibles :")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("Le résumé d’analyse existe, mais aucun run exploitable n’a été trouvé dedans.")

    if report_path.exists():
        with st.expander("Voir le rapport texte complet"):
            st.text(report_path.read_text(encoding="utf-8"))

    if not summary_path.exists() and not report_path.exists():
        st.info("Aucune analyse générée pour l’instant. Clique sur le bouton ci-dessous après avoir produit des résultats.")

    if st.button("Lancer l’analyse globale"):
        with st.spinner("Analyse en cours..."):
            process = run_analysis_script()

        if process.returncode == 0:
            st.success("Analyse terminée.")
            st.rerun()
        else:
            st.error("L’analyse a échoué.")
            st.code(process.stderr or process.stdout)


st.title("Interface BigDataProjet - Challenge ELOQUENT")
st.write(
    "Cette interface permet de sélectionner des configurations YAML existantes, "
    "de lancer le pipeline, de visualiser les sorties et d’exporter les fichiers utiles."
)

yaml_configs = list_yaml_configs()

if not yaml_configs:
    st.error("Aucune configuration YAML trouvée dans `app/config/`.")
    st.stop()

config_labels = [path.name for path in yaml_configs]

with st.sidebar:
    st.title("Paramètres du run")

    selected_labels = st.multiselect(
        "Configurations YAML à lancer",
        options=config_labels,
        default=[config_labels[0]],
        help="Il est possible de sélectionner une ou plusieurs configurations.",
    )

    selected_paths = [path for path in yaml_configs if path.name in selected_labels]

    st.divider()
    st.subheader("Filtres rapides")

    family_filter = st.selectbox(
        "Filtrer par famille",
        options=["Toutes", "baseline", "variant_rewrite", "variant_system_prompt", "cultural", "neutral", "openai_compatible"],
    )

    if family_filter != "Toutes":
        filtered_paths = [path for path in yaml_configs if path.name.startswith(family_filter)]
        filtered_labels = [path.name for path in filtered_paths]

        st.caption(f"{len(filtered_paths)} configuration(s) trouvée(s).")
        if st.button("Sélectionner cette famille"):
            st.session_state["selected_family"] = filtered_labels
            st.rerun()

    if "selected_family" in st.session_state:
        selected_labels = st.session_state.pop("selected_family")
        selected_paths = [path for path in yaml_configs if path.name in selected_labels]

    st.divider()
    st.subheader("Overrides optionnels")

    enable_overrides = st.checkbox(
        "Modifier temporairement provider/modèle/hyperparamètres",
        value=False,
        help="Les fichiers YAML originaux ne sont pas modifiés. Une config temporaire est créée dans runs/ui_temp_configs.",
    )

    provider_name = None
    model_name = None
    base_url = None
    temperature = None
    top_p = None
    max_tokens = None

    if enable_overrides:
        provider_name = st.selectbox(
            "Provider",
            options=["", "ollama", "openai_compatible"],
            index=0,
        ) or None

        model_name = st.text_input("Nom du modèle", value="").strip() or None
        base_url = st.text_input("Base URL", value="").strip() or None

        temperature = st.slider("Température", 0.0, 1.5, 0.0, 0.1)
        top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.05)
        max_tokens = st.number_input("Max tokens", min_value=1, max_value=2048, value=80, step=10)

tab_run, tab_results, tab_analysis, tab_export = st.tabs(
    ["Lancement", "Résultats", "Analyse", "Export"]
)

loaded_configs = []

for path in selected_paths:
    try:
        loaded_configs.append(read_yaml(path))
    except Exception as error:
        st.error(f"Impossible de charger `{path.name}` : {error}")

if enable_overrides:
    loaded_configs = [
        apply_overrides(
            config,
            provider_name=provider_name,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        for config in loaded_configs
    ]

with tab_run:
    st.subheader("Configurations sélectionnées")

    if not selected_paths:
        st.warning("Sélectionne au moins une configuration YAML.")
    else:
        for path, config in zip(selected_paths, loaded_configs):
            with st.expander(path.name, expanded=len(selected_paths) == 1):
                show_config_summary(config)

                st.write("Entrée :", f"`{config.get('input_path', 'N/A')}`")
                st.write("Sortie :", f"`{config.get('output_path', 'N/A')}`")
                st.write("Métadonnées :", f"`{config.get('metadata_path', 'N/A')}`")
                st.write("Log :", f"`{config.get('log_file', 'N/A')}`")

                with st.expander("Voir le YAML chargé"):
                    st.code(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), language="yaml")

    run_disabled = not selected_paths or not loaded_configs

    if st.button("Lancer le pipeline", disabled=run_disabled):
        progress_bar = st.progress(0)
        status = st.empty()
        errors = []

        for index, (original_path, config) in enumerate(zip(selected_paths, loaded_configs), start=1):
            status.info(f"Lancement de `{original_path.name}`...")

            try:
                if enable_overrides:
                    temp_config_path = write_temp_yaml(config, original_path)
                    config["config_file"] = str(temp_config_path)

                run_experiment(config)
                st.success(f"`{original_path.name}` terminé.")
            except Exception as error:
                errors.append((original_path.name, str(error)))
                st.error(f"Erreur pendant `{original_path.name}` : {error}")

            progress_bar.progress(index / len(loaded_configs))

        if errors:
            st.warning("Certains runs ont échoué. Les runs réussis restent exploitables.")
            with st.expander("Détail des erreurs"):
                for name, error in errors:
                    st.write(f"**{name}**")
                    st.code(error)
        else:
            status.success("Tous les runs sélectionnés sont terminés.")

with tab_results:
    display_results(loaded_configs)

with tab_analysis:
    display_analysis_results()

with tab_export:
    st.subheader("Export des fichiers utiles")

    include_runs = st.checkbox(
        "Inclure le dossier runs/",
        value=True,
    )

    files_to_export = get_existing_file_paths(
        loaded_configs,
        selected_paths,
        include_runs=include_runs,
    )

    if not files_to_export:
        st.info("Aucun fichier exportable trouvé pour les configurations sélectionnées.")
    else:
        st.write("Fichiers qui seront inclus dans le ZIP :")

        export_df = pd.DataFrame(
            {
                "fichier": [
                    str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
                    for path in files_to_export
                ]
            }
        )

        st.dataframe(export_df, use_container_width=True)

        zip_content = build_zip(files_to_export)

        st.download_button(
            label="Télécharger le package ZIP",
            data=zip_content,
            file_name="bigdata_eloquent_export.zip",
            mime="application/zip",
        )