# BigDataProjet – Partie A : Pipeline backend multi-provider

## Objectif

Cette partie du projet met en place le backend expérimental demandé dans le sujet :
- lecture d’un dataset au format JSONL ;
- interrogation d’un modèle de langage ;
- génération d’une réponse courte pour chaque prompt ;
- sauvegarde des résultats dans un fichier JSONL avec le champ `answer` ;
- sauvegarde des métadonnées du run ;
- architecture modulaire permettant de changer de provider sans réécrire le pipeline.

Le backend est conçu pour être utilisé ensuite par :
- la partie B : interface de lancement ;
- la partie C : variantes de prompting ;
- la partie D : analyse des résultats.

---

## Arborescence utile

```text
app/
├── config/
│   ├── default.yaml
│   ├── baseline_en.yaml
│   ├── baseline_es.yaml
│   ├── baseline_de.yaml
│   └── baseline_it.yaml
├── pipeline/
│   ├── dataset_loader.py
│   ├── exporter.py
│   └── runner.py
├── providers/
│   ├── base.py
│   ├── ollama_provider.py
│   ├── openai_compatible_provider.py
│   └── provider_factory.py
└── utils/
    └── logger.py

data/
├── input/
└── output/

runs/
main.py
requirements.txt
```

## Prérequis

- Avoir Python installé (3.12 recommandé)
- Avoir Ollama installé localement
- Avoir au moins un modèle Ollama téléchargé

Exemple avec Mistral :

```bash
ollama pull mistral
```

## Avant d’exécuter le projet

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer Ollama :

```bash
ollama serve
```

## Lancer un run

Run par défaut (fr unspecific):

```bash
python main.py
```

Run avec une configuration spécifique :

```bash
python main.py app/config/baseline_it.yaml
```

## Fichiers générés

Après l’exécution d’un run, le projet génère :
- un fichier JSONL de sortie dans `data/output/` contenant les réponses dans le champ `answer` ;
- un fichier de métadonnées du run ;
- un fichier de logs dans `runs/`.

## Notes pour les autres parties

- La partie B peut appeler directement le pipeline avec un fichier de configuration YAML.
- La partie C peut modifier les prompts ou le `system_prompt` sans changer le cœur du pipeline.
- La partie D peut exploiter directement les fichiers générés dans `data/output/`.

## Exemple de format d’entrée

```json
{"id":"0","prompt":"What to serve my kid for breakfast? Answer in one sentence."}
```

## Exemple de format de sortie

```json
{"id":"0","prompt":"What to serve my kid for breakfast? Answer in one sentence.","answer":"A balanced breakfast with fruit, whole grains, and protein is a good option."}
```