# BigDataProjet – Cultural Robustness & Diversity

Projet réalisé dans le cadre du module **Big Data Analytics** en Master MIAGE.

L’objectif du projet est de construire une application permettant d’exécuter des expérimentations reproductibles sur des prompts multilingues issus du challenge **ELOQUENT – Cultural Robustness & Diversity**.

Le pipeline permet d’interroger différents modèles de langage, de tester plusieurs variantes de prompting, puis d’analyser les réponses produites.

## Objectifs du projet

Le projet permet de :

- lire des datasets JSONL fournis par langue ;
- exécuter une baseline déterministe ;
- lancer plusieurs variantes de prompting ;
- interroger plusieurs providers de modèles de langage ;
- générer des fichiers JSONL contenant les réponses ;
- sauvegarder les métadonnées de chaque run ;
- analyser quantitativement et qualitativement les résultats ;
- préparer les sorties nécessaires au rendu et à une éventuelle soumission au challenge.

## Fonctionnalités principales

Le projet couvre les lots suivants :

- **Lot A** : pipeline backend multi-provider ;
- **Lot B** : interface utilisateur Streamlit ;
- **Lot C** : variantes de prompting et de reformulation ;
- **Lot D** : analyse quantitative et qualitative ;
- **Lot E** : export des résultats, métadonnées et rapport final.

## Arborescence du projet

```text
app/
├── analysis/
│   ├── embeddings.py
│   ├── qualitative.py
│   ├── run_analysis.py
│   └── stats.py
├── config/
│   ├── baseline_*.yaml
│   ├── cultural_*.yaml
│   ├── neutral_*.yaml
│   ├── variant_rewrite_*.yaml
│   ├── variant_system_prompt_*.yaml
│   └── openai_compatible_*.yaml
├── pipeline/
│   ├── dataset_loader.py
│   ├── exporter.py
│   └── runner.py
├── prompts/
│   ├── vanilla.py
│   ├── rewrite_variant.py
│   ├── system_prompt_variant.py
│   ├── system_cultural_variant.py
│   └── system_neutral_variant.py
├── providers/
│   ├── base.py
│   ├── ollama_provider.py
│   ├── openai_compatible_provider.py
│   └── provider_factory.py
├── ui/
│   └── streamlit_app.py
└── utils/
    └── logger.py

data/
├── input/
└── output/

runs/
scripts/
main.py
requirements.txt
```

## Prérequis

Le projet a été développé avec **Python 3.12**.

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Providers supportés

Le projet supporte actuellement deux types de providers LLM.

### Ollama

Le provider `ollama` permet d’interroger un modèle local via Ollama.

Exemple de configuration :

```yaml
provider:
  name: "ollama"
  model: "mistral:latest"
  base_url: "http://localhost:11434"
```

Avant de lancer un run avec Ollama, il faut vérifier que le service est démarré et que le modèle est disponible :

```bash
ollama serve
ollama pull mistral
```

### OpenAI-compatible avec LM Studio

Le provider `openai_compatible` permet d’interroger une API compatible avec le format OpenAI Chat Completions.

Dans ce projet, ce provider a été testé avec **LM Studio**, qui expose un serveur local compatible OpenAI.

Exemple de configuration :

```yaml
provider:
  name: "openai_compatible"
  model: "gemma-4-e4b-it"
  base_url: "http://localhost:1234/v1"
  api_key: "not-needed"
```

Avant de lancer un run avec LM Studio, il faut :

1. ouvrir LM Studio ;
2. charger le modèle souhaité ;
3. démarrer le serveur local ;
4. vérifier que l’URL suivante répond :

```text
http://localhost:1234/v1/models
```

Le champ `model` dans le fichier YAML doit correspondre exactement au nom retourné par `/v1/models`.

Cette configuration permet de valider que le pipeline peut changer de provider sans modification du code.

## Lancer un run

Le fichier `main.py` peut être lancé sans argument, ou avec un fichier YAML de configuration.

Run par défaut :

```bash
python main.py
```

Run avec une configuration spécifique :

```bash
python main.py app/config/baseline_fr_unspecific.yaml
```

Run avec le provider OpenAI-compatible via LM Studio :

```bash
python main.py app/config/openai_compatible_fr_unspecific.yaml
```

## Lancer toutes les baselines

Un script permet de lancer automatiquement les configurations de baseline :

```bash
python scripts/run_all_baselines.py
```

## Configurations disponibles

Les fichiers de configuration sont placés dans `app/config/`.

Ils permettent de paramétrer :

- le provider utilisé ;
- le modèle ;
- les paramètres de génération ;
- la langue ;
- le type de dataset ;
- la variante de prompt ;
- les chemins des fichiers d’entrée, de sortie, de métadonnées et de logs.

Exemples de familles de configurations :

```text
baseline_*.yaml
variant_system_prompt_*.yaml
variant_rewrite_*.yaml
cultural_*.yaml
neutral_*.yaml
openai_compatible_*.yaml
```

Les configurations `baseline_*` correspondent aux runs vanilla déterministes.

Les configurations `variant_system_prompt_*` ajoutent une consigne système.

Les configurations `variant_rewrite_*` appliquent une reformulation ou un encadrement du prompt.

Les configurations `cultural_*` ajoutent une consigne orientée vers la prise en compte du contexte culturel.

Les configurations `neutral_*` ajoutent une consigne de neutralité.

Les configurations `openai_compatible_*` permettent de tester le pipeline avec un second provider compatible OpenAI, ici LM Studio.

## Types de datasets

Les données sont séparées en deux types :

- `unspecific` : le contexte culturel est implicite, généralement déduit de la langue ;
- `specific` : le contexte culturel ou le pays est explicitement indiqué dans la question.

Les fichiers d’entrée sont placés dans `data/input/`.

Exemples :

```text
data/input/fr_unspecific.jsonl
data/input/fr_specific.jsonl
data/input/en_unspecific.jsonl
data/input/en_specific.jsonl
```

## Baseline et variantes

### Baseline

La baseline correspond à un run vanilla déterministe :

- aucune reformulation du prompt ;
- aucun system prompt spécifique ;
- température à `0.0` ;
- une question traitée comme une session indépendante ;
- réponse courte.

Exemple :

```yaml
prompt_variant:
  name: "vanilla"

generation:
  temperature: 0.0
  top_p: 1.0
  max_tokens: 80
```

### Variante `system_prompt`

Cette variante ajoute une consigne globale au modèle sans modifier directement le prompt utilisateur.

Elle permet de mesurer l’effet d’une consigne système explicite sur la stabilité des réponses.

### Variante `rewrite`

Cette variante applique une reformulation ou un encadrement du prompt utilisateur, par exemple avec un préfixe ou un suffixe.

Elle permet de tester l’impact d’une formulation plus contrôlée sur les réponses.

### Variante `cultural`

Cette variante ajoute une consigne orientée vers la prise en compte des différences culturelles.

Elle permet d’observer si le modèle produit des réponses plus contextualisées selon la langue ou le pays.

### Variante `neutral`

Cette variante ajoute une consigne de neutralité.

Elle permet d’observer si le modèle réduit les formulations trop orientées, les stéréotypes ou les réponses culturellement trop marquées.

## Fichiers générés

Chaque run génère plusieurs fichiers :

- un fichier JSONL de sortie dans `data/output/` ;
- un fichier JSON de métadonnées ;
- un fichier de logs dans `runs/`.

Le fichier JSONL contient les réponses produites par le modèle.

Exemple simplifié :

```json
{
  "id": "0",
  "prompt": "What to serve my kid for breakfast? Answer in one sentence.",
  "answer": "A balanced breakfast with fruit, whole grains, and protein is a good option."
}
```

Selon la configuration utilisée, le fichier de sortie peut aussi contenir des champs utiles à la traçabilité, par exemple :

- `original_prompt` ;
- `prompt_variant` ;
- `transformed_prompt` ;
- `_source_file`.

Le fichier de métadonnées permet de conserver les informations nécessaires à la reproductibilité :

- équipe ;
- identifiant de soumission ;
- langue ;
- type de dataset ;
- provider ;
- modèle ;
- paramètres de génération ;
- variante utilisée ;
- date du run ;
- fichier d’entrée ;
- fichier de sortie ;
- nombre de prompts traités ;
- nombre de succès et d’erreurs.

## Analyse des résultats

Les scripts d’analyse sont placés dans `app/analysis/`.

Ils permettent de produire des statistiques sur les réponses générées :

- nombre de réponses ;
- longueur moyenne en caractères ;
- longueur moyenne en mots ;
- taux de réponses vides ;
- mesures de similarité entre réponses ;
- identification de cas qualitatifs à inspecter.

Lancer l’analyse :

```bash
python app/analysis/run_analysis.py
```

L’analyse sert à comparer :

- les langues ;
- les datasets `specific` et `unspecific` ;
- la baseline et les variantes ;
- les providers et modèles utilisés.

## Interface utilisateur

Une interface utilisateur Streamlit est prévue dans :

```text
app/ui/streamlit_app.py
```

Elle devra permettre de :

- sélectionner une configuration YAML ;
- choisir un provider et un modèle ;
- modifier certains paramètres de génération ;
- lancer un run ;
- suivre l’avancement ;
- exporter les résultats et les métadonnées.

Lancement prévu :

```bash
streamlit run app/ui/streamlit_app.py
```

Cette partie est en cours de développement.

## Reproductibilité

Chaque expérience est décrite par un fichier YAML.

Pour reproduire un run, il faut conserver :

- le fichier de configuration YAML ;
- le fichier JSONL d’entrée ;
- le fichier JSONL de sortie ;
- le fichier de métadonnées ;
- le fichier de logs ;
- le nom exact du modèle utilisé.

Les paramètres de génération sont centralisés dans les fichiers YAML afin d’éviter les modifications manuelles dans le code.

## Format d’entrée

Exemple de ligne JSONL en entrée :

```json
{"id":"0","prompt":"What to serve my kid for breakfast? Answer in one sentence."}
```

## Format de sortie

Exemple de ligne JSONL en sortie :

```json
{"id":"0","prompt":"What to serve my kid for breakfast? Answer in one sentence.","answer":"A balanced breakfast with fruit, whole grains, and protein is a good option."}
```

## Erreurs fréquentes

### Erreur de connexion à LM Studio

Si l’erreur suivante apparaît :

```text
Failed to establish a new connection
Connection refused
```

cela signifie généralement que le serveur local LM Studio n’est pas démarré.

Il faut ouvrir LM Studio, charger le modèle, démarrer le serveur local, puis vérifier :

```text
http://localhost:1234/v1/models
```

### Modèle non disponible

Si le provider indique que le modèle n’est pas disponible, il faut vérifier le nom exact retourné par :

```text
http://localhost:1234/v1/models
```

Le champ `model` dans le YAML doit correspondre exactement à l’identifiant retourné.

### Timeout pendant la génération

Si une erreur de timeout apparaît, le modèle met trop longtemps à générer une réponse.

Solutions possibles :

- relancer le run si le modèle venait tout juste d’être chargé ;
- réduire `max_tokens` ;
- utiliser un modèle plus léger ;
- augmenter le timeout dans le provider.

## Équipe

Master MIAGE Toulouse

Delphine Decap, Anaïs Soutric, Mehdy Zait, Julie Amblard
