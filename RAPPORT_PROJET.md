# Pipeline ML-Ops de traduction Français ↔ Wolof

## Fine-tuning NLLB-200 avec LoRA, suivi MLflow et déploiement Docker

## Contexte & Objectif

Le wolof est une langue à faibles ressources linguistiques : peu de corpus parallèles, peu d'outils outillés, qualité de traduction commerciale très inégale. Ce projet vise à fournir une API de traduction bidirectionnelle FR↔WO exploitable en production, en adaptant un modèle multilingue généraliste (NLLB-200) au couple wolof-français via fine-tuning paramétriquement efficace, plutôt qu'un réentraînement complet hors de portée d'un GPU unique.

## Architecture Technique

| Composant | Rôle | Techno |
|---|---|---|
| [training/kaggle_training.ipynb](training/kaggle_training.ipynb) | Fine-tuning bidirectionnel + tracking expériences | HuggingFace Transformers, PEFT, MLflow |
| [setup_module.py](setup_module.py) | Récupération de l'adapter depuis Kaggle ou MLflow Registry | argparse, shutil, MLflow client |
| [deployment/inference.py](deployment/inference.py) | Chargement modèle base + adapter, génération beam search | Transformers, PEFT, PyTorch |
| [deployment/app.py](deployment/app.py) | Exposition REST et orchestration des requêtes | Flask, Flask-CORS |
| [deployment/Dockerfile](deployment/Dockerfile) + [deployment/docker-compose.yml](deployment/docker-compose.yml) | Conteneurisation et runtime de service | gunicorn (gthread), image PyTorch CUDA |

## Fonctionnement

```
[Dataset HF galsenai]  ──► [Tokenisation NLLB]  ──► [Fine-tuning LoRA, 4 epochs]
                                                              │
                                                              ▼
                                       [MLflow Registry + adapter LoRA versionné]
                                                              │
                                                              ▼
                                  [setup_module.py → ./deployment/model/]
                                                              │
                                                              ▼
                          [Docker build → gunicorn (2 workers × 2 threads)]
                                                              │
                                                              ▼
   Client HTTP ──► POST /predict ──► TranslationInference.translate() ──► JSON
```

1. Chargement du dataset `galsenai/centralized_wolof_french_translation_data` (98 345 paires), doublement bidirectionnel FR→WO + WO→FR (196 690 exemples), split 88 / 6 / 6.
2. Tokenisation NLLB avec préfixage explicite du code langue cible (`fra_Latn` ou `wol_Latn`) en tête des labels.
3. Fine-tuning LoRA (r=16, α=32, dropout 0.05) sur `q_proj` et `v_proj`, 4 epochs, lr 1.2e-4, fp16, beam search 4, sélection du best checkpoint sur BLEU.
4. Log de l'adapter (`adapter_model.safetensors`, 9.4 MB) et des métriques dans MLflow.
5. Récupération locale via `setup_module.py`, montage du dossier `./model/` dans le conteneur.
6. Au runtime, chargement de `facebook/nllb-200-distilled-600M` puis attachement de l'adapter via `PeftModel.from_pretrained`.

## Points techniques notables

- **Fine-tuning paramétriquement efficace (LoRA)** — 2 359 296 paramètres entraînables sur 1 404 497 920, soit **0,17 %** du modèle. L'adapter pèse 9.4 MB contre 2.4 GB pour le modèle base, ce qui permet de versionner et redistribuer l'adaptation au domaine indépendamment du backbone.
- **Conditionnement de la génération par token de langue forcé** — `forced_bos_token_id` pointe vers l'ID du code langue cible (`fra_Latn` / `wol_Latn`), exigence spécifique des modèles NLLB pour orienter le décodage sans paramètres supplémentaires ([inference.py:228](deployment/inference.py#L228)).
- **Apprentissage bidirectionnel dans un run unique** — chaque paire est dupliquée et marquée par `tgt_lang_code` dès la tokenisation, ce qui mutualise l'entraînement et évite de maintenir deux modèles distincts.
- **Sélection du meilleur checkpoint sur BLEU** — `load_best_model_at_end=True` avec `metric_for_best_model="bleu"` sur la validation. La métrique chrF est loggée en parallèle pour capter les similarités sous-token, plus pertinentes pour une langue morphologiquement riche comme le wolof.
- **Suivi d'expérimentations MLflow** — paramètres LoRA, arguments de training, métriques par epoch et artefacts (adapter, tokenizer) loggés dans un Registry, avec récupération orchestrée par `setup_module.py --source mlflow`.
- **Service web avec workers persistants** — 2 workers × 2 threads en `gthread`, timeout 120 s, healthcheck Docker toutes les 30 s sur `/health`, redémarrage `on-failure`.

## Interfaces exposées

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/health` | GET | Disponibilité + état CUDA |
| `/model-info` | GET | Métadonnées du modèle chargé |
| `/predict` | POST | Traduction unitaire avec auto-détection de langue |
| `/batch-predict` | POST | Lot jusqu'à 100 textes par requête |

Paramètres `/predict` : `text` (requis, ≤ 1000 caractères), `source_lang`, `target_lang`, `num_beams` (1-8).

## Stack & compétences

- **NLP / Deep Learning** : NLLB-200, fine-tuning LoRA (PEFT), beam search, métriques BLEU et chrF (sacrebleu).
- **ML-Ops** : MLflow tracking + Model Registry, versionnement d'adapter, séparation stricte train / inference.
- **Backend** : Flask, gunicorn, validation des entrées et logging structuré.
- **Infra** : Docker, docker-compose, healthchecks, image PyTorch CUDA.

## Résultat

Une API conteneurisée servant un modèle NLLB-200-distilled-600M augmenté de l'adapter LoRA fine-tuné, atteignant **BLEU 18.89** et **chrF 40.10** sur le set de test (11 802 paires), avec une latence d'inférence de **100–250 ms** par requête sur GPU (≈ 4 GB VRAM, ≈ 2 GB RAM). Le pipeline d'entraînement, l'adapter et l'API sont découplés, ce qui permet de réentraîner sur de nouvelles données et de promouvoir une nouvelle version du modèle sans toucher au code de déploiement.

*Le même schéma — modèle multilingue généraliste + adapter LoRA spécialisé + API conteneurisée — s'applique à toute langue ou domaine sous-représenté nécessitant une adaptation rapide et économe : terminologie médicale, juridique, langues régionales africaines ou variétés dialectales.*
