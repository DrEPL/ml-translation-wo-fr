# NLLB FR↔WO Translation - Architecture ML-Ops Complète

Architecture production-ready pour l'entraînement et le déploiement d'un modèle de traduction français <---> wolof basé sur NLLB-200-distilled-600M.

## Vue d'ensemble
 
### Objectifs
- Fine-tuner NLLB-200-distilled-600M sur données Wolof-Français
- Tracker expériences avec MLflow
- Déployer API Flask conteneurisée (Docker)
- Support bidirectionnel FR↔WO avec auto-détection langue

### Résultats observés
- BLEU Score: 18.89 (test)
- chrF Score: 40.10 (test)
- Latence inférence: 100-250ms (single)
- Modèle LoRA: 2.36M params (0.38% du total)

## Structure du projet

```
ml-translation-wo-fr/
├── kaggle_training.ipynb           # Notebook d'entraînement MLflow
├── deployment/
│   ├── app.py                      # API Flask
│   ├── inference.py                # Moteur d'inférence
│   ├── requirements.txt             # Dépendances
│   ├── Dockerfile                   # Image conteneur
│   ├── docker-compose.yml          # Orchestration
│   └── setup_model.py              # Script chargement modèle
└── README.md                       # Cette documentation
```

## Flux principal

1. ENTRAÎNEMENT (Kaggle)
   - Dataset chargement + tokenisation
   - LoRA fine-tuning avec MLflow logging
   - Sauvegarde adapter + metriques

2. SETUP LOCAL
   - Script setup_model.py copie fichiers
   - Préparation pour conteneurisation

3. DÉPLOIEMENT DOCKER
   - Construction image avec Dockerfile
   - Démarrage service avec docker-compose
   - API listen sur 0.0.0.0:5000

4. INFÉRENCE
   - Auto-détection langue source
   - Chargement modèle + adapter LoRA
   - Beam search + retour traduction

## Entraînement sur Kaggle

### Prérequis
- GPU disponible (P100 ou T4)
- Dataset public: galsenai/centralized_wolof_french_translation_data

### Exécution
1. Copier contenu kaggle_training.ipynb dans Kaggle Notebook
2. Exécuter les 8 cellules dans l'ordre
3. Résultats sauvegardés:
   - /kaggle/working/nllb-fr-wo-lora/ (fichiers LoRA)
   - /kaggle/working/mlruns/ (artefacts MLflow)

### Outputs importants
- adapter_config.json (config LoRA)
- adapter_model.bin (poids fins, 2-3 MB)
- tokenizer.model (SentencePiece)
- tokenizer_config.json
- special_tokens_map.json

## Déploiement local

### Étape 1: Préparation modèle

Depuis Kaggle (Option A):
```bash
python setup_model.py \
    --source kaggle \
    --kaggle-path ./nllb-fr-wo-lora \
    --output ./model
```

Depuis MLflow serveur distant (Option B):
```bash
python setup_model.py \
    --source mlflow \
    --mlflow-uri http://mlflow-server:5000 \
    --model-name nllb-fr-wo-translation \
    --model-version production \
    --output ./model
```

### Étape 2: Docker setup

```bash
# Build
docker-compose build

# Démarrer
docker-compose up -d

# Vérifier
docker-compose ps
docker-compose logs -f translation-api
```

### Étape 3: Test

```bash
curl http://localhost:5000/health

# Réponse:
# {"status": "healthy", "model_loaded": true, ...}
```

## Utilisation API

### Endpoint: /predict (Single)

Request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bonjour, comment allez-vous?",
    "source_lang": "fr",
    "target_lang": "wo",
    "num_beams": 4
  }'
```

Response:
```json
{
  "source_text": "Bonjour, comment allez-vous?",
  "translated_text": "Salaam, naajaat?",
  "source_lang": "fr",
  "target_lang": "wo",
  "confidence": 0.85,
  "processing_time_ms": 245.32
}
```

### Endpoint: /batch-predict

Request:
```json
{
  "texts": ["Bonjour", "Au revoir"],
  "source_lang": "fr"
}
```

Response:
```json
{
  "results": [
    {
      "source_text": "Bonjour",
      "translated_text": "Salaam",
      "success": true
    },
    ...
  ],
  "total_texts": 2,
  "successful": 2,
  "total_time_ms": 412.45
}
```

### Endpoint: /health

Vérification santé du service:
```bash
curl http://localhost:5000/health
```

### Endpoint: /model-info

Informations du modèle:
```bash
curl http://localhost:5000/model-info
```

## Architecture technique

### app.py - API Flask
- Routes /health, /model-info, /predict, /batch-predict
- CORS support
- Gestion erreurs et validation
- Init modèle lazy au premier appel

### inference.py - Inférence
Deux classes:

**LanguageDetector**
- Détecte langue (fr/wo) basé sur mots clés
- Support auto-détection si source_lang=null

**TranslationInference**
- Chargement modèle NLLB base
- Chargement adapter LoRA si présent
- Tokenisation + beam search
- Gestion tokens langue NLLB (fra_Latn, wol_Latn)

## Gestion du modèle

### Chargement
- Model base: facebook/nllb-200-distilled-600M (auto-downloaded HF)
- LoRA adapter: depuis répertoire local ./model/
- Device: auto-détection GPU/CPU

### Inférence
- Tokenisation: max_length=128
- Generation: beam_size=1-8, temperature=0
- Language tokens: forcés au début via forced_bos_token_id

## Performance

### Entraînement
- Epochs: 4
- Train time: ~6h40m (P100)
- Batch size: 12 (train), 12 (eval)
- Learning rate: 1.2e-4

### Inférence (single request)
- Latence moyenne: 150-200ms (GPU)
- Memory: ~4GB GPU, ~2GB RAM
- Throughput: ~50 req/min

### Inférence (batch)
- Batch de 10: 800-1200ms total
- Linear scaling: O(n) temps

## Monitoring

### Logs
```bash
docker-compose logs -f translation-api
```

Logs structure:
```
2024-01-20 15:30:00,123 - app - INFO - Prediction - Text: Bonjour...
2024-01-20 15:30:00,245 - app - INFO - ✓ Prediction reussie - 122.45ms
```

### Health check
```bash
docker-compose ps  # Check container status
curl http://localhost:5000/health  # Check API
```

## Troubleshooting

### "CUDA not available"
Vérifier nvidia-smi et Dockerfile image

### "Model path not found"
```bash
# Vérifier structure ./model/
ls -la ./model/
# Doit avoir: adapter_config.json, adapter_model.bin, tokenizer.model

# Recréer:
python setup_model.py --source kaggle --kaggle-path ...
```

### "Out of memory"
Réduire workers/threads dans docker-compose ou Dockerfile

### Port 5000 utilisé
Changer dans docker-compose.yml: ports: ["8000:5000"]

## Cas d'usage

### Single request Python
```python
import requests

result = requests.post(
    "http://localhost:5000/predict",
    json={"text": "Bonjour", "source_lang": "fr"}
).json()
print(result["translated_text"])
```

### Batch processing
```python
texts = ["Bonjour", "Au revoir", "Merci"]
result = requests.post(
    "http://localhost:5000/batch-predict",
    json={"texts": texts, "source_lang": "fr"}
).json()

for item in result["results"]:
    print(f"{item['source_text']} → {item['translated_text']}")
```

### Auto-détection langue
```python
# Pas besoin de source_lang - détection auto
result = requests.post(
    "http://localhost:5000/predict",
    json={"text": "At yu bari la def."}  # Wolof
).json()
# Détecte automatiquement "wo" → "fr"
```

## Production Checklist

- [ ] Tester tous endpoints
- [ ] Monitoring & alerting
- [ ] Rate limiting
- [ ] API keys/authentication
- [ ] Reverse proxy (nginx)
- [ ] Caching (Redis optionnel)
- [ ] Auto-scaling (k8s optionnel)
- [ ] Backups modèle
- [ ] Documentation API
- [ ] SLA monitoring

## Modifications clés vs notebook original

1. **Modulaire**: Séparation entraînement/inférence/api
2. **MLflow**: Logging complet (params, metrics, models)
3. **LoRA**: Fine-tuning léger en production
4. **Auto-détection**: LanguageDetector intégré
5. **Batch**: Support traitement multiple
6. **Docker**: Conteneurisation production-ready
7. **Error handling**: Gestion robuste des erreurs
8. **Logging**: Traces structurées

## Références

- NLLB: https://arxiv.org/abs/2207.04672
- Transformers: https://huggingface.co/docs/transformers
- PEFT/LoRA: https://huggingface.co/docs/peft
- MLflow: https://mlflow.org
- Flask: https://flask.palletsprojects.com
- Docker: https://docs.docker.com

---

Créé février 2026
