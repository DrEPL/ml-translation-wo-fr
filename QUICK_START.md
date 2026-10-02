# 🚀 QUICK START GUIDE

Démarrer rapidement avec le système complet de traduction FR↔WO.

## 5 MIN: Déploiement local

### Prérequis
- Docker & docker-compose installés
- Port 5000 disponible
- ~8GB RAM libre

### Étapes

1. **Préparation du modèle** (depuis Kaggle)
```bash
# Télécharger depuis Kaggle:
# /kaggle/working/nllb-fr-wo-lora/ → dossier local

# Copier dans le répertoire de déploiement
python setup_model.py \
    --source kaggle \
    --kaggle-path ./nllb-fr-wo-lora \
    --output ./model
```

2. **Lancer Docker**
```bash
docker-compose up
```

gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 2 --worker-class gthread --timeout 120 --access-logfile - --error-logfile - app:app

1. **Tester l'API**
```bash
curl http://localhost:5000/health

# Réponse:
# {"status": "healthy", "model_loaded": true, ...}
```

## 2 MIN: Utilisation basique

### Traduction simple
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bonjour",
    "source_lang": "fr"
  }'

# Réponse:
# {
#   "translated_text": "Salaam",
#   "source_lang": "fr",
#   "target_lang": "wo",
#   "processing_time_ms": 245.32
# }
```

### Traduction batch
```bash
curl -X POST http://localhost:5000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Bonjour", "Au revoir"],
    "source_lang": "fr"
  }'
```

## Python: Simple utilisation

```python
import requests

# Single
result = requests.post(
    "http://localhost:5000/predict",
    json={"text": "Bonjour", "source_lang": "fr"}
).json()
print(result["translated_text"])

# Batch
result = requests.post(
    "http://localhost:5000/batch-predict",
    json={
        "texts": ["Bonjour", "Au revoir"],
        "source_lang": "fr"
    }
).json()
for item in result["results"]:
    print(f"{item['source_text']} → {item['translated_text']}")
```

## Entraînement (Kaggle)

1. Copier `kaggle_training.ipynb` dans un Kaggle Notebook
2. Sélectionner GPU (P100 ou T4)
3. Exécuter toutes les cellules
4. Résultats dans `/kaggle/working/nllb-fr-wo-lora/`

## Endpoints

| Route | Méthode | Description |
|-------|---------|-------------|
| `/health` | GET | Vérifier santé API |
| `/model-info` | GET | Infos modèle |
| `/predict` | POST | Traduire un texte |
| `/batch-predict` | POST | Traduire multiple |

## Paramètres /predict

```json
{
  "text": "string",                    // REQUIS
  "source_lang": "fr|wo",             // OPTIONNEL (auto-detect)
  "target_lang": "fr|wo",             // OPTIONNEL (inverse auto)
  "num_beams": 1-8                    // OPTIONNEL (défaut: 4)
}
```

## Troubleshooting rapide

**Port 5000 utilisé:**
```bash
# docker-compose.yml: ports: ["8000:5000"]
```

**Model not found:**
```bash
# Vérifier: ls ./model/
# Doit avoir: adapter_config.json, adapter_model.bin, tokenizer.model
```

**GPU not available:**
```bash
# nvidia-smi
# Vérifier Dockerfile image appropriée
```

**Slow inference:**
```bash
# Normal au premier appel (~500ms)
# Requêtes suivantes: 100-250ms
```

## Performance

| Métrique | Valeur |
|----------|--------|
| BLEU Score | 18.89 |
| chrF Score | 40.10 |
| Latence (ms) | 100-250 |
| Memory GPU | 4GB |
| Memory RAM | 2GB |

## Exemples complets

Voir `examples.py`:
```bash
python examples.py
```

## Documentation complète

- `README.md` : Architecture & guide complet
- `MODIFICATIONS.md` : Détails des changements
- `kaggle_training.ipynb` : Code entraînement
- `app.py` : API Flask
- `inference.py` : Moteur inférence

## Fichiers importants

```
├── kaggle_training.ipynb       # Entraînement
├── app.py                      # API Flask
├── inference.py                # Inférence
├── requirements.txt            # Dépendances
├── Dockerfile                  # Conteneur
├── docker-compose.yml          # Orchestration
├── setup_model.py              # Setup modèle
├── examples.py                 # Exemples
└── README.md                   # Complet
```

## Next steps

1. ✓ Déployer localement (ci-dessus)
2. Test avec exemples.py
3. Intégrer dans votre app
4. (Optionnel) Déployer en production
5. (Optionnel) Ajouter monitoring/CI-CD

---

**Besoin d'aide?**
- Vérifier logs: `docker-compose logs -f`
- Voir TROUBLESHOOTING dans README.md
- Consulter docstrings du code
