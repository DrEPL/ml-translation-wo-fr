#!/usr/bin/env python3
"""
Script de configuration: Télécharge le modèle entraîné depuis Kaggle/MLflow
et le place dans le répertoire local pour le déploiement Docker.

Usage:
    python setup_model.py --source kaggle --kaggle-path <path> --output ./model
    python setup_model.py --source mlflow --mlflow-uri <uri> --output ./model
"""

import os
import sys
import argparse
import shutil
import logging
from pathlib import Path
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_from_kaggle(source_path: str, output_dir: str):
    """
    Copie le modèle depuis un chemin local Kaggle
    
    Expected structure:
        source_path/
        ├── adapter_config.json
        ├── adapter_model.bin
        ├── tokenizer.model
        ├── tokenizer_config.json
        └── special_tokens_map.json
    """
    logger.info(f"Configuration depuis Kaggle: {source_path}")
    
    source_path = Path(source_path)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Fichiers obligatoires pour LoRA
    required_files = [
        "adapter_config.json",
        "adapter_model.bin",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    for file_name in required_files:
        src_file = source_path / file_name
        dst_file = output_path / file_name
        
        if src_file.exists():
            logger.info(f"Copying {file_name}...")
            shutil.copy2(src_file, dst_file)
        else:
            logger.warning(f"File not found: {file_name}")
    
    logger.info(f"✓ Modèle copié dans {output_path}")

def setup_from_mlflow(mlflow_uri: str, model_name: str, version: str, output_dir: str):
    """
    Télécharge le modèle depuis MLflow Model Registry
    
    Example:
        mlflow_uri: "http://mlflow.example.com"
        model_name: "nllb-fr-wo-translation"
        version: "1" ou "production"
    """
    logger.info(f"Configuration depuis MLflow: {mlflow_uri}/{model_name}/{version}")
    
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Télécharger le modèle
        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Downloading model from: {model_uri}")
        
        local_path = mlflow.pyfunc.load_model(model_uri)
        
        # Copier dans le répertoire output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copier les artefacts
        for item in Path(local_path).iterdir():
            if item.is_file():
                shutil.copy2(item, output_path / item.name)
                logger.info(f"Copied {item.name}")
        
        logger.info(f"✓ Modèle téléchargé depuis MLflow dans {output_path}")
        
    except ImportError:
        logger.error("MLflow not installed. Install with: pip install mlflow")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error downloading from MLflow: {e}")
        sys.exit(1)

def create_config(output_dir: str):
    """Crée un fichier de configuration pour le modèle"""
    config = {
        "model_name": "nllb-200-distilled-600M",
        "task": "translation",
        "source_languages": ["fr", "wo"],
        "target_languages": ["fr", "wo"],
        "adapter_type": "lora",
        "loaded_at": __import__('datetime').datetime.utcnow().isoformat()
    }
    
    config_path = Path(output_dir) / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✓ Configuration sauvegardée dans {config_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Setup model for deployment"
    )
    parser.add_argument(
        "--source",
        choices=["kaggle", "mlflow"],
        default="kaggle",
        help="Source du modèle"
    )
    parser.add_argument(
        "--kaggle-path",
        type=str,
        default="/kaggle/working/nllb-fr-wo-lora",
        help="Chemin local Kaggle du modèle"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://localhost:5000",
        help="URI du serveur MLflow"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nllb-fr-wo-translation",
        help="Nom du modèle dans MLflow"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="production",
        help="Version du modèle (ex: '1', 'production')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./model",
        help="Répertoire de sortie"
    )
    
    args = parser.parse_args()
    
    try:
        if args.source == "kaggle":
            setup_from_kaggle(args.kaggle_path, args.output)
        elif args.source == "mlflow":
            setup_from_mlflow(
                args.mlflow_uri,
                args.model_name,
                args.model_version,
                args.output
            )
        
        create_config(args.output)
        logger.info("✓ Configuration terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()