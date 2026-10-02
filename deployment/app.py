"""
API Flask pour l'inférence - Traduction FR ↔ WO avec NLLB-200-distilled-600M

Architecture:
- Charge le meilleur modèle depuis MLflow Model Registry
- Expose un endpoint /predict pour l'inférence
- Gère les deux directions de traduction automatiquement
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import logging
import os
from datetime import datetime
from typing import Dict, Any
import json

from inference import TranslationInference

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialiser Flask
app = Flask(__name__)
CORS(app)

# Variables globales
inference_engine = None
model_info = {}

@app.before_request
def before_request():
    """Initialiser le modèle au premier appel"""
    global inference_engine, model_info
    
    if inference_engine is None:
        logger.info("Initialisation du modèle d'inférence...")
        try:
            inference_engine = TranslationInference(
                model_path=os.getenv("MODEL_PATH", "/app/model"),
                mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", None)
            )
            model_info = {
                "model_name": "nllb-200-distilled-600M",
                "task": "translation_fr_wo",
                "framework": "transformers",
                "adapter": "lora",
                "initialized_at": datetime.utcnow().isoformat(),
                "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                "cuda_available": torch.cuda.is_available()
            }
            logger.info("✓ Modèle initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur initialisation modèle: {e}", exc_info=True)
            raise

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de santé"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": inference_engine is not None,
        "cuda_available": torch.cuda.is_available()
    }), 200

@app.route("/model-info", methods=["GET"])
def get_model_info():
    """Retourne les informations du modèle"""
    return jsonify({
        "info": model_info,
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint de prediction pour la traduction
    
    JSON attendu:
    {
        "text": "string",           # Texte a traduire (requis)
        "source_lang": "fr|wo",     # Langue source (optionnel, auto-detectee)
        "target_lang": "fr|wo",     # Langue cible (optionnel)
        "num_beams": 4              # Nombre de beams (optionnel)
    }
    
    Reponse:
    {
        "source_text": "string",
        "translated_text": "string",
        "source_lang": "string",
        "target_lang": "string",
        "confidence": float,
        "processing_time_ms": float,
        "model_info": {...}
    }
    """
    try:
        # Validation de la requete
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if "text" not in data or not data["text"].strip():
            return jsonify({"error": "Field 'text' is required and cannot be empty"}), 400
        
        text = data["text"].strip()
        source_lang = data.get("source_lang", None)  # Auto-detection si None
        target_lang = data.get("target_lang", None)  # Inverse de source si None
        num_beams = data.get("num_beams", 4)
        
        # Validation
        if len(text) > 1000:
            return jsonify({"error": "Text exceeds max length of 1000 characters"}), 400
        
        if not isinstance(num_beams, int) or num_beams < 1 or num_beams > 8:
            num_beams = 4
        
        logger.info(f"Prediction - Text: {text[:100]}... | Source: {source_lang} | Target: {target_lang}")
        
        # Appel inference
        start_time = datetime.utcnow()
        result = inference_engine.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            num_beams=num_beams
        )
        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Preparation reponse
        response = {
            "source_text": text,
            "translated_text": result["translation"],
            "source_lang": result["source_lang"],
            "target_lang": result["target_lang"],
            "confidence": result.get("confidence", 0.0),
            "processing_time_ms": round(elapsed, 2),
            "model_info": model_info
        }
        
        logger.info(f"✓ Prediction reussie - {elapsed:.2f}ms")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Erreur dans /predict: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """
    Endpoint batch pour traiter plusieurs textes
    
    JSON attendu:
    {
        "texts": ["string1", "string2", ...],
        "source_lang": "fr|wo",
        "target_lang": "fr|wo"
    }
    
    Reponse:
    {
        "results": [...],
        "total_time_ms": float
    }
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if "texts" not in data or not isinstance(data["texts"], list):
            return jsonify({"error": "Field 'texts' must be a list"}), 400
        
        texts = [str(t).strip() for t in data["texts"] if str(t).strip()]
        
        if not texts:
            return jsonify({"error": "No valid texts provided"}), 400
        
        if len(texts) > 100:
            return jsonify({"error": "Maximum 100 texts per batch"}), 400
        
        source_lang = data.get("source_lang", None)
        target_lang = data.get("target_lang", None)
        num_beams = data.get("num_beams", 4)
        
        logger.info(f"Batch prediction - {len(texts)} textes")
        
        start_time = datetime.utcnow()
        results = []
        
        for text in texts:
            try:
                result = inference_engine.translate(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    num_beams=num_beams
                )
                results.append({
                    "source_text": text,
                    "translated_text": result["translation"],
                    "source_lang": result["source_lang"],
                    "target_lang": result["target_lang"],
                    "success": True
                })
            except Exception as e:
                logger.warning(f"Erreur pour texte '{text[:50]}...': {e}")
                results.append({
                    "source_text": text,
                    "translated_text": None,
                    "error": str(e),
                    "success": False
                })
        
        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return jsonify({
            "results": results,
            "total_texts": len(texts),
            "successful": sum(1 for r in results if r["success"]),
            "total_time_ms": round(elapsed, 2)
        }), 200
        
    except Exception as e:
        logger.error(f"Erreur dans /batch-predict: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Configuration
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    
    logger.info(f"Demarrage API Flask - Host: {host}, Port: {port}, Debug: {debug_mode}")
    logger.info(f"CUDA disponible: {torch.cuda.is_available()}")
    
    app.run(host=host, port=port, debug=debug_mode, threaded=True)
