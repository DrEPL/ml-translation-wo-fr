"""
Module d'inference pour la traduction FR <-> WO

Responsabilites:
- Charger le modèle depuis un chemin local ou MLflow
- Gerer les deux directions FR -> WO et WO -> FR
- Auto-detecter la langue source si nécessaire
- Faire l'inférence avec paramètres configurables
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import logging
import os
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Simple detecteur de langue base sur des patterns/mots cles"""
    
    # Mots courants francais
    FRENCH_KEYWORDS = {
        "le", "la", "les", "de", "du", "et", "est", "un", "une", "des",
        "au", "aux", "ce", "cette", "qui", "que", "se", "dans", "pour",
        "avec", "sans", "par", "sur", "sous", "entre", "apres", "avant"
    }
    
    # Mots courants wolof
    WOLOF_KEYWORDS = {
        "bu", "ba", "bi", "be", "ko", "mu", "ju", "su", "nu", "ni",
        "n", "a", "u", "i", "o", "e", "ak", "laa", "wax", "jox", "am"
    }
    
    @staticmethod
    def detect(text: str) -> str:
        """
        Detecte la langue (fr ou wo)
        Retour: 'fr', 'wo', ou 'unknown' si ambigu
        """
        text_lower = text.lower()
        words = set(text_lower.split()[:20])  # Premiers 20 mots
        
        french_matches = len(words & LanguageDetector.FRENCH_KEYWORDS)
        wolof_matches = len(words & LanguageDetector.WOLOF_KEYWORDS)
        
        if french_matches > wolof_matches:
            return "fr"
        elif wolof_matches > french_matches:
            return "wo"
        else:
            # Par defaut francais (langue plus courante)
            return "fr"

class TranslationInference:
    """
    Classe pour l'inférence de traduction avec support LoRA
    """
    
    # Codes langue NLLB
    LANG_CODES = {
        "fr": "fra_Latn",
        "wo": "wol_Latn"
    }
    
    def __init__(
        self,
        model_path: str,
        mlflow_tracking_uri: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialise le moteur d'inférence
        
        Args:
            model_path: Chemin vers le modèle (local ou MLflow)
            mlflow_tracking_uri: URI MLflow (optionnel)
            device: 'cuda' ou 'cpu' (auto-detecte si None)
        """
        self.model_path = model_path
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initialisation TranslationInference")
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  MLflow URI: {mlflow_tracking_uri}")
        
        self.model = None
        self.tokenizer = None
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Charge le modèle et le tokenizer"""
        
        # Determiner le chemin base (où se trouve le modèle non fine-tune)
        base_model_name = "facebook/nllb-200-distilled-600M"
        
        logger.info(f"Chargement tokenizer depuis {base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        # logger.info(f"Chargement modèle base depuis {base_model_name}...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Si des fichiers LoRA existent, les charger
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        
        if os.path.exists(adapter_config_path):
            logger.info(f"Chargement adapteur LoRA depuis {self.model_path}...")
            try:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.model_path,
                    device_map="auto" if self.device == "cuda" else None
                )
                logger.info("✓ Adapteur LoRA charge")
            except Exception as e:
                logger.warning(f"Impossible de charger l'adapteur LoRA: {e}")
                logger.info("Continuant avec le modèle base...")
        
        # Passer au device
        if self.device == "cpu":
            if self.model is not None:
                self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("✓ Modèle et tokenizer charges avec succes")
    
    def detect_language(self, text: str) -> str:
        """Auto-detecte la langue du texte (fr ou wo)"""
        detected = LanguageDetector.detect(text)
        return detected
    
    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        num_beams: int = 4,
        max_length: int = 128,
        min_length: int = 20
    ) -> Dict[str, any]:
        """
        Traduit un texte
        
        Args:
            text: Texte a traduire
            source_lang: Langue source ('fr' ou 'wo'). Auto-detectee si None
            target_lang: Langue cible. Si None, inverse de source
            num_beams: Nombre de beams pour la recherche (1-8)
            max_length: Longueur max de la traduction
            min_length: Longueur min de la traduction
        
        Returns:
            Dict avec keys: translation, source_lang, target_lang, confidence
        """
        
        # Validation input
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Le texte ne peut pas etre vide")
        
        if len(text) > 1000:
            logger.warning(f"Texte tres long ({len(text)} chars), tronque a 1000")
            text = text[:1000]
        
        # Auto-detection langue source
        if source_lang is None:
            source_lang = self.detect_language(text)
            logger.info(f"Langue source auto-detectee: {source_lang}")
        else:
            source_lang = source_lang.lower()
        
        # Validation langue source
        if source_lang not in ["fr", "wo"]:
            raise ValueError(f"Langue source invalide: {source_lang}. Doit etre 'fr' ou 'wo'")
        
        # Determiner langue cible
        if target_lang is None:
            target_lang = "wo" if source_lang == "fr" else "fr"
        else:
            target_lang = target_lang.lower()
        
        # Validation langue cible
        if target_lang not in ["fr", "wo"]:
            raise ValueError(f"Langue cible invalide: {target_lang}. Doit etre 'fr' ou 'wo'")
        
        if source_lang == target_lang:
            logger.warning("Source et cible identiques, retour du texte original")
            return {
                "translation": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "confidence": 1.0
            }
        
        logger.info(f"Traduction {source_lang.upper()} -> {target_lang.upper()} ({len(text)} chars)")
        
        try:
            # Preparation inputs
            source_code = self.LANG_CODES[source_lang]
            target_code = self.LANG_CODES[target_lang]
            
            # Tokenization
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            # Deplacer sur device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generation
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_code),
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False,
                    temperature=None,
                    top_p=None
                )
            
            # Decodage
            translation = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            logger.info(f"✓ Traduction completee: {translation[:100]}...")
            
            return {
                "translation": translation,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "confidence": 0.85
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la traduction: {e}", exc_info=True)
            raise RuntimeError(f"Erreur traduction: {str(e)}")
