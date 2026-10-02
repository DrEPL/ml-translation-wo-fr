#!/usr/bin/env python3
"""
Script d'inférence - Compatible avec adapter_model.safetensors

Utilise: facebook/nllb-200-distilled-600M + LoRA adapter
Support: FR ↔ WO bidirectionnel avec auto-détection langue
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationModel:
    def __init__(self, model_dir: str = "./nllb-fr-wo-lora", device: str = None):
        """
        Initialise le modèle NLLB + LoRA adapter
        
        Args:
            model_dir: Chemin vers le dossier contenant adapter_config.json et adapter_model.safetensors
            device: 'cuda' ou 'cpu' (auto-détecté si None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Utilisation du device: {self.device}")
        
        # 1. Charger le modèle base et tokenizer
        logger.info("Chargement du modèle NLLB-200-distilled-600M...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            trust_remote_code=True
        )
        
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(
        #     "facebook/nllb-200-distilled-600M",
        #     device_map="auto" if self.device == "cuda" else None,
        #     torch_dtype=torch.float32
        # )
        
        # 2. Charger l'adapter LoRA
        logger.info(f"Chargement de l'adapter LoRA depuis {model_dir}...")
        self.model = PeftModel.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            model_dir,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Mettre en mode inference
        self.model.eval()
        logger.info("✓ Modèle + adapter LoRA chargés avec succès")
        
        # Codes langue NLLB
        self.LANG_CODES = {
            "fr": "fra_Latn",
            "wo": "wol_Latn"
        }
    
    def detect_language(self, text: str) -> str:
        """Détection simple de langue basée sur mots clés"""
        french_keywords = {"le", "la", "les", "de", "du", "et", "est", "un", "une"}
        wolof_keywords = {"bu", "ba", "bi", "be", "ko", "mu", "laa", "wax"}
        
        words = set(text.lower().split()[:15])
        
        fr_score = len(words & french_keywords)
        wo_score = len(words & wolof_keywords)
        
        return "fr" if fr_score >= wo_score else "wo"
    
    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        num_beams: int = 4
    ) -> Dict[str, str]:
        """
        Traduit un texte FR ↔ WO
        
        Args:
            text: Texte à traduire
            source_lang: 'fr' ou 'wo' (auto-détecté si None)
            target_lang: 'fr' ou 'wo' (inverse de source si None)
            num_beams: Nombre de beams pour la génération
        
        Returns:
            {
                "source": "Bonjour",
                "target": "Salaam",
                "source_lang": "fr",
                "target_lang": "wo"
            }
        """
        # Auto-détection langue source
        if source_lang is None:
            source_lang = self.detect_language(text)
            logger.info(f"Langue source détectée: {source_lang}")
        else:
            source_lang = source_lang.lower()
        
        # Langue cible par défaut
        if target_lang is None:
            target_lang = "wo" if source_lang == "fr" else "fr"
        else:
            target_lang = target_lang.lower()
        
        # Codes langue NLLB
        target_code = self.LANG_CODES[target_lang]
        
        # Tokenisation
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Déplacer sur device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Génération avec forced_bos_token_id
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_code),
                max_length=128,
                num_beams=num_beams,
                early_stopping=True,
                temperature=0.7
            )
        
        # Décodage
        translation = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return {
            "source": text,
            "target": translation,
            "source_lang": source_lang,
            "target_lang": target_lang
        }


# ============================================================================
# EXEMPLES D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    # Initialiser le modèle
    model = TranslationModel(model_dir="./nllb-fr-wo-lora")
    
    # Exemples
    print("\n" + "="*70)
    print("EXEMPLES DE TRADUCTION")
    print("="*70)
    
    # 1. FR → WO (explicite)
    print("\n1. Français → Wolof")
    result = model.translate("Bonjour, comment allez-vous?", source_lang="fr", target_lang="wo")
    print(f"   {result['source']} → {result['target']}")
    
    # 2. WO → FR (explicite)
    print("\n2. Wolof → Français")
    result = model.translate("Salaam alekum", source_lang="wo")
    print(f"   {result['source']} → {result['target']}")
    
    # 3. Auto-détection
    print("\n3. Auto-détection (texte français)")
    result = model.translate("Enchanté de vous rencontrer")
    print(f"   Détecté: {result['source_lang']}")
    print(f"   {result['source']} → {result['target']}")
    
    # 4. Batch simple
    print("\n4. Batch (plusieurs textes)")
    texts = ["Bonjour", "Au revoir", "Merci", "Excusez-moi"]
    for text in texts:
        result = model.translate(text, source_lang="fr")
        print(f"   {text:20} → {result['target']}")