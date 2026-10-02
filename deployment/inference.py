"""
Module d'inference pour la traduction FR <-> WO

Responsabilites:
- Charger le modèle depuis le MLflow Model Registry (DagsHub) ou un chemin local (fallback)
- Gérer les deux directions FR -> WO et WO -> FR
- Auto-détecter la langue source si nécessaire
- Faire l'inférence avec paramètres configurables
"""

import os
import logging
from typing import Dict, Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Détecteur de langue
# ---------------------------------------------------------------------------

class LanguageDetector:
    """Détecteur simple de langue basé sur des mots-clés."""

    FRENCH_KEYWORDS = {
        "le", "la", "les", "de", "du", "et", "est", "un", "une", "des",
        "au", "aux", "ce", "cette", "qui", "que", "se", "dans", "pour",
        "avec", "sans", "par", "sur", "sous", "entre", "après", "avant",
        "bonjour", "merci", "oui", "non", "je", "tu", "il", "nous", "vous",
    }

    WOLOF_KEYWORDS = {
        "bu", "ba", "bi", "be", "ko", "mu", "ju", "su", "nu", "ni",
        "ak", "laa", "wax", "jox", "am", "nanga", "def", "jamm", "salaam",
        "waaw", "déedéet", "man", "yaw", "moom",
    }

    @staticmethod
    def detect(text: str) -> str:
        """Retourne 'fr' ou 'wo'. Par défaut 'fr' si ambigu."""
        text_lower = text.lower()
        words = set(text_lower.split()[:30])

        french_matches = len(words & LanguageDetector.FRENCH_KEYWORDS)
        wolof_matches = len(words & LanguageDetector.WOLOF_KEYWORDS)

        if french_matches > wolof_matches:
            return "fr"
        if wolof_matches > french_matches:
            return "wo"
        return "fr"  # défaut


# ---------------------------------------------------------------------------
# Moteur d'inférence
# ---------------------------------------------------------------------------

class TranslationInference:
    """
    Moteur d'inférence FR <-> WO.

    Stratégie de chargement (dans l'ordre) :
    1. MLflow Model Registry  → si MLFLOW_MODEL_URI est renseigné
       (ex. "models:/nllb-fr-wo-translation/latest")
    2. Chemin local           → MODEL_PATH (adapters LoRA ou modèle fusionné)
    3. Modèle de base NLLB    → facebook/nllb-200-distilled-600M (fallback)
    """

    LANG_CODES = {
        "fr": "fra_Latn",
        "wo": "wol_Latn",
    }

    BASE_MODEL_NAME = "facebook/nllb-200-distilled-600M"

    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model_name: str = BASE_MODEL_NAME,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_model_uri: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Chemin local vers les adapters LoRA ou le modèle fusionné.
                        Ignoré si mlflow_model_uri est fourni.
            base_model_name: Modèle NLLB de base (utilisé pour charger les adapters LoRA).
            mlflow_tracking_uri: URI du serveur MLflow.
                                 Ex: "https://dagshub.com/DrEPL/ml-translation-wo-fr.mlflow"
            mlflow_model_uri: URI du modèle dans le Registry MLflow.
                              Ex: "models:/nllb-fr-wo-translation/latest"
                              Si fourni, le modèle est téléchargé depuis MLflow.
            device: "cuda" | "cpu" | None (auto-détection).
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_model_uri = mlflow_model_uri
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None

        logger.info("=== Initialisation TranslationInference ===")
        logger.info(f"  device            : {self.device}")
        logger.info(f"  mlflow_model_uri  : {mlflow_model_uri or '(non fourni)'}")
        logger.info(f"  mlflow_uri        : {mlflow_tracking_uri or '(non fourni)'}")
        logger.info(f"  model_path        : {model_path or '(non fourni)'}")

        self._load_model_and_tokenizer()

    # ------------------------------------------------------------------
    # Résolution du chemin / téléchargement depuis MLflow
    # ------------------------------------------------------------------

    def _download_from_mlflow(self) -> str:
        """
        Télécharge les artefacts du modèle depuis le MLflow Model Registry.

        Retourne le chemin local vers les artefacts téléchargés.
        """
        import mlflow
        import mlflow.pyfunc

        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI défini : {self.mlflow_tracking_uri}")

        logger.info(f"Téléchargement du modèle depuis MLflow : {self.mlflow_model_uri}")

        # Téléchargement des artefacts bruts (dossier contenant les fichiers du modèle)
        local_path = mlflow.artifacts.download_artifacts(self.mlflow_model_uri)
        logger.info(f"Artefacts téléchargés dans : {local_path}")
        return local_path

    def _resolve_local_path(self) -> Optional[str]:
        """
        Résout le chemin local vers les artefacts.

        Priorité :
        1. MLflow Model Registry (si mlflow_model_uri renseigné)
        2. Chemin local model_path (si fourni et existant)
        3. None → chargement du modèle de base uniquement
        """
        # --- 1. MLflow Registry ---
        if self.mlflow_model_uri:
            try:
                return self._download_from_mlflow()
            except Exception as exc:
                logger.error(
                    f"Échec du téléchargement depuis MLflow : {exc}",
                    exc_info=True,
                )
                logger.warning("Bascule sur le chemin local si disponible.")

        # --- 2. Chemin local ---
        if self.model_path and Path(self.model_path).exists():
            logger.info(f"Utilisation du chemin local : {self.model_path}")
            return self.model_path

        logger.warning(
            "Ni MLflow Model URI ni chemin local valide. "
            "Chargement du modèle de base NLLB uniquement (sans fine-tuning)."
        )
        return None

    # ------------------------------------------------------------------
    # Chargement du modèle et du tokenizer
    # ------------------------------------------------------------------

    def _load_model_and_tokenizer(self):
        """Charge le tokenizer et le modèle (avec LoRA si disponible)."""

        resolved_path = self._resolve_local_path()

        # ---- Tokenizer ----
        # On essaie d'abord depuis le dossier résolu (inclut vocab NLLB custom)
        tokenizer_source = resolved_path if resolved_path else self.base_model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source, trust_remote_code=True
            )
            logger.info(f"Tokenizer chargé depuis : {tokenizer_source}")
        except Exception:
            logger.warning(
                f"Impossible de charger le tokenizer depuis {tokenizer_source}. "
                f"Bascule sur {self.base_model_name}."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name, trust_remote_code=True
            )
            logger.info(f"Tokenizer chargé depuis : {self.base_model_name}")

        # ---- Modèle ----
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        device_map = "auto" if self.device == "cuda" else None

        if resolved_path is None:
            # Aucun fine-tuning disponible → modèle de base uniquement
            logger.info(f"Chargement modèle de base : {self.base_model_name}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

        else:
            adapter_config = Path(resolved_path) / "adapter_config.json"

            # Chercher dans les sous-dossiers typiques si l'artefact MLflow
            # contient un sous-répertoire (ex. "lora_adapters/", "nllb-merged-model/")
            if not adapter_config.exists():
                for sub in ["lora_adapters", "nllb-merged-model", "merged"]:
                    candidate = Path(resolved_path) / sub
                    if (candidate / "adapter_config.json").exists():
                        resolved_path = str(candidate)
                        adapter_config = candidate / "adapter_config.json"
                        logger.info(f"Sous-dossier détecté : {resolved_path}")
                        break

            has_lora = adapter_config.exists()

            if has_lora:
                # Adapters LoRA → charger base + adapter
                logger.info(
                    f"Chargement modèle de base ({self.base_model_name}) "
                    f"+ adapters LoRA ({resolved_path})"
                )
                base = AutoModelForSeq2SeqLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                )
                self.model = PeftModel.from_pretrained(
                    base,
                    resolved_path,
                    device_map=device_map,
                )
                logger.info("✓ Adapters LoRA chargés avec succès")
            else:
                # Modèle fusionné
                logger.info(f"Chargement modèle fusionné depuis : {resolved_path}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    resolved_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                )

        # Passer sur le device si device_map n'a pas déjà géré ça
        if device_map is None:
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("✓ Modèle et tokenizer prêts")

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> str:
        """Auto-détecte la langue du texte ('fr' ou 'wo')."""
        return LanguageDetector.detect(text)

    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        num_beams: int = 4,
        max_length: int = 128,
        min_length: int = 1,
    ) -> Dict:
        """
        Traduit un texte FR <-> WO.

        Args:
            text: Texte à traduire (max 1000 caractères).
            source_lang: 'fr' ou 'wo'. Auto-détecté si None.
            target_lang: 'fr' ou 'wo'. Inverse de source si None.
            num_beams: Nombre de beams (beam search). Plage conseillée : 1–8.
            max_length: Longueur maximale de la séquence générée.
            min_length: Longueur minimale de la séquence générée.

        Returns:
            Dict avec les clés : translation, source_lang, target_lang, confidence.
        """
        # Validation
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Le texte ne peut pas être vide.")

        text = text.strip()
        if len(text) > 1000:
            logger.warning(f"Texte tronqué à 1000 caractères (longueur originale : {len(text)})")
            text = text[:1000]

        # Langue source
        if source_lang is None:
            source_lang = self.detect_language(text)
            logger.info(f"Langue source auto-détectée : {source_lang}")
        source_lang = source_lang.lower()

        if source_lang not in self.LANG_CODES:
            raise ValueError(
                f"Langue source invalide : '{source_lang}'. Valeurs acceptées : {list(self.LANG_CODES)}"
            )

        # Langue cible
        if target_lang is None:
            target_lang = "wo" if source_lang == "fr" else "fr"
        target_lang = target_lang.lower()

        if target_lang not in self.LANG_CODES:
            raise ValueError(
                f"Langue cible invalide : '{target_lang}'. Valeurs acceptées : {list(self.LANG_CODES)}"
            )

        # Même langue → retour direct
        if source_lang == target_lang:
            logger.warning("Langues source et cible identiques — retour du texte original.")
            return {
                "translation": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "confidence": 1.0,
            }

        logger.info(
            f"Traduction {source_lang.upper()} → {target_lang.upper()} ({len(text)} chars)"
        )

        source_code = self.LANG_CODES[source_lang]
        target_code = self.LANG_CODES[target_lang]

        # Important pour NLLB : définir la langue source sur le tokenizer
        self.tokenizer.src_lang = source_code

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_code)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,
            )

        translation = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        logger.info(f"✓ Traduction : {translation[:120]}{'...' if len(translation) > 120 else ''}")

        return {
            "translation": translation,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "confidence": 0.85,  # placeholder — score réel non disponible via generate()
        }
