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
    # Chargement depuis MLflow
    # ------------------------------------------------------------------

    def _load_from_mlflow_registry(self) -> bool:
        """
        Charge le modèle et le tokenizer directement via mlflow.transformers.load_model().

        C'est la méthode recommandée quand le modèle a été loggé avec
        mlflow.transformers.log_model(). Elle gère automatiquement la
        structure interne des artefacts MLflow (sous-dossier model/, etc.).

        Retourne True si le chargement a réussi, False sinon.
        """
        try:
            import mlflow
            import mlflow.transformers

            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                logger.info(f"MLflow tracking URI défini : {self.mlflow_tracking_uri}")

            logger.info(
                f"Chargement du modèle via mlflow.transformers.load_model "
                f"({self.mlflow_model_uri}) ..."
            )

            components = mlflow.transformers.load_model(
                self.mlflow_model_uri,
                return_type="components",
            )

            # components est un dict : {"model": ..., "tokenizer": ..., ...}
            self.model = components["model"]
            self.tokenizer = components["tokenizer"]

            logger.info("✓ Modèle et tokenizer chargés via MLflow transformers")
            return True

        except Exception as exc:
            logger.warning(
                f"mlflow.transformers.load_model a échoué : {exc}. "
                f"Bascule sur le téléchargement d'artefacts bruts.",
                exc_info=True,
            )
            return False

    def _download_mlflow_artifacts(self) -> Optional[str]:
        """
        Télécharge les artefacts bruts et résout le dossier contenant config.json.

        Retourne le chemin local vers le dossier du modèle HuggingFace, ou None.
        """
        try:
            import mlflow

            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)

            logger.info(f"Téléchargement des artefacts depuis MLflow : {self.mlflow_model_uri}")
            local_path = mlflow.artifacts.download_artifacts(self.mlflow_model_uri)
            logger.info(f"Artefacts téléchargés dans : {local_path}")

            # Chercher le dossier qui contient réellement config.json
            # Structure typique mlflow.transformers.log_model() :
            #   <root>/model/config.json
            # Structure typique mlflow.log_artifacts() :
            #   <root>/config.json  ou  <root>/lora_adapters/adapter_config.json
            resolved = self._find_model_dir(local_path)
            if resolved:
                logger.info(f"Dossier modèle résolu : {resolved}")
                return resolved

            logger.warning(
                f"Aucun config.json ou adapter_config.json trouvé dans {local_path}"
            )
            return None

        except Exception as exc:
            logger.error(f"Échec du téléchargement depuis MLflow : {exc}", exc_info=True)
            return None

    @staticmethod
    def _find_model_dir(root: str) -> Optional[str]:
        """
        Recherche récursive du dossier contenant config.json ou adapter_config.json
        dans l'arborescence d'artefacts MLflow.

        Ordre de priorité :
        1. Sous-dossiers connus : model/, lora_adapters/, nllb-merged-model/, merged/
        2. Racine elle-même
        3. Recherche récursive dans tous les sous-dossiers
        """
        root_path = Path(root)

        # 1. Vérifier les sous-dossiers connus en premier
        known_subdirs = ["model", "lora_adapters", "nllb-merged-model", "merged"]
        for sub in known_subdirs:
            candidate = root_path / sub
            if (candidate / "config.json").exists() or (candidate / "adapter_config.json").exists():
                return str(candidate)

        # 2. Vérifier la racine
        if (root_path / "config.json").exists() or (root_path / "adapter_config.json").exists():
            return root

        # 3. Recherche récursive
        for config_name in ["config.json", "adapter_config.json"]:
            for p in root_path.rglob(config_name):
                return str(p.parent)

        return None

    # ------------------------------------------------------------------
    # Chargement du modèle et du tokenizer
    # ------------------------------------------------------------------

    def _load_model_and_tokenizer(self):
        """
        Charge le tokenizer et le modèle.

        Stratégie (dans l'ordre) :
        1. mlflow.transformers.load_model()  — méthode recommandée
        2. Téléchargement d'artefacts MLflow + chargement HuggingFace manuel
        3. Chemin local (MODEL_PATH)
        4. Modèle de base NLLB uniquement (sans fine-tuning)
        """

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        device_map = "auto" if self.device == "cuda" else None

        # ============================================================
        # 1. MLflow transformers.load_model (méthode recommandée)
        # ============================================================
        if self.mlflow_model_uri:
            if self._load_from_mlflow_registry():
                # Déplacer le modèle sur le bon device
                if device_map is None:
                    self.model = self.model.to(self.device)
                self.model.eval()
                logger.info("✓ Modèle et tokenizer prêts (via MLflow load_model)")
                return

            # ========================================================
            # 2. Fallback : téléchargement d'artefacts + chargement manuel
            # ========================================================
            resolved_path = self._download_mlflow_artifacts()
            if resolved_path:
                self._load_from_local_dir(resolved_path, torch_dtype, device_map)
                return

            logger.warning("Toutes les méthodes MLflow ont échoué. Bascule sur le chemin local.")

        # ============================================================
        # 3. Chemin local
        # ============================================================
        if self.model_path and Path(self.model_path).exists():
            logger.info(f"Utilisation du chemin local : {self.model_path}")
            # Résoudre le bon sous-dossier si nécessaire
            resolved = self._find_model_dir(self.model_path) or self.model_path
            self._load_from_local_dir(resolved, torch_dtype, device_map)
            return

        # ============================================================
        # 4. Erreur (pas de fallback)
        # ============================================================
        raise RuntimeError(
            "Aucune source fine-tunée disponible (ni MLflow, ni locale). "
            "Le basculement sur le modèle de base a été désactivé à la demande de l'utilisateur."
        )

    def _load_from_local_dir(self, resolved_path: str, torch_dtype, device_map):
        """Charge le modèle et le tokenizer depuis un dossier local résolu."""

        # ---- Tokenizer ----
        tokenizer_path = resolved_path
        
        # Structure MLflow: si le modèle est dans <root>/model, le tokenizer est souvent dans <root>/components/tokenizer
        parent_dir = Path(resolved_path).parent
        if (parent_dir / "components" / "tokenizer").exists():
            tokenizer_path = str(parent_dir / "components" / "tokenizer")
        elif (Path(resolved_path) / "components" / "tokenizer").exists():
            tokenizer_path = str(Path(resolved_path) / "components" / "tokenizer")
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
            logger.info(f"Tokenizer chargé depuis : {tokenizer_path}")
        except Exception as exc:
            logger.error(
                f"Tokenizer non trouvé dans {tokenizer_path}. "
                f"Pas de bascule sur le modèle de base autorisée."
            )
            raise RuntimeError(f"Impossible de charger le tokenizer depuis {tokenizer_path}") from exc

        # ---- Modèle ----
        has_lora = (Path(resolved_path) / "adapter_config.json").exists()

        if has_lora:
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
                base, resolved_path, device_map=device_map
            )
            logger.info("✓ Adapters LoRA chargés avec succès")
        else:
            logger.info(f"Chargement modèle fusionné depuis : {resolved_path}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                resolved_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

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
