#!/usr/bin/env python3
"""
Exemples d'utilisation de l'API de traduction

Exécuter depuis votre machine locale (après docker-compose up):
    python examples.py
"""

import requests
import json
import time
from typing import Dict, List
import urllib3

# Désactiver warnings SSL si nécessaire
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
API_URL = "http://localhost:5000"
TIMEOUT = 30

# ============================================================================
# 1. VÉRIFICATION SANTÉ
# ============================================================================

def example_health_check():
    """Vérifier que l'API est disponible"""
    print("\n" + "="*70)
    print("1. HEALTH CHECK")
    print("="*70)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        print(f"Status: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("\n✓ API est healthy!")
        else:
            print("\n✗ API non disponible")
            
    except requests.exceptions.ConnectionError:
        print("✗ Impossible de se connecter à l'API")
        print(f"  Vérifier que docker-compose est lancé sur {API_URL}")

# ============================================================================
# 2. INFORMATIONS DU MODÈLE
# ============================================================================

def example_model_info():
    """Récupérer les informations du modèle"""
    print("\n" + "="*70)
    print("2. MODEL INFO")
    print("="*70)
    
    response = requests.get(f"{API_URL}/model-info", timeout=TIMEOUT)
    data = response.json()
    
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(data, indent=2)}")

# ============================================================================
# 3. TRADUCTION SIMPLE
# ============================================================================

def example_single_translation():
    """Traduire un texte simple"""
    print("\n" + "="*70)
    print("3. TRADUCTION SIMPLE")
    print("="*70)
    
    # Exemple 1: FR → WO (explicite)
    print("\n--- Exemple 1: Français → Wolof ---")
    request_data = {
        "text": "Bonjour, comment allez-vous?",
        "source_lang": "fr",
        "target_lang": "wo"
    }
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    response = requests.post(
        f"{API_URL}/predict",
        json=request_data,
        timeout=TIMEOUT
    )
    print(f"\nStatus: {response.status_code}")
    result = response.json()
    print(f"Response:\n{json.dumps(result, indent=2)}")
    print(f"\n→ Traduction: {result['translated_text']}")
    print(f"  Temps: {result['processing_time_ms']}ms")
    
    # Exemple 2: WO → FR (implicite)
    print("\n--- Exemple 2: Wolof → Français (inverse automatique) ---")
    request_data = {
        "text": "Salaam, naajaat?",
        "source_lang": "wo"
        # target_lang pas spécifié → inverse auto
    }
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    response = requests.post(
        f"{API_URL}/predict",
        json=request_data,
        timeout=TIMEOUT
    )
    result = response.json()
    print(f"\nStatus: {response.status_code}")
    print(f"→ Traduction: {result['translated_text']}")
    print(f"  Temps: {result['processing_time_ms']}ms")
    
    # Exemple 3: Auto-détection langue
    print("\n--- Exemple 3: Auto-détection langue ---")
    request_data = {
        "text": "At yu bari la def.",  # Wolof
        # source_lang pas spécifié → auto-detect
    }
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    response = requests.post(
        f"{API_URL}/predict",
        json=request_data,
        timeout=TIMEOUT
    )
    result = response.json()
    print(f"\nStatus: {response.status_code}")
    print(f"Langue détectée: {result['source_lang']} → {result['target_lang']}")
    print(f"→ Traduction: {result['translated_text']}")

# ============================================================================
# 4. TRADUCTION BATCH
# ============================================================================

def example_batch_translation():
    """Traduire plusieurs textes à la fois"""
    print("\n" + "="*70)
    print("4. TRADUCTION BATCH")
    print("="*70)
    
    request_data = {
        "texts": [
            "Bonjour",
            "Au revoir",
            "Merci beaucoup",
            "Quoi de neuf?",
            "Ça va?"
        ],
        "source_lang": "fr"
    }
    
    print(f"Request: {len(request_data['texts'])} textes")
    for i, text in enumerate(request_data['texts'], 1):
        print(f"  {i}. {text}")
    
    start_time = time.time()
    response = requests.post(
        f"{API_URL}/batch-predict",
        json=request_data,
        timeout=TIMEOUT
    )
    elapsed = time.time() - start_time
    
    result = response.json()
    print(f"\nStatus: {response.status_code}")
    print(f"Temps total: {result['total_time_ms']}ms")
    print(f"Résultats:\n")
    
    for item in result['results']:
        status = "✓" if item['success'] else "✗"
        print(f"{status} {item['source_text']:20} → {item['translated_text']}")
    
    print(f"\nSummary: {result['successful']}/{result['total_texts']} successful")

# ============================================================================
# 5. PARAMÈTRES OPTIONNELS
# ============================================================================

def example_parameters():
    """Montrer l'effet des paramètres num_beams"""
    print("\n" + "="*70)
    print("5. EFFETS DES PARAMÈTRES")
    print("="*70)
    
    text = "Bonjour, comment allez-vous aujourd'hui?"
    
    for num_beams in [1, 2, 4, 8]:
        print(f"\n--- num_beams={num_beams} ---")
        
        request_data = {
            "text": text,
            "source_lang": "fr",
            "target_lang": "wo",
            "num_beams": num_beams
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            json=request_data,
            timeout=TIMEOUT
        )
        elapsed = time.time() - start_time
        
        result = response.json()
        print(f"Traduction: {result['translated_text']}")
        print(f"Temps: {result['processing_time_ms']}ms")

# ============================================================================
# 6. GESTION D'ERREURS
# ============================================================================

def example_error_handling():
    """Montrer comment les erreurs sont gérées"""
    print("\n" + "="*70)
    print("6. GESTION D'ERREURS")
    print("="*70)
    
    # Erreur 1: Texte vide
    print("\n--- Erreur 1: Texte vide ---")
    request_data = {"text": ""}
    response = requests.post(f"{API_URL}/predict", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Error: {response.json()['error']}")
    
    # Erreur 2: Texte trop long
    print("\n--- Erreur 2: Texte trop long ---")
    request_data = {"text": "a" * 2000}
    response = requests.post(f"{API_URL}/predict", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Error: {response.json()['error']}")
    
    # Erreur 3: Langue invalide
    print("\n--- Erreur 3: Langue invalide ---")
    request_data = {"text": "Bonjour", "source_lang": "es"}
    response = requests.post(f"{API_URL}/predict", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Error: {response.json()['message']}")
    
    # Erreur 4: Batch trop gros
    print("\n--- Erreur 4: Batch trop gros ---")
    request_data = {"texts": ["text"] * 150}
    response = requests.post(f"{API_URL}/batch-predict", json=request_data)
    print(f"Status: {response.status_code}")
    print(f"Error: {response.json()['error']}")

# ============================================================================
# 7. PERFORMANCE BENCHMARK
# ============================================================================

def example_performance_benchmark():
    """Mesurer performance inférence"""
    print("\n" + "="*70)
    print("7. BENCHMARK PERFORMANCE")
    print("="*70)
    
    texts = [
        "Bonjour",
        "Comment allez-vous?",
        "Je vais très bien merci",
        "Enchanté de vous rencontrer",
        "À bientôt"
    ]
    
    print(f"\nMesurant latence pour {len(texts)} requêtes...")
    
    latencies = []
    for text in texts:
        request_data = {"text": text, "source_lang": "fr"}
        
        start = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            json=request_data,
            timeout=TIMEOUT
        )
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)
    
    print(f"\nRésultats:")
    print(f"  Min:     {min(latencies):.2f}ms")
    print(f"  Max:     {max(latencies):.2f}ms")
    print(f"  Moyenne: {sum(latencies)/len(latencies):.2f}ms")
    print(f"  Médiane: {sorted(latencies)[len(latencies)//2]:.2f}ms")

# ============================================================================
# 8. UTILISATION PYTHON - CLASS WRAPPER
# ============================================================================

class TranslationClient:
    """Client wrapper pour l'API de traduction"""
    
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
    
    def predict(self, text: str, source_lang: str = None, target_lang: str = None) -> Dict:
        """Traduire un texte"""
        response = requests.post(
            f"{self.api_url}/predict",
            json={
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang
            }
        )
        return response.json()
    
    def batch_predict(self, texts: List[str], source_lang: str = None) -> Dict:
        """Traduire plusieurs textes"""
        response = requests.post(
            f"{self.api_url}/batch-predict",
            json={
                "texts": texts,
                "source_lang": source_lang
            }
        )
        return response.json()
    
    def health(self) -> bool:
        """Vérifier la santé de l'API"""
        try:
            response = requests.get(f"{self.api_url}/health")
            return response.status_code == 200
        except:
            return False

def example_client_wrapper():
    """Exemple utilisation class wrapper"""
    print("\n" + "="*70)
    print("8. CLIENT WRAPPER PYTHON")
    print("="*70)
    
    client = TranslationClient()
    
    # Check health
    print(f"\nAPI Health: {client.health()}")
    
    # Single prediction
    print("\n--- Single Prediction ---")
    result = client.predict("Bonjour", source_lang="fr")
    print(f"Input:  Bonjour")
    print(f"Output: {result['translated_text']}")
    
    # Batch prediction
    print("\n--- Batch Prediction ---")
    texts = ["Bonjour", "Au revoir", "Merci"]
    result = client.batch_predict(texts, source_lang="fr")
    for item in result['results']:
        print(f"{item['source_text']} → {item['translated_text']}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Exécuter tous les exemples"""
    print("\n" + "="*70)
    print("EXEMPLES D'UTILISATION - API TRADUCTION FR↔WO")
    print("="*70)
    
    try:
        # Vérifier que l'API est disponible
        example_health_check()
        
        # Continuer avec d'autres exemples
        example_model_info()
        example_single_translation()
        example_batch_translation()
        example_parameters()
        example_error_handling()
        example_performance_benchmark()
        example_client_wrapper()
        
        print("\n" + "="*70)
        print("✓ TOUS LES EXEMPLES EXÉCUTÉS AVEC SUCCÈS")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        print("\nVérifier que l'API est lancée:")
        print("  docker-compose up")

if __name__ == "__main__":
    main()
