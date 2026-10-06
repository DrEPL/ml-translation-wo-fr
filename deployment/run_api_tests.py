import urllib.request
import json

base_url = "http://localhost:5000"

def test_request(name, path, method="GET", payload=None, expected_status=200):
    url = f"{base_url}{path}"
    headers = {"Content-Type": "application/json"} if payload else {}
    data = json.dumps(payload).encode('utf-8') if payload else None
    
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            body = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        status = e.code
        body = json.loads(e.read().decode('utf-8'))
        
    passed = (status == expected_status)
    status_icon = "✓ PASS" if passed else "✗ FAIL"
    print(f"[{status_icon}] {name} - Status: {status} (Expected {expected_status})")
    if not passed:
        print(f"  Response: {body}")
    return passed

print("--- EXÉCUTION DU SUITE DE TESTS POSTMAN SUR L'API FLASK ---")
results = []
results.append(test_request("01. Health Check", "/health", "GET", expected_status=200))
results.append(test_request("02. Model Info", "/model-info", "GET", expected_status=200))
results.append(test_request("03. Predict FR->WO Auto-detect", "/predict", "POST", {"text": "Bonjour tout le monde"}, expected_status=200))
results.append(test_request("04. Predict WO->FR Explicit", "/predict", "POST", {"text": "Naka suba si", "source_lang": "wo", "target_lang": "fr"}, expected_status=200))
results.append(test_request("05. Batch Predict", "/batch-predict", "POST", {"texts": ["Comment vas-tu ?", "Merci"], "source_lang": "fr", "target_lang": "wo"}, expected_status=200))
results.append(test_request("06. Error - Missing Text", "/predict", "POST", {"source_lang": "fr"}, expected_status=400))
results.append(test_request("07. Error - Empty Text", "/predict", "POST", {"text": "   "}, expected_status=400))
results.append(test_request("08. Error - Max Chars Limit", "/predict", "POST", {"text": "a" * 1001}, expected_status=400))
results.append(test_request("09. Error - Invalid Batch Format", "/batch-predict", "POST", {"texts": "not_a_list"}, expected_status=400))
results.append(test_request("10. Error - 404 Endpoint", "/non-existing-path", "GET", expected_status=404))

print("\n--- RÉSUMÉ DES TESTS ---")
print(f"Total: {len(results)} | Reussis: {sum(results)} | Échoués: {len(results) - sum(results)}")
