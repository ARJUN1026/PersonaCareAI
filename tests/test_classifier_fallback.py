from src.classifier import _rule_based_fallback

def test_technical():
    assert _rule_based_fallback("Bearer token API returns 401 and logs show invalid_token").persona == "Technical Expert"

def test_frustrated():
    assert _rule_based_fallback("Nothing works and I am frustrated!!!").persona == "Frustrated User"

def test_executive():
    assert _rule_based_fallback("What is the business impact and SLA timeline?").persona == "Business Executive"
