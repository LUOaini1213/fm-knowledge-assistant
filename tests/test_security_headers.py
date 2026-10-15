"""Every API response carries the browser hardening headers; only the docs pages are allowed scripts."""
from fastapi.testclient import TestClient

from fmka import api


def test_json_responses_carry_security_headers():
    client = TestClient(api.app)
    for response in (client.get("/health"), client.post("/search", json={"query": "x"})):  # 200 and a 422
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"


def test_docs_page_is_not_given_the_json_policy():
    response = TestClient(api.app).get("/docs")
    assert response.status_code == 200
    assert "content-security-policy" not in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"
