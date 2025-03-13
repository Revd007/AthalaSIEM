import pytest
from fastapi.testclient import TestClient
from main import app
from database.connection import get_db, AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

client = TestClient(app)

# Test Authentication
def test_login():
    response = client.post("/auth/login", json={
        "email": "admin@athala.com",
        "password": "admin123"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()
    return response.json()["access_token"]

# Test Events API
def test_get_events():
    token = test_login()
    response = client.get(
        "/api/events", 
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# Test Alerts API
def test_get_alerts():
    token = test_login()
    response = client.get(
        "/api/alerts", 
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# Test Dashboard API
def test_get_dashboard_stats():
    token = test_login()
    response = client.get(
        "/api/dashboard/stats", 
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200