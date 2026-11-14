from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app


def test_health_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body) == {"status", "app", "environment"}


def test_health_uses_settings_override() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        app_name="Test", app_env="test"
    )
    response = TestClient(app).get("/health")
    assert response.json() == {"status": "ok", "app": "Test", "environment": "test"}


def test_settings_read_environment(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("CORS_ORIGINS", '["http://a.test"]')
    s = Settings()
    assert s.app_env == "production"
    assert s.cors_origins == ["http://a.test"]
