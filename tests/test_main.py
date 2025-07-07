import pytest
from app.main import app
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

@pytest_asyncio.fixture(scope="function")
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_root(async_client):
    response = await async_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello in our async FastAPI app for named entity recognition!"}

@pytest.mark.asyncio
async def test_token_classification(async_client):
    response = await async_client.post("/token_classification", json={"text": "OpenAI is in San Francisco."})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
    for entity in response.json():
        assert "entity_group" in entity
        assert "score" in entity
        assert "word" in entity
        assert "start" in entity
        assert "end" in entity