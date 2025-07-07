import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello in our async FastAPI app for named entity recognition!"}

@pytest.mark.asyncio
async def test_token_classification():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/token_classification", json={"text": "OpenAI is in San Francisco."})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
    for entity in response.json():
        assert "entity_group" in entity
        assert "score" in entity
        assert "word" in entity
        assert "start" in entity
        assert "end" in entity

# from fastapi.testclient import TestClient
# from app.main import app
# import pytest

# client = TestClient(app)

# @pytest.mark.asyncio
# async def test_root():
#     response = await client.get("/")
#     assert response.status_code == 200
#     assert response.json() == {"message": "Hello in our async FastAPI app for named entity recognition!"}

# @pytest.mark.asyncio
# async def test_token_classification():
#     response = await client.post("/token_classification", json={"text": "OpenAI is in San Francisco."})
#     assert response.status_code == 200
#     assert isinstance(response.json(), list)
#     assert len(response.json()) > 0
#     for entity in response.json():
#         assert "entity_group" in entity
#         assert "score" in entity
#         assert "word" in entity
#         assert "start" in entity
#         assert "end" in entity
        
        
# # def test_predict():
# #     response = client.post("/predict", json={"text": "OpenAI is in San Francisco."})
# #     assert response.status_code == 200
# #     assert "entities" in response.json()
# #     # assert isinstance(response.json()["entities"], list)