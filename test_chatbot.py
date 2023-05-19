import json
import pytest
from flask import Flask

from chat import get_response
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_get(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.data

def test_predict_valid_input(client):
    data = {'message': 'Hi'}
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    assert json.loads(response.data) == {'answer': 'Hey :-)'}

def test_predict_invalid_input(client):
    data = {'invalid_key': 'Hi'}
    response = client.post('/predict', json=data)
    assert response.status_code == 400
    assert b"Bad Request" in response.data

def test_get_response():
    response = get_response('Hi')
    assert response in ['Hey :-)', 'Hello, thanks for visiting', 'Hi there, what can I do for you?', 'Hi there, how can I help?']

