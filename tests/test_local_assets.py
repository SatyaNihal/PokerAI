"""The compact app bundle must contain every record linked by the dashboards."""
from fastapi.testclient import TestClient

from backend.app import create_app


def test_dashboard_records_survive_without_training_run_directories():
    with TestClient(create_app()) as client:
        small = client.get('/research-summary').json()
        holdem = client.get('/holdem/research-summary').json()
        required = {run['experiment_id'] for run in small['runs'] + holdem['runs']}
        required.update([small['benchmark_experiment'], small['tournament_experiment']])
        assert {row['experiment_id'] for row in client.get('/experiments').json()} == required
        for identifier in required:
            response = client.get('/experiments/' + identifier)
            assert response.status_code == 200
            record = response.json()
            assert record['experiment_id'] == identifier and record['metrics']
            assert record['provenance']['code_fingerprint']
        for agent in client.get('/agents').json():
            assert agent['experiment_id'] in required
            assert 'artifact' not in agent
        assert client.get('/experiments/missing').status_code == 404

