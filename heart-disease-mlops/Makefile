.PHONY: setup data eda train test lint run docker-build docker-run mlflow clean

setup:          ## install dependencies
	pip install -r requirements.txt

data:           ## download dataset
	python scripts/download_data.py

eda:            ## generate EDA figures
	python -m src.eda

train:          ## train + tune models, log to MLflow, save best pipeline
	python -m src.train

test:           ## run unit tests
	pytest -v

lint:           ## run flake8
	flake8 src app tests scripts --max-line-length=100

run:            ## run the API locally (gunicorn)
	gunicorn -w 2 -b 0.0.0.0:8000 app.main:app

mlflow:         ## open the MLflow UI
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-build:   ## build the serving image
	docker build -t heart-disease-api:latest .

docker-run:     ## run the container locally
	docker run --rm -p 8000:8000 heart-disease-api:latest

monitor:        ## start API + Prometheus + Grafana
	docker compose -f monitoring/docker-compose.yml up --build

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache mlruns mlflow.db
