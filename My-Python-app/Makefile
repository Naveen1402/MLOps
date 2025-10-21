# Python and tools
PYTHON ?= python3
PIP ?= pip3
APP_NAME = my-python-app
IMAGE_NAME = my-python-app:latest

.PHONY: all install lint format test docker

all: install lint test

install:
	$(PIP) install -r requirements.txt

lint:
	flake8 app tests

format:
	black app tests

test:
	pytest -v
	
loadtest:
	locust -f locustfile.py --headless -u 10 -r 1 -t 30s --host http://localhost:8000


docker:
	docker build -t $(IMAGE_NAME) .

push-docker:
	docker tag $(IMAGE_NAME) <YOUR_REGISTRY>/$(IMAGE_NAME)
	docker push <YOUR_REGISTRY>/$(IMAGE_NAME)
