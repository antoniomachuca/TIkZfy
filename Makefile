.PHONY: format lint test docker-build docker-run

format:
	ruff format .

lint:
	ruff check .
	mypy .

test:
	pytest tests/

docker-build:
	docker build -t tikzfy .

docker-run:
	docker run --rm -it tikzfy
