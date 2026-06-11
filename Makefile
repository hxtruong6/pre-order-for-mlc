# Convenience targets. Install with: pip install -e ".[dev]"  (add ,viz for figures)
DOCKER_IMAGE_NAME = preorder4mlc

lint:
	ruff check preorder4mlc tests

test:
	python -m pytest tests/ -q

build:
	python -m build && python -m twine check dist/*

reproduce:
	bash scripts/reproduce_capsule.sh results/capsule

docker-run:
	docker build -t $(DOCKER_IMAGE_NAME) .
	docker run --rm -v "$(PWD)/results:/results" $(DOCKER_IMAGE_NAME)
