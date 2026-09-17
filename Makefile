.PHONY: install corpus index evaluate evaluate-hybrid evaluate-answers search serve benchmark test docker-up docker-down

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-api.txt

corpus:
	python3 scripts/build_corpus.py

index:
	.venv/bin/python scripts/build_vector_index.py

evaluate:
	.venv/bin/python scripts/evaluate_retrieval.py

evaluate-hybrid:
	.venv/bin/python scripts/evaluate_hybrid.py

search:
	.venv/bin/python scripts/search.py "$(QUERY)" $(if $(DEPARTMENT),--department $(DEPARTMENT),)

evaluate-answers:
	.venv/bin/python scripts/evaluate_answers.py

serve:
	.venv/bin/uvicorn enterprise_rag.api:app --host 0.0.0.0 --port 8000

benchmark:
	.venv/bin/python scripts/benchmark_api.py

test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
