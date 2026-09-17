.PHONY: corpus index evaluate evaluate-hybrid search test

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

test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
