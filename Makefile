.PHONY: corpus index evaluate test

corpus:
	python3 scripts/build_corpus.py

index:
	.venv/bin/python scripts/build_vector_index.py

evaluate:
	.venv/bin/python scripts/evaluate_retrieval.py

test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
