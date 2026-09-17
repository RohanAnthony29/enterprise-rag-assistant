.PHONY: corpus test

corpus:
	python3 scripts/build_corpus.py

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

