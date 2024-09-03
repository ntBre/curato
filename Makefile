.PHONY: help test install
help:
	@echo Available commands:
	@echo -e '\ttest - run the tests'
	@echo -e '\tinstall - pip install a development version of the package'

test:
	pytest cura/_tests

install:
	pip install -e .

serve:
	flask --app cura/serve.py run --debug
