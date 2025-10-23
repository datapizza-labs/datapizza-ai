
test:
	uv run pytest --tb=short -v

watch-tests:
	find . -name "*.py" -not -path "*/site-packages/*" | entr uv run pytest --tb=short -v

format:
	uvx ruff format .

linter-check:
	uvx ruff check .

linter-fix:
	uvx ruff check --fix

linter-force-fix:
	uvx ruff check --fix --unsafe-fixes

dependency-check:
	uv run deptry .

run_docs:
	uv pip install mkdocs-material  pymdown-extensions mkdocs-awesome-pages-plugin mkdocstrings-python
	uv run mkdocs serve --livereload
