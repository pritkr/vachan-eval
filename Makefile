.PHONY: install run eval eval-stream test web docker lint clean
PY ?= python3

install:
	$(PY) -m pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --port 8000

eval:
	$(PY) eval/run.py --seed 7

# v3: streaming STT (fake clock) + semantic cache + voice-budget gate
eval-stream:
	$(PY) eval/run.py --seed 7 --stream --budget

test:
	$(PY) -m pytest -q

web:
	cd web && $(PY) -m http.server 5174

docker:
	docker compose up --build

lint:
	$(PY) -m py_compile app/*.py eval/*.py tests/*.py

clean:
	rm -rf __pycache__ app/__pycache__ eval/__pycache__ tests/__pycache__ .pytest_cache vachan.db eval/results.json eval/results.md
