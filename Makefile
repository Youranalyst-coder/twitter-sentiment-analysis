.PHONY: install test train app format verify-live vercel-redirect

install:
	python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

test:
	pytest -q

train:
	python scripts/train.py

app:
	streamlit run app/app.py

format:
        python -m compileall src

verify-live:
        python scripts/check_live_app.py $(if $(URL),--url $(URL),)

vercel-redirect:
        python scripts/update_vercel_redirect.py $(if $(URL),--url $(URL),)
