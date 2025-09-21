# Makefile for finoptiagents-remote-new
# This combines best practices for local development and deployment.

.PHONY: help install playground backend test lint setup-dev-env

# Default target when no arguments are given to make
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install          Install dependencies in editable mode using uv"
	@echo "  playground       Launch the Streamlit UI for local testing"
	@echo "  backend          Deploy agent to Vertex AI Agent Engine"
	@echo "  test             Run unit and integration tests"
	@echo "  lint             Run code quality checks"
	@echo "  setup-dev-env    Set up development environment resources using Terraform"

# Install dependencies in editable mode for local development.
install:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; . "$$HOME/.cargo/env"; }
	uv pip install -e ".[dev,jupyter]"

# Run the local Streamlit playground
playground:
	uv run streamlit run app/playground.py

# Deploy to Agent Engine.
# This first exports the production dependencies to .requirements.txt, which is
# then used by the deployment script.
backend:
	uv export --no-hashes --no-header --no-dev --no-emit-project --no-annotate > .requirements.txt 2>/dev/null || \
	uv export --no-hashes --no-header --no-dev --no-emit-project > .requirements.txt
	uv run python app/agent_engine_app.py

# Run tests
test:
	uv run pytest

# Lint and format code
lint:
	uv run ruff check . && uv run ruff format . --check && uv run mypy .

# Setup development environment
#setup-dev-env:
#terraform -chdir=deployment/dev apply -auto-approve
setup-dev-env:
	PROJECT_ID=$$(gcloud config get-value project) && \
	(cd deployment/terraform/dev && terraform init && terraform apply --var-file vars/env.tfvars --var dev_project_id=$$PROJECT_ID --auto-approve)
