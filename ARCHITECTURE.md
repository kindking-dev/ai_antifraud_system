# Architecture Decision Record: AI Behavioral Anti-Fraud Service

## 1. System Metadata
* **Project Name:** AI-Powered Behavioral Anti-Fraud Service
* **Target Audience:** Banks and Fintech Companies
* **Version:** 1.0.0
* **SLA Requirement:** E2E Inference $< 50\text{ms}$
* **Core Stack:** Python 3.14.3, FastAPI, PostgreSQL, Redis (Feature Store), CatBoost, SHAP.

## 2. System Architecture & Performance
The system is built on an asynchronous, non-blocking architecture using modern Python features. 
* **API Layer:** FastAPI with `ORJSONResponse` for maximum serialization throughput. 
* **Feature Store:** Redis is utilized for $O(1)$ time complexity lookups of user historical aggregates (velocity metrics, IP-to-ASN reputation).
* **Concurrency:** All I/O bound operations (database queries, cache retrieval) strictly utilize `async/await` patterns to prevent event loop blocking.
* **Observability:** Structured logging implemented via `structlog` for precise latency tracking and distributed tracing.

## 3. Directory Structure
```text
ai_antifraud_system/
├── app/
│   ├── api/v1/            # API routing and HTTP handlers (e.g., scoring.py)
│   ├── core/              # Global configuration, Pydantic settings
│   ├── ml/                # ML pipeline, inference logic, feature engineering
│   ├── models/            # SQLAlchemy ORM models for PostgreSQL
│   ├── repositories/      # Data access layer (CRUD operations)
│   ├── schemas/           # Pydantic V2 data contracts (transaction, response)
│   └── services/          # Core business logic isolating API from DB
├── data/                  # Raw and processed datasets (IEEE-CIS Fraud Detection)
├── ml_artifacts/          # Compiled binary model artifacts (.cbm)
├── tests/                 # Unit and integration test suites
├── .env                   # Environment variables (excluded from VCS)
├── .gitignore             # VCS exclusion rules (__pycache__, venv)
├── ARCHITECTURE.md        # Single Source of Truth (this document)
├── docker-compose.yml     # Container orchestration for DB/Cache
└── requirements.txt       # Strict dependency locking