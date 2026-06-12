# Academic Screenshot Acquisition and Deployment Guide

This guide describes how to capture and crop real project deployment and monitoring screenshots for your Software Engineering bachelor thesis to achieve a professional, academically rigorous presentation.

---

## 1. Project Directory Structure Export
For insertion in your thesis description of the implementation, below is the clean project directory layout. Use the `verbatim` environment in LaTeX to print it.

```text
ai_antifraud_system/
├── app/                        # FastAPI Backend Application Source
│   ├── api/                    # API Endpoints (scoring.py, explain.py)
│   ├── core/                   # System Configuration (config.py, security)
│   ├── ml/                     # Machine Learning Inference Engines
│   ├── models/                 # Database Entity Declarations (db_entities.py)
│   ├── repositories/           # PostgreSQL and Redis Connectors
│   └── schemas/                # Pydantic Request/Response Contracts
├── data/                       # Dataset Storage
│   └── processed/              # Parquet files for training and validation
├── frontend/                   # UI Components
│   ├── dashboard.py            # Streamlit Security Inspector Console
│   └── mobile/                 # Mobile Telemetry Mock/Simulator
├── ml_artifacts/               # Trained models (.cbm) and feature parameters
├── reports/                    # Thesis Evaluation and Appendix Workspace
│   ├── thesis_figures/         # Vector and high-DPI evaluation plots
│   ├── appendix_exports/       # LaTeX ready JSON payloads and logs
│   ├── api_examples/           # API request/response samples
│   ├── backend_logs/           # Structlog JSON production log snippets
│   └── dashboard_screenshots/  # Capture guides and crop templates
├── docker-compose.yml          # Containerized deployment blueprint
├── Dockerfile                  # API service image builder
└── requirements.txt            # Python dependencies
```

---

## 2. Docker Compose Deployment Fragment
This configuration shows the containerized deployment of the complete six-service network. 

```yaml
version: "3.8"

services:
  # 1. Reverse Proxy Gateway
  nginx:
    image: nginx:alpine
    container_name: antifraud_nginx
    ports:
      - "8080:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
      - ./frontend/mobile:/usr/share/nginx/html
    depends_on:
      - api
    restart: always

  # 2. Real-Time Anti-Fraud API (FastAPI)
  api:
    build: .
    container_name: antifraud_api
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - POSTGRES_HOST=db
      - POSTGRES_PORT=5432
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=secretpassword
      - POSTGRES_DB=antifraud_db
    depends_on:
      - db
      - redis
    restart: always

  # 3. Security Analytics Inspector Console (Streamlit)
  dashboard:
    build: .
    container_name: antifraud_dashboard
    ports:
      - "8501:8501"
    command: streamlit run frontend/dashboard.py --server.port=8501
    environment:
      - API_BASE_URL=http://api:8000/api/v1
    depends_on:
      - api
    restart: always

  # 4. Relational Database (Transaction Audit Trails)
  db:
    image: postgres:16-alpine
    container_name: antifraud_db
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secretpassword
      POSTGRES_DB: antifraud_db
    ports:
      - "5433:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: always

  # 5. In-Memory Store (Behavioral Profiles)
  redis:
    image: redis:7.2-alpine
    container_name: redis-antifraud
    ports:
      - "6379:6379"
    restart: always

  # 6. Monitoring & Metrics Dashboard (Grafana)
  grafana:
    image: grafana/grafana:latest
    container_name: antifraud_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafanadata:/var/lib/grafana
    restart: always

volumes:
  pgdata:
  grafanadata:
```

---

## 3. Real-Time Scoring Endpoint Code Snippet
FastAPI code showing the core Late Fusion and Veto Matrix decision engine:

```python
@router.post("/score-transaction", response_model=FraudAnalysisResponse)
async def score_transaction(
    request_body: FraudAnalysisRequest,
    request: Request,
    background_tasks: BackgroundTasks
) -> FraudAnalysisResponse:
    start = time.perf_counter()
    uid = request_body.user_id
    tx_id = request_body.transaction_id
    redis_store = request.app.state.redis_store

    # 1. Fetch live behavioral biometrics telemetry score from Redis cache
    redis_behavior_score = 0.0
    state_key = f"user:{uid}:state"
    raw_score = await redis_store.client.hget(state_key, "latest_behavior_score")
    if raw_score is not None:
        redis_behavior_score = float(raw_score)

    # 2. Fuse with device risk metrics
    device_risk_score = 1.0 - request_body.session_trust_score
    behavior_score = max(redis_behavior_score, device_risk_score)

    # 3. Compute CatBoost model fraud probability
    payload = request_body.model_dump(mode="json")
    tx_prob = float(predict_transaction_model(payload))

    # 4. Apply Late Fusion (Decision Veto Matrix)
    action = FraudAction.ALLOW
    reasons = []

    # High-Risk Vetoes
    if behavior_score >= 0.75 or tx_prob >= 0.85:
        action = FraudAction.BLOCK
        if behavior_score >= 0.75: reasons.append(ReasonCode.SUSPICIOUS_BEHAVIOR)
        if tx_prob >= 0.85: reasons.append(ReasonCode.HIGH_ML_RISK)

    # Joint Risk and Challenge triggers
    if action != FraudAction.BLOCK:
        if tx_prob >= 0.60 and behavior_score >= 0.60:
            action = FraudAction.BLOCK
            reasons.extend([ReasonCode.ELEVATED_RISK, ReasonCode.SUSPICIOUS_BEHAVIOR])
        elif tx_prob >= 0.50:
            action = FraudAction.CHALLENGE
            reasons.append(ReasonCode.ELEVATED_RISK)
        elif behavior_score >= 0.65:
            action = FraudAction.CHALLENGE
            reasons.append(ReasonCode.SUSPICIOUS_BEHAVIOR)

    final_fused_prob = float(max(tx_prob, behavior_score))
    latency_ms = (time.perf_counter() - start) * 1000

    # 5. Queue asynchronous audit logging (Non-blocking Hot Path)
    background_tasks.add_task(
        save_transaction_log,
        {
            "transaction_id": tx_id,
            "user_id": uid,
            "amount_kzt": request_body.amount_kzt,
            "fraud_probability": final_fused_prob,
            "action": action.value,
            "processing_time_ms": latency_ms,
            "timestamp_utc": request_body.timestamp_utc
        }
    )

    return FraudAnalysisResponse(
        transaction_id=tx_id,
        action=action,
        fraud_probability=round(final_fused_prob, 4),
        reason_codes=reasons,
        feature_impacts={
            "behavior_score_impact": round(behavior_score, 4), 
            "tx_model_impact": round(tx_prob, 4)
        },
        processing_time_ms=round(latency_ms, 2)
    )
```

---

## 4. Screenshot Capture & Cropping Guidelines

To ensure your screenshots look like professional, publication-quality software engineering artifacts, strictly adhere to the following acquisition guidelines.

### A. General Presentation Rules (All Screenshots)
* **Hide UI Clutter**: Never include desktop taskbars (Windows Start menu, macOS Dock), system trays (time/date, battery), browser frames (URL bar, tabs, window close/minimize buttons), or background windows.
* **Aspect Ratio and Sizing**: Crop screens to tight bounding boxes focusing only on the application container. Maintain consistent widths (e.g., standardizing on $1280 \times 720$ pixels before downscaling in LaTeX).
* **Color Schemes**: Use high-contrast settings. If the thesis is printed in grayscale, verify that elements have sufficient luminance differences. Light backgrounds are strongly preferred for thesis formatting, unless depicting terminal output.
* **Resolution**: Keep visual text readable. Do not downsample images excessively; save screenshots at 200–300 DPI to prevent rasterization blur on scaling.

### B. Docker Container Deployment Screenshot
Capture the output of `docker ps` to demonstrate the active, containerized runtime of the six distinct modules.
* **Command**: Run `docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"` in your terminal to generate a clean tabular layout without excessive horizontal wrapping.
* **Verification Scope**: Ensure all six containers are visible:
  1. `antifraud_nginx` (Up)
  2. `antifraud_api` (Up)
  3. `antifraud_dashboard` (Up)
  4. `antifraud_db` (Up)
  5. `redis-antifraud` (Up)
  6. `antifraud_grafana` (Up)
* **Capture Boundary**: Crop to the terminal window boundaries. Use a dark theme for the terminal window (monochrome background like deep black or navy blue, white text) for authentic command-line evidence.

### C. Security Analytics Dashboard Screenshot
Capture the dashboard console to provide visual evidence of live system diagnostics.
* **Target Interface**: Open the Streamlit inspector panel at `http://localhost:8501`.
* **Required Metrics**: Ensure the following visual panels are active:
  1. **Latency Metrics**: Sparkline or gauge showing average processing speed (typically $< 15$ ms) and P95 latency ($< 30$ ms).
  2. **Fraud Activity**: Current blocked vs allowed transaction chart.
  3. **Anomaly Scores**: Biometric risk profile similarity indicators.
  4. **API Statistics**: Total requests served and system throughput (TPS).
* **Capture Boundary**: Crop exactly to the Streamlit layout container. Remove the browser menu bar, scrolls, and external boundaries. Use a clean, light-mode background in Streamlit so that it fits natively inside the paper margins of the thesis.

### D. Grafana Monitoring Dashboard Screenshot
Capture the performance monitoring charts showing telemetry aggregation.
* **Target Interface**: Open Grafana at `http://localhost:3000` (credentials: `admin` / `admin`).
* **Required Metrics**:
  1. **HTTP Request Throughput**: Total API hits per second (SLA health).
  2. **FastAPI SLA Latency Heatmap**: Real-time bucketed latency.
  3. **Redis Memory Usage**: RAM footprint monitoring for session profiles.
  4. **PostgreSQL Connection Pool Status**: Connection count metrics.
* **Capture Boundary**: Crop strictly to the main grid of dashboard panels. Hide Grafana's left navigation drawer (collapse it using the menu button) and top time-range selectors to maximize screen real estate.
