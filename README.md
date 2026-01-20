# Alphapoly

Polymarket alpha detection platform. Finds covering portfolios across correlated prediction markets using predefined rules and LLM decisions. The system detects relationships between markets, classifies them to identify hedging pairs, and tracks their prices—acting when favorable pricing creates profit opportunities. Automated trading is under development (see experiments/trading); for now, users can seamlessly open identified markets directly on Polymarket.

![Dashboard Screenshot](assets/dashboard-screenshot.png)

## How It Works

1. **Groups** — Fetches multi-outcome markets from Polymarket (e.g., "US election by X date")
2. **Implications** — LLM extracts logical relationships between groups
3. **Portfolios** — Finds position pairs that hedge each other with high coverage probability

## Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/)
- **Node.js 18+** via [fnm](https://github.com/Schniz/fnm), nvm, or brew

## Quick Start

```bash
cp .env.example .env          # Add OPENROUTER_API_KEY (and CHAINSTACK_NODE for blockchain transactions)

# With make
make install && make dev

# Without make
cd backend && uv sync
cd frontend && npm install
cd backend && uv run python -m uvicorn server.main:app --port 8000 &
cd frontend && npm run dev
```

Dashboard: http://localhost:3000 · API: http://localhost:8000/docs

## Commands

**With make** (auto-detects fnm/nvm/volta):
```bash
make install    # Install deps
make dev        # Start both servers
```

**Without make**:
```bash
# Backend
cd backend && uv sync
cd backend && uv run python -m uvicorn server.main:app --reload --port 8000

# Frontend
cd frontend && npm install
cd frontend && npm run dev
```
