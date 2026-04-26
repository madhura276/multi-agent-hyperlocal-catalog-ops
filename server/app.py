"""
FastAPI application for the Multi-Agent Hyperlocal Catalog Ops environment.
"""

import os

from openenv.core.env_server.http_server import create_app # type: ignore

try:
    from ..models import MultiAgentAction, MultiAgentObservation
    from .environment import MultiAgentHyperlocalCatalogOpsEnvironment
except ImportError:
    from models import MultiAgentAction, MultiAgentObservation
    from server.environment import MultiAgentHyperlocalCatalogOpsEnvironment


app = create_app(
    MultiAgentHyperlocalCatalogOpsEnvironment,
    MultiAgentAction,
    MultiAgentObservation,
    env_name="multi_agent_hyperlocal_catalog_ops",
    max_concurrent_envs=4,
)
from fastapi.responses import HTMLResponse # type: ignore

@app.get("/")
def root():
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Multi-Agent Hyperlocal Catalog Ops</title>
  <style>
    :root {
      --bg: #f6f4ff;
      --ink: #19162b;
      --muted: #5f587a;
      --card: rgba(255, 255, 255, 0.78);
      --line: rgba(69, 49, 130, 0.14);
      --accent: #0f9d7a;
      --accent-2: #5d39d6;
      --accent-3: #ff5f8f;
      --shadow: 0 24px 60px rgba(47, 31, 96, 0.14);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Segoe UI", "Inter", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 95, 143, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(93, 57, 214, 0.16), transparent 30%),
        linear-gradient(180deg, #fcfbff 0%, var(--bg) 100%);
      min-height: 100vh;
    }

    .wrap {
      width: min(1100px, calc(100% - 32px));
      margin: 0 auto;
      padding: 40px 0 56px;
    }

    .hero {
      background: linear-gradient(135deg, rgba(43, 31, 88, 0.96), rgba(94, 51, 145, 0.92));
      color: #fff;
      border-radius: 28px;
      padding: 36px;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }

    .hero::after {
      content: "";
      position: absolute;
      width: 280px;
      height: 280px;
      right: -60px;
      top: -80px;
      background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%);
    }

    .eyebrow {
      display: inline-block;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.13);
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 16px;
    }

    h1 {
      font-size: clamp(2rem, 4vw, 3.5rem);
      line-height: 1.02;
      margin: 0 0 14px;
      max-width: 760px;
    }

    .subtitle {
      max-width: 760px;
      font-size: 1.05rem;
      line-height: 1.6;
      color: rgba(255,255,255,0.86);
      margin: 0 0 28px;
    }

    .actions {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }

    .btn {
      text-decoration: none;
      padding: 13px 18px;
      border-radius: 14px;
      font-weight: 600;
      transition: transform 0.18s ease, opacity 0.18s ease, background 0.18s ease;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }

    .btn:hover { transform: translateY(-1px); }

    .btn-primary {
      background: #fff;
      color: #271b59;
    }

    .btn-secondary {
      background: rgba(255,255,255,0.12);
      color: #fff;
      border: 1px solid rgba(255,255,255,0.16);
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 18px;
      margin-top: 24px;
    }

    .card {
      background: var(--card);
      backdrop-filter: blur(10px);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 10px 28px rgba(61, 44, 118, 0.06);
    }

    .card h2, .card h3 {
      margin: 0 0 10px;
    }

    .card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }

    .list {
      margin: 14px 0 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.8;
    }

    .agents {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 24px;
    }

    .agent {
      border-radius: 18px;
      padding: 18px;
      color: #fff;
      min-height: 140px;
    }

    .agent h3 {
      margin-bottom: 8px;
      font-size: 1.05rem;
    }

    .agent p {
      color: rgba(255,255,255,0.86);
      line-height: 1.5;
      margin: 0;
    }

    .curation { background: linear-gradient(135deg, #15785d, #1ea780); }
    .dedupe { background: linear-gradient(135deg, #8f6c00, #d49d00); }
    .pricing { background: linear-gradient(135deg, #5a34c8, #815ff2); }
    .oversight { background: linear-gradient(135deg, #ad2557, #ff5f8f); }

    .footer {
      margin-top: 28px;
      text-align: center;
      color: var(--muted);
      font-size: 0.95rem;
    }

    code {
      background: rgba(93, 57, 214, 0.08);
      color: #3f289c;
      padding: 2px 7px;
      border-radius: 8px;
      font-size: 0.95em;
    }

    @media (max-width: 640px) {
      .hero { padding: 28px 22px; }
      .wrap { width: min(100% - 20px, 1100px); padding-top: 20px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">OpenEnv Benchmark</div>
      <h1>Multi-Agent Hyperlocal Catalog Ops</h1>
      <p class="subtitle">
        A quick-commerce inventory curation environment where specialized agents coordinate under oversight
        to clean noisy catalog data, handle ambiguity safely, and produce measurable post-training improvement.
      </p>
      <div class="actions">
        <a class="btn btn-primary" href="/docs">Open API Docs</a>
        <a class="btn btn-secondary" href="/health">Health Check</a>
        <a class="btn btn-secondary" href="/metadata">Environment Metadata</a>
      </div>
    </section>

    <section class="grid">
      <article class="card">
        <h2>What It Solves</h2>
        <p>
          Hyperlocal inventory records arrive from merchants, POS exports, dark stores, and sync systems.
          This environment trains agents to resolve messy titles, duplicate records, pricing anomalies,
          and ambiguous cases without unsafe catalog updates.
        </p>
      </article>
      <article class="card">
        <h2>Task Design</h2>
        <p>
          The benchmark includes three deterministic tasks with increasing difficulty:
          <code>easy_single_store_cleanup</code>, <code>medium_multistore_conflict</code>,
          and <code>hard_ambiguous_oversight_batch</code>.
        </p>
      </article>
      <article class="card">
        <h2>Reward Signal</h2>
        <p>
          Reward combines correctness, coordination, and oversight quality so the system improves not just
          fluent output, but safe and operationally useful multi-agent behavior.
        </p>
      </article>
    </section>

    <section class="agents">
      <article class="agent curation">
        <h3>Curation Agent</h3>
        <p>Normalizes titles, sizes, and categories for noisy inventory records.</p>
      </article>
      <article class="agent dedupe">
        <h3>Dedupe Agent</h3>
        <p>Proposes safe duplicate merges using structured record-level signals.</p>
      </article>
      <article class="agent pricing">
        <h3>Pricing Agent</h3>
        <p>Finds obvious pricing anomalies and suggests corrected values.</p>
      </article>
      <article class="agent oversight">
        <h3>Oversight Agent</h3>
        <p>Approves, rejects, or escalates proposals before shared state changes.</p>
      </article>
    </section>

    <section class="grid">
      <article class="card">
        <h3>Core API Endpoints</h3>
        <ul class="list">
          <li><code>POST /reset</code> to start a new episode</li>
          <li><code>POST /step</code> to execute an agent action</li>
          <li><code>GET /state</code> to inspect current environment state</li>
          <li><code>GET /schema</code> to retrieve action and observation schemas</li>
        </ul>
      </article>
      <article class="card">
        <h3>Why It Matters</h3>
        <p>
          This benchmark goes beyond chatbot output by modeling safe multi-agent decision-making in a realistic
          enterprise workflow, with a training-ready setup and measurable post-training improvement.
        </p>
      </article>
    </section>

    <p class="footer">
      Shared state changes only after oversight review. Explore the environment through <a href="/docs">/docs</a>.
    </p>
  </div>
</body>
</html>
        """
    )

def main() -> None:
    import uvicorn # type: ignore

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
