from __future__ import annotations

import json
import queue
import time
from collections import Counter
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse


def create_dashboard_app(alert_manager) -> FastAPI:
    app = FastAPI(title="Clinical Risk Dashboard", version="2.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/alerts")
    def alerts(limit: int = 50) -> list[dict[str, Any]]:
        return alert_manager.get_latest(limit=limit)

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        data = alert_manager.get_summary()
        data["active_streams"] = max(int(data.get("active_streams", 0)), len(alert_manager.get_stream_ids()))
        return data

    @app.get("/api/alerts")
    def api_alerts(limit: int = 120, stream_id: str | None = None, min_level: str | None = None) -> list[dict[str, Any]]:
        alerts = alert_manager.get_latest(limit=limit)
        if stream_id:
            alerts = [a for a in alerts if a.get("stream_id") == stream_id]
        if min_level:
            rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
            floor = rank.get(min_level.upper(), 0)
            alerts = [a for a in alerts if rank.get((a.get("event") or {}).get("risk_level", "LOW"), 0) >= floor]
        return alerts

    @app.get("/api/streams")
    def streams() -> dict[str, list[str]]:
        return {"streams": alert_manager.get_stream_ids()}

    @app.get("/api/stream/{stream_id}.mjpg")
    def stream(stream_id: str, fps: int = 12) -> StreamingResponse:
        target_fps = max(1, min(int(fps), 30))
        interval = 1.0 / float(target_fps)

        def gen():
            last_seq = -1
            while True:
                frame, seq = alert_manager.get_latest_frame(stream_id)
                if frame is None or seq == last_seq:
                    time.sleep(interval)
                    continue

                last_seq = seq
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                )
                yield header + frame + b"\r\n"
                time.sleep(interval)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/events")
    def event_stream() -> StreamingResponse:
        def gen():
            sub = alert_manager.subscribe()
            try:
                while True:
                    try:
                        event = sub.get(timeout=15.0)
                    except queue.Empty:
                        yield "event: ping\ndata: {}\n\n"
                        continue
                    payload = json.dumps(event)
                    yield f"event: alert\ndata: {payload}\n\n"
            finally:
                alert_manager.unsubscribe(sub)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return _dashboard_html()

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        return _dashboard_html()

    return app


def _dashboard_html() -> str:
    return """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Clinical Risk Dashboard</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #11202d;
      --muted: #5f7284;
      --line: #dbe4ec;
      --low: #2b8a3e;
      --med: #b06f00;
      --high: #c74a12;
      --critical: #c92a2a;
      --focus: #0a6ca8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: \"Segoe UI\", \"Aptos\", \"Source Sans Pro\", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    .shell {
      max-width: 1320px;
      margin: 0 auto;
      padding: 20px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
      margin-bottom: 14px;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      letter-spacing: .2px;
      font-weight: 700;
    }
    .subtitle {
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }
    .stamp {
      font-size: 12px;
      color: var(--muted);
      border: 1px solid var(--line);
      padding: 7px 10px;
      border-radius: 8px;
      background: var(--panel);
    }
    .metrics {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(4, minmax(0,1fr));
      margin-bottom: 12px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
    }
    .metric .k {
      font-size: 11px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: .7px;
    }
    .metric .v {
      margin-top: 4px;
      font-size: 30px;
      font-weight: 700;
      line-height: 1;
    }
    .workspace {
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 15px;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      font-size: 13px;
      margin-bottom: 10px;
      color: var(--ink);
      background: #fff;
    }
    select:focus {
      outline: 2px solid color-mix(in srgb, var(--focus) 28%, transparent);
      border-color: var(--focus);
    }
    .main {
      display: grid;
      gap: 12px;
    }
    .feed {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0f1820;
      overflow: hidden;
      aspect-ratio: 16 / 9;
      position: relative;
    }
    .feed img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #0f1820;
    }
    .feed-state {
      position: absolute;
      left: 10px;
      bottom: 10px;
      font-size: 12px;
      color: #e9f1f7;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(0, 0, 0, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .alerts {
      display: grid;
      gap: 8px;
      max-height: 44vh;
      overflow: auto;
      padding-right: 2px;
    }
    .alert {
      border: 1px solid var(--line);
      border-left: 4px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    .alert .row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 4px;
    }
    .track {
      font-size: 14px;
      font-weight: 700;
    }
    .stream {
      font-size: 12px;
      color: var(--muted);
    }
    .badge {
      font-size: 11px;
      font-weight: 700;
      border-radius: 999px;
      padding: 3px 8px;
      color: #fff;
    }
    .meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      font-size: 12px;
      color: var(--muted);
    }
    .LOW { border-left-color: var(--low); }
    .MEDIUM { border-left-color: var(--med); }
    .HIGH { border-left-color: var(--high); }
    .CRITICAL { border-left-color: var(--critical); }
    .LOW .badge { background: var(--low); }
    .MEDIUM .badge { background: var(--med); }
    .HIGH .badge { background: var(--high); }
    .CRITICAL .badge { background: var(--critical); }
    .empty {
      text-align: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 32px 10px;
      font-size: 13px;
      background: #fff;
    }
    @media (max-width: 960px) {
      .metrics { grid-template-columns: repeat(2, minmax(0,1fr)); }
      .workspace { grid-template-columns: 1fr; }
      .alerts { max-height: none; }
    }
  </style>
</head>
<body>
  <div class=\"shell\">
    <div class=\"topbar\">
      <div>
        <h1>Clinical Risk Dashboard</h1>
        <div class=\"subtitle\">Edge AI monitoring with live pose overlays and temporal risk scoring.</div>
      </div>
      <div class=\"stamp\">Updated <span id=\"lastUpdate\">-</span></div>
    </div>

    <section class=\"metrics\">
      <article class=\"metric\"><div class=\"k\">Active Streams</div><div id=\"mStreams\" class=\"v\">0</div></article>
      <article class=\"metric\"><div class=\"k\">Active Tracks</div><div id=\"mTracks\" class=\"v\">0</div></article>
      <article class=\"metric\"><div class=\"k\">High</div><div id=\"mHigh\" class=\"v\">0</div></article>
      <article class=\"metric\"><div class=\"k\">Critical</div><div id=\"mCritical\" class=\"v\">0</div></article>
    </section>

    <section class=\"workspace\">
      <aside class=\"panel\">
        <h2>Filters</h2>
        <label for=\"streamFilter\">Stream</label>
        <select id=\"streamFilter\"><option value=\"\">All Streams</option></select>

        <label for=\"levelFilter\">Minimum Severity</label>
        <select id=\"levelFilter\">
          <option value=\"LOW\">Low</option>
          <option value=\"MEDIUM\">Medium</option>
          <option value=\"HIGH\">High</option>
          <option value=\"CRITICAL\">Critical</option>
        </select>
      </aside>
      <main class=\"panel main\">
        <section>
          <h2>Live Camera + Pose</h2>
          <div class=\"feed\">
            <img id=\"streamFeed\" alt=\"Live stream\" />
            <div class=\"feed-state\" id=\"streamState\">Waiting for stream...</div>
          </div>
        </section>
        <section>
          <h2>Live Alert Feed</h2>
          <div id=\"alerts\" class=\"alerts\"></div>
        </section>
      </main>
    </section>
  </div>

  <script>
    const alertsEl = document.getElementById('alerts');
    const streamFilter = document.getElementById('streamFilter');
    const levelFilter = document.getElementById('levelFilter');
    const streamFeed = document.getElementById('streamFeed');
    const streamState = document.getElementById('streamState');
    const mStreams = document.getElementById('mStreams');
    const mTracks = document.getElementById('mTracks');
    const mHigh = document.getElementById('mHigh');
    const mCritical = document.getElementById('mCritical');
    const lastUpdate = document.getElementById('lastUpdate');

    let cache = [];

    function fmtTs(ts) {
      if (!ts) return '-';
      return new Date(ts * 1000).toLocaleTimeString();
    }

    function renderAlerts() {
      if (!cache.length) {
        alertsEl.innerHTML = '<div class="empty">No alerts yet. System is online and waiting for risk events.</div>';
        return;
      }
      alertsEl.innerHTML = cache.slice().reverse().map((row) => {
        const e = row.event || {};
        const level = e.risk_level || 'LOW';
        const eventName = e.event || 'stable';
        const reasons = (e.reasons || []).length ? (e.reasons || []).join(', ') : 'none';
        return `
          <article class="alert ${level}">
            <div class="row">
              <div>
                <div class="track">Track #${e.track_id ?? '-'}</div>
                <div class="stream">${row.stream_id ?? 'unknown stream'}</div>
              </div>
              <span class="badge">${level}</span>
            </div>
            <div class="meta">
              <span>event: ${eventName}</span>
              <span>confidence: ${(e.confidence ?? 0).toFixed(2)}</span>
              <span>time: ${fmtTs(e.timestamp)}</span>
              <span>reason: ${reasons}</span>
            </div>
          </article>
        `;
      }).join('');
    }

    function updateFeed() {
      const sid = streamFilter.value;
      if (!sid) {
        streamFeed.removeAttribute('src');
        streamState.textContent = 'Waiting for stream...';
        return;
      }
      streamFeed.src = `/api/stream/${encodeURIComponent(sid)}.mjpg?fps=12&t=${Date.now()}`;
      streamState.textContent = `Live: ${sid}`;
    }

    async function refreshStreams() {
      try {
        const res = await fetch('/api/streams');
        const data = await res.json();
        const streams = data.streams || [];
        const current = streamFilter.value;

        streamFilter.innerHTML = '<option value="">All Streams</option>' + streams.map(s => `<option value="${s}">${s}</option>`).join('');
        if (streams.length === 0) {
          streamFilter.value = '';
        } else if (streams.includes(current)) {
          streamFilter.value = current;
        } else {
          streamFilter.value = streams[0];
        }
        updateFeed();
      } catch (_) {}
    }

    async function refreshSummary() {
      try {
        const res = await fetch('/api/summary');
        const s = await res.json();
        mStreams.textContent = s.active_streams ?? 0;
        mTracks.textContent = s.active_tracks ?? 0;
        mHigh.textContent = s.high_alerts ?? 0;
        mCritical.textContent = s.critical_alerts ?? 0;
      } catch (_) {}
    }

    async function refreshAlerts() {
      try {
        const query = new URLSearchParams({ limit: '220' });
        if (streamFilter.value) query.set('stream_id', streamFilter.value);
        if (levelFilter.value) query.set('min_level', levelFilter.value);
        const res = await fetch('/api/alerts?' + query.toString());
        cache = await res.json();
        renderAlerts();
        lastUpdate.textContent = new Date().toLocaleTimeString();
      } catch (_) {}
    }

    streamFilter.addEventListener('change', async () => {
      updateFeed();
      await refreshAlerts();
    });
    levelFilter.addEventListener('change', refreshAlerts);

    async function boot() {
      await refreshStreams();
      await refreshSummary();
      await refreshAlerts();

      const es = new EventSource('/api/events');
      es.addEventListener('alert', async () => {
        await refreshSummary();
        await refreshAlerts();
      });

      setInterval(refreshStreams, 5000);
      setInterval(refreshSummary, 3000);
      setInterval(refreshAlerts, 3000);
    }

    boot();
  </script>
</body>
</html>
"""


def build_summary(alerts: list[dict[str, Any]]) -> dict[str, Any]:
    streams = {a.get("stream_id", "unknown") for a in alerts}
    levels = Counter((a.get("event") or {}).get("risk_level", "LOW") for a in alerts)
    active_tracks = {
        (a.get("stream_id"), (a.get("event") or {}).get("track_id"))
        for a in alerts
        if (a.get("event") or {}).get("track_id") is not None
    }

    return {
        "active_streams": len(streams),
        "active_tracks": len(active_tracks),
        "high_alerts": int(levels.get("HIGH", 0)),
        "critical_alerts": int(levels.get("CRITICAL", 0)),
        "generated_at": time.time(),
    }
