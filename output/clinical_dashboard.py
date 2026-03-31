from __future__ import annotations

import json
import queue
import time
from collections import Counter
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse


def create_dashboard_app(alert_manager) -> FastAPI:
    app = FastAPI(title="Clinical Risk Dashboard", version="3.0.0")

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

    @app.get("/api/open_alerts")
    def api_open_alerts(limit: int = 100, min_level: str = "HIGH") -> list[dict[str, Any]]:
        return alert_manager.get_open_alerts(limit=limit, min_level=min_level)

    @app.post("/api/ack/{stream_id}/{track_id}")
    def ack(stream_id: str, track_id: int) -> dict[str, Any]:
        alert_manager.ack_track(stream_id=stream_id, track_id=track_id)
        return {"ok": True, "stream_id": stream_id, "track_id": int(track_id), "acknowledged": True}

    @app.post("/api/unack/{stream_id}/{track_id}")
    def unack(stream_id: str, track_id: int) -> dict[str, Any]:
        alert_manager.unack_track(stream_id=stream_id, track_id=track_id)
        return {"ok": True, "stream_id": stream_id, "track_id": int(track_id), "acknowledged": False}

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
  <title>Clinical Risk Command Center</title>
  <style>
    :root {
      --bg: #eef3f7;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #52606d;
      --line: #d9e2ec;
      --accent: #0b7285;
      --low: #2f9e44;
      --med: #f08c00;
      --high: #d9480f;
      --critical: #c92a2a;
      --shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: \"Segoe UI\", \"Aptos\", \"Source Sans 3\", sans-serif;
      background:
        radial-gradient(circle at 12% 8%, #d6e6f0 0, transparent 32%),
        radial-gradient(circle at 88% 92%, #d9efe3 0, transparent 26%),
        var(--bg);
      color: var(--ink);
    }
    .shell {
      max-width: 1380px;
      margin: 0 auto;
      padding: 20px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 10px;
      margin-bottom: 14px;
    }
    h1 {
      margin: 0;
      font-size: 25px;
      letter-spacing: .2px;
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
      border-radius: 999px;
      background: var(--panel);
      box-shadow: var(--shadow);
      padding: 7px 12px;
      white-space: nowrap;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: var(--shadow);
      padding: 10px 12px;
    }
    .metric .k {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .7px;
      color: var(--muted);
    }
    .metric .v {
      margin-top: 4px;
      font-size: 30px;
      font-weight: 700;
      line-height: 1;
    }
    .layout {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: var(--shadow);
      padding: 12px;
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 15px;
      letter-spacing: .2px;
    }
    .stack { display: grid; gap: 12px; }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    select, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font-size: 13px;
      padding: 8px;
    }
    button {
      cursor: pointer;
      font-weight: 600;
      transition: .15s ease;
    }
    .btn-muted { background: #f8fafc; }
    .btn-muted:hover { background: #f1f5f9; }
    .btn-accent {
      background: var(--accent);
      color: #fff;
      border-color: transparent;
    }
    .btn-accent:hover { filter: brightness(.96); }
    .btn-danger {
      background: #ffe3e3;
      border-color: #ffa8a8;
      color: #a51111;
    }
    .queue, .events {
      display: grid;
      gap: 8px;
      max-height: 36vh;
      overflow: auto;
      padding-right: 2px;
    }
    .event-card {
      border: 1px solid var(--line);
      border-left: 5px solid var(--line);
      border-radius: 10px;
      background: #fff;
      padding: 10px;
    }
    .event-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 4px;
    }
    .track { font-weight: 700; font-size: 14px; }
    .stream { color: var(--muted); font-size: 12px; }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 7px;
    }
    .row-actions {
      display: flex;
      gap: 8px;
    }
    .row-actions button { width: auto; padding: 6px 9px; font-size: 12px; }
    .badge {
      font-size: 11px;
      font-weight: 700;
      color: #fff;
      border-radius: 999px;
      padding: 3px 8px;
    }
    .LOW { border-left-color: var(--low); }
    .MEDIUM { border-left-color: var(--med); }
    .HIGH { border-left-color: var(--high); }
    .CRITICAL { border-left-color: var(--critical); }
    .LOW .badge { background: var(--low); }
    .MEDIUM .badge { background: var(--med); }
    .HIGH .badge { background: var(--high); }
    .CRITICAL .badge { background: var(--critical); }
    .workspace { display: grid; gap: 12px; }
    .feed {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #111827;
      overflow: hidden;
      aspect-ratio: 16 / 9;
      position: relative;
    }
    .feed img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    .feed-badge {
      position: absolute;
      left: 10px;
      bottom: 10px;
      background: rgba(0,0,0,.55);
      color: #eaf2ff;
      border: 1px solid rgba(255,255,255,.22);
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
    }
    .feed-actions {
      position: absolute;
      right: 10px;
      top: 10px;
      display: flex;
      gap: 8px;
    }
    .feed-actions button {
      width: auto;
      border-color: rgba(255,255,255,.28);
      background: rgba(17,24,39,.62);
      color: #fff;
      padding: 6px 8px;
      font-size: 12px;
    }
    .empty {
      text-align: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 10px;
      padding: 30px 10px;
      background: #fff;
      font-size: 13px;
    }
    .ack {
      border-left-color: #94a3b8 !important;
      opacity: 0.78;
    }
    @media (max-width: 1060px) {
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .layout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"shell\">
    <div class=\"topbar\">
      <div>
        <h1>Clinical Risk Command Center</h1>
        <div class=\"subtitle\">Live stream triage with acknowledgement workflow and pose-based risk analytics.</div>
      </div>
      <div class=\"stamp\">Updated <span id=\"lastUpdate\">-</span></div>
    </div>

    <section class=\"metrics\">
      <article class=\"metric\"><div class=\"k\">Active Streams</div><div id=\"mStreams\" class=\"v\">0</div></article>
      <article class=\"metric\"><div class=\"k\">Active Tracks</div><div id=\"mTracks\" class=\"v\">0</div></article>
      <article class=\"metric\"><div class=\"k\">Open High Priority</div><div id=\"mOpen\" class=\"v\">0</div></article>
      <article class=\"metric\"><div class=\"k\">High Alerts</div><div id=\"mHigh\" class=\"v\">0</div></article>
      <article class=\"metric\"><div class=\"k\">Critical Alerts</div><div id=\"mCritical\" class=\"v\">0</div></article>
    </section>

    <section class=\"layout\">
      <aside class=\"stack\">
        <article class=\"panel\">
          <h2>Controls</h2>
          <label for=\"streamFilter\">Stream</label>
          <select id=\"streamFilter\"><option value=\"\">All Streams</option></select>

          <label for=\"levelFilter\">Minimum Severity</label>
          <select id=\"levelFilter\">
            <option value=\"LOW\">Low</option>
            <option value=\"MEDIUM\">Medium</option>
            <option value=\"HIGH\">High</option>
            <option value=\"CRITICAL\">Critical</option>
          </select>

          <div style=\"display:grid; gap:8px; margin-top:8px;\">
            <button id=\"btnSound\" class=\"btn-muted\">Alert Sound: Off</button>
            <button id=\"btnRefresh\" class=\"btn-accent\">Refresh Now</button>
          </div>
        </article>

        <article class=\"panel\">
          <h2>Triage Queue (Open)</h2>
          <div id=\"triageQueue\" class=\"queue\"></div>
        </article>
      </aside>

      <main class=\"workspace\">
        <article class=\"panel\">
          <h2>Live Camera + Pose</h2>
          <div class=\"feed\">
            <img id=\"streamFeed\" alt=\"Live stream\" />
            <div class=\"feed-badge\" id=\"streamState\">Waiting for stream...</div>
            <div class=\"feed-actions\">
              <button id=\"btnFullscreen\">Fullscreen</button>
            </div>
          </div>
        </article>

        <article class=\"panel\">
          <h2>Recent Events</h2>
          <div id=\"events\" class=\"events\"></div>
        </article>
      </main>
    </section>
  </div>

  <script>
    const rank = { LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3 };
    const mStreams = document.getElementById('mStreams');
    const mTracks = document.getElementById('mTracks');
    const mOpen = document.getElementById('mOpen');
    const mHigh = document.getElementById('mHigh');
    const mCritical = document.getElementById('mCritical');
    const lastUpdate = document.getElementById('lastUpdate');

    const streamFilter = document.getElementById('streamFilter');
    const levelFilter = document.getElementById('levelFilter');
    const streamFeed = document.getElementById('streamFeed');
    const streamState = document.getElementById('streamState');
    const btnRefresh = document.getElementById('btnRefresh');
    const btnSound = document.getElementById('btnSound');
    const btnFullscreen = document.getElementById('btnFullscreen');

    const triageQueue = document.getElementById('triageQueue');
    const eventsEl = document.getElementById('events');

    let alertsCache = [];
    let openCache = [];
    let soundOn = false;
    let lastCriticalTs = 0;

    function fmtTs(ts) {
      if (!ts) return '-';
      return new Date(ts * 1000).toLocaleTimeString();
    }

    function playBeep() {
      if (!soundOn) return;
      try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.value = 880;
        gain.gain.value = 0.06;
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.12);
      } catch (_) {}
    }

    async function ack(streamId, trackId, acknowledged) {
      const endpoint = acknowledged ? 'unack' : 'ack';
      await fetch(`/api/${endpoint}/${encodeURIComponent(streamId)}/${trackId}`, { method: 'POST' });
      await refreshAll();
    }

    function cardTemplate(row, showActions) {
      const e = row.event || {};
      const level = e.risk_level || 'LOW';
      const eventName = e.event || 'stable';
      const reasons = (e.reasons || []).length ? e.reasons.join(', ') : 'none';
      const cls = `${level} ${row.acknowledged ? 'ack' : ''}`;
      const ackLabel = row.acknowledged ? 'Unack' : 'Acknowledge';
      const sid = String(row.stream_id ?? '').replace(/'/g, "\\'");
      const action = showActions && rank[level] >= rank.HIGH
        ? `<button class=\"btn-muted\" onclick=\"window._ack('${sid}', ${e.track_id ?? -1}, ${row.acknowledged ? 'true' : 'false'})\">${ackLabel}</button>`
        : '';

      return `
        <article class=\"event-card ${cls}\">
          <div class=\"event-head\">
            <div>
              <div class=\"track\">Track #${e.track_id ?? '-'}</div>
              <div class=\"stream\">${row.stream_id ?? 'unknown stream'}</div>
            </div>
            <span class=\"badge\">${level}</span>
          </div>
          <div class=\"meta\">
            <span>event: ${eventName}</span>
            <span>confidence: ${(e.confidence ?? 0).toFixed(2)}</span>
            <span>time: ${fmtTs(e.timestamp)}</span>
          </div>
          <div class=\"meta\"><span>reason: ${reasons}</span></div>
          <div class=\"row-actions\">${action}</div>
        </article>
      `;
    }

    function renderQueue() {
      if (!openCache.length) {
        triageQueue.innerHTML = '<div class=\"empty\">No open high-priority alerts.</div>';
        return;
      }
      triageQueue.innerHTML = openCache.map((a) => cardTemplate(a, true)).join('');
    }

    function renderEvents() {
      if (!alertsCache.length) {
        eventsEl.innerHTML = '<div class=\"empty\">No events yet.</div>';
        return;
      }
      eventsEl.innerHTML = alertsCache.slice().reverse().map((a) => cardTemplate(a, true)).join('');
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
        mOpen.textContent = s.open_high_priority ?? 0;
        mHigh.textContent = s.high_alerts ?? 0;
        mCritical.textContent = s.critical_alerts ?? 0;
      } catch (_) {}
    }

    async function refreshAlerts() {
      try {
        const q = new URLSearchParams({ limit: '220' });
        if (streamFilter.value) q.set('stream_id', streamFilter.value);
        if (levelFilter.value) q.set('min_level', levelFilter.value);
        const res = await fetch('/api/alerts?' + q.toString());
        alertsCache = await res.json();
        renderEvents();
      } catch (_) {}
    }

    async function refreshOpenQueue() {
      try {
        const q = new URLSearchParams({ limit: '120', min_level: 'HIGH' });
        const res = await fetch('/api/open_alerts?' + q.toString());
        openCache = await res.json();
        if (streamFilter.value) {
          openCache = openCache.filter(a => a.stream_id === streamFilter.value);
        }
        renderQueue();

        const criticalNow = openCache
          .map(a => (a.event || {}))
          .filter(e => (e.risk_level || 'LOW') === 'CRITICAL')
          .map(e => e.timestamp || 0)
          .reduce((m, v) => Math.max(m, v), 0);

        if (criticalNow > lastCriticalTs) {
          lastCriticalTs = criticalNow;
          playBeep();
        }
      } catch (_) {}
    }

    async function refreshAll() {
      await refreshSummary();
      await refreshAlerts();
      await refreshOpenQueue();
      lastUpdate.textContent = new Date().toLocaleTimeString();
    }

    btnRefresh.addEventListener('click', refreshAll);
    btnSound.addEventListener('click', () => {
      soundOn = !soundOn;
      btnSound.textContent = `Alert Sound: ${soundOn ? 'On' : 'Off'}`;
    });
    btnFullscreen.addEventListener('click', async () => {
      const feed = document.querySelector('.feed');
      if (feed && feed.requestFullscreen) {
        await feed.requestFullscreen();
      }
    });

    streamFilter.addEventListener('change', async () => {
      updateFeed();
      await refreshAll();
    });
    levelFilter.addEventListener('change', refreshAlerts);

    window._ack = ack;

    async function boot() {
      await refreshStreams();
      await refreshAll();

      const es = new EventSource('/api/events');
      es.addEventListener('alert', async () => {
        await refreshAll();
      });

      setInterval(refreshStreams, 5000);
      setInterval(refreshAll, 3000);
    }

    boot();
  </script>
</body>
</html>
"""


def build_summary(alerts: list[dict[str, Any]], open_alerts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    streams = {a.get("stream_id", "unknown") for a in alerts}
    levels = Counter((a.get("event") or {}).get("risk_level", "LOW") for a in alerts)
    active_tracks = {
        (a.get("stream_id"), (a.get("event") or {}).get("track_id"))
        for a in alerts
        if (a.get("event") or {}).get("track_id") is not None
    }

    open_rows = open_alerts or []
    open_levels = Counter((a.get("event") or {}).get("risk_level", "LOW") for a in open_rows)

    return {
        "active_streams": len(streams),
        "active_tracks": len(active_tracks),
        "high_alerts": int(levels.get("HIGH", 0)),
        "critical_alerts": int(levels.get("CRITICAL", 0)),
        "open_high_priority": int(len(open_rows)),
        "open_high": int(open_levels.get("HIGH", 0)),
        "open_critical": int(open_levels.get("CRITICAL", 0)),
        "generated_at": time.time(),
    }
