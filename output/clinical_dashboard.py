from __future__ import annotations

import json
import queue
import time
from collections import Counter
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse


def create_dashboard_app(alert_manager) -> FastAPI:
    app = FastAPI(title="Clinical Risk Dashboard", version="3.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return alert_manager.get_health()

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

    @app.get("/api/privacy/{stream_id}")
    def privacy_status(stream_id: str) -> dict[str, Any]:
        return alert_manager.get_privacy_status(stream_id)

    @app.post("/api/privacy/{stream_id}")
    def set_privacy(stream_id: str, enabled: bool) -> dict[str, Any]:
        try:
            return alert_manager.set_privacy_enabled(stream_id, enabled)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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
  <title>Risk monitor</title>
  <style>
    :root {
      --bg: #f3f4f6;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #687076;
      --line: #d5d8dc;
      --accent: #1769aa;
      --low: #2e7d32;
      --med: #b26a00;
      --high: #c14600;
      --critical: #b3261e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, \"Segoe UI\", sans-serif;
      background: var(--bg);
      color: var(--ink);
      font-size: 13px;
    }
    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 14px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }
    .stamp {
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }
    .metrics {
      display: flex;
      flex-wrap: wrap;
      background: var(--panel);
      border: 1px solid var(--line);
      margin-bottom: 8px;
    }
    .metric {
      display: flex;
      align-items: baseline;
      gap: 7px;
      min-width: 125px;
      padding: 7px 10px;
      border-right: 1px solid var(--line);
    }
    .metric .k {
      font-size: 12px;
      color: var(--muted);
    }
    .metric .v {
      font-size: 17px;
      font-weight: 650;
      line-height: 1;
    }
    .toolbar {
      display: flex;
      align-items: end;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
      padding: 8px;
      background: var(--panel);
      border: 1px solid var(--line);
    }
    .field { min-width: 130px; }
    .toolbar-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-left: auto;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 8px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 9px;
    }
    .panel h2 {
      margin: 0 0 7px;
      font-size: 13px;
      font-weight: 650;
    }
    label {
      display: block;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 3px;
    }
    select, button {
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      min-height: 30px;
      padding: 5px 8px;
    }
    select { width: 100%; }
    button {
      cursor: pointer;
      font-weight: 550;
    }
    button:hover { background: #f1f3f4; }
    .btn-muted { background: #fff; }
    .btn-accent {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .btn-accent:hover { background: #12598f; }
    .btn-danger {
      background: #fce8e6;
      border-color: #e6aaa5;
      color: #8c1d18;
    }
    .queue, .events {
      display: grid;
      gap: 5px;
      max-height: 42vh;
      overflow: auto;
    }
    .queue { max-height: calc(100vh - 172px); }
    .event-card {
      border: 1px solid var(--line);
      border-left: 3px solid var(--line);
      background: #fff;
      padding: 7px 8px;
    }
    .event-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 3px;
    }
    .track { font-weight: 650; }
    .stream { color: var(--muted); font-size: 11px; }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 4px;
    }
    .meta > span + span::before { content: "·"; margin-right: 5px; }
    .row-actions { display: flex; gap: 5px; }
    .row-actions button { min-height: 25px; padding: 3px 7px; font-size: 11px; }
    .badge {
      font-size: 10px;
      font-weight: 650;
      letter-spacing: .3px;
    }
    .LOW { border-left-color: var(--low); }
    .MEDIUM { border-left-color: var(--med); }
    .HIGH { border-left-color: var(--high); }
    .CRITICAL { border-left-color: var(--critical); }
    .LOW .badge { color: var(--low); }
    .MEDIUM .badge { color: var(--med); }
    .HIGH .badge { color: var(--high); }
    .CRITICAL .badge { color: var(--critical); }
    .workspace { display: grid; gap: 8px; }
    .feed {
      border: 1px solid var(--line);
      background: #101214;
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
      left: 7px;
      bottom: 7px;
      background: rgba(0,0,0,.65);
      color: #fff;
      padding: 3px 6px;
      font-size: 11px;
    }
    .feed-actions {
      position: absolute;
      right: 7px;
      top: 7px;
      display: flex;
    }
    .feed-actions button {
      border-color: rgba(255,255,255,.28);
      background: rgba(0,0,0,.6);
      color: #fff;
      min-height: 27px;
      padding: 3px 7px;
      font-size: 11px;
    }
    .empty {
      color: var(--muted);
      padding: 12px 4px;
      font-size: 12px;
    }
    .ack {
      border-left-color: #94a3b8 !important;
      opacity: 0.78;
    }
    @media (max-width: 1060px) {
      .layout { grid-template-columns: 1fr; }
      .queue { max-height: 42vh; }
    }
    @media (max-width: 680px) {
      .shell { padding: 8px; }
      .field { flex: 1 1 120px; }
      .toolbar-actions { margin-left: 0; }
      .metric { min-width: 50%; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class=\"shell\">
    <div class=\"topbar\">
      <h1>Risk monitor</h1>
      <div class=\"stamp\">Updated <span id=\"lastUpdate\">-</span></div>
    </div>

    <section class=\"metrics\">
      <div class=\"metric\"><div class=\"k\">Streams</div><div id=\"mStreams\" class=\"v\">0</div></div>
      <div class=\"metric\"><div class=\"k\">Tracks</div><div id=\"mTracks\" class=\"v\">0</div></div>
      <div class=\"metric\"><div class=\"k\">Open</div><div id=\"mOpen\" class=\"v\">0</div></div>
      <div class=\"metric\"><div class=\"k\">High</div><div id=\"mHigh\" class=\"v\">0</div></div>
      <div class=\"metric\"><div class=\"k\">Critical</div><div id=\"mCritical\" class=\"v\">0</div></div>
    </section>

    <section class=\"toolbar\">
      <div class=\"field\">
        <label for=\"streamFilter\">Stream</label>
        <select id=\"streamFilter\"><option value=\"\">All</option></select>
      </div>
      <div class=\"field\">
        <label for=\"levelFilter\">Min severity</label>
        <select id=\"levelFilter\">
          <option value=\"LOW\">Low</option>
          <option value=\"MEDIUM\">Medium</option>
          <option value=\"HIGH\">High</option>
          <option value=\"CRITICAL\">Critical</option>
        </select>
      </div>
      <div class=\"field\">
        <label for=\"refreshMs\">Refresh</label>
        <select id=\"refreshMs\">
          <option value=\"1000\">1s</option>
          <option value=\"2000\">2s</option>
          <option value=\"3000\" selected>3s</option>
          <option value=\"5000\">5s</option>
        </select>
      </div>
      <div class=\"toolbar-actions\">
        <button id=\"btnOpenOnly\" class=\"btn-muted\">Open only: off</button>
        <button id=\"btnSound\" class=\"btn-muted\">Sound: off</button>
        <button id=\"btnPrivacy\" class=\"btn-muted\">Privacy: …</button>
        <button id=\"btnRefresh\" class=\"btn-accent\">Refresh</button>
      </div>
    </section>

    <section class=\"layout\">
      <main class=\"workspace\">
        <article class=\"panel\">
          <h2>Camera</h2>
          <div class=\"feed\">
            <img id=\"streamFeed\" alt=\"Live stream\" />
            <div class=\"feed-badge\" id=\"streamState\">No stream</div>
            <div class=\"feed-actions\">
              <button id=\"btnFullscreen\">Fullscreen</button>
            </div>
          </div>
        </article>

        <article class=\"panel\">
          <h2>Events</h2>
          <div id=\"events\" class=\"events\"></div>
        </article>
      </main>
      <aside class=\"panel\">
        <h2>Open alerts</h2>
        <div id=\"triageQueue\" class=\"queue\"></div>
      </aside>
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
    const btnOpenOnly = document.getElementById('btnOpenOnly');
    const btnSound = document.getElementById('btnSound');
    const btnPrivacy = document.getElementById('btnPrivacy');
    const btnFullscreen = document.getElementById('btnFullscreen');
    const refreshMsSel = document.getElementById('refreshMs');

    const triageQueue = document.getElementById('triageQueue');
    const eventsEl = document.getElementById('events');

    let alertsCache = [];
    let openCache = [];
    let soundOn = false;
    let openOnly = false;
    let lastCriticalTs = 0;
    let refreshTimerId = null;
    let privacyEnabled = true;

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
      const reasons = (e.reasons || []).length ? e.reasons.join(', ') : '—';
      const cls = `${level} ${row.acknowledged ? 'ack' : ''}`;
      const ackLabel = row.acknowledged ? 'Undo' : 'Ack';
      const sid = String(row.stream_id ?? '').replace(/'/g, "\\'");
      const action = showActions && rank[level] >= rank.HIGH
        ? `<button class=\"btn-muted\" onclick=\"window._ack('${sid}', ${e.track_id ?? -1}, ${row.acknowledged ? 'true' : 'false'})\">${ackLabel}</button>`
        : '';

      return `
        <article class=\"event-card ${cls}\">
          <div class=\"event-head\">
            <div>
              <div class=\"track\">Track ${e.track_id ?? '-'}</div>
              <div class=\"stream\">${row.stream_id ?? '—'}</div>
            </div>
            <span class=\"badge\">${level}</span>
          </div>
          <div class=\"meta\">
            <span>${eventName}</span>
            <span>${(e.confidence ?? 0).toFixed(2)}</span>
            <span>${fmtTs(e.timestamp)}</span>
          </div>
          <div class=\"meta\"><span>${reasons}</span></div>
          <div class=\"row-actions\">${action}</div>
        </article>
      `;
    }

    function renderQueue() {
      if (!openCache.length) {
        triageQueue.innerHTML = '<div class=\"empty\">Clear</div>';
        return;
      }
      triageQueue.innerHTML = openCache.map((a) => cardTemplate(a, true)).join('');
    }

    function renderEvents() {
      let rows = alertsCache;
      if (openOnly) {
        rows = alertsCache.filter(a => {
          const e = a.event || {};
          const lvl = e.risk_level || 'LOW';
          return !a.acknowledged && rank[lvl] >= rank.HIGH;
        });
      }
      if (!rows.length) {
        eventsEl.innerHTML = '<div class=\"empty\">No events</div>';
        return;
      }
      eventsEl.innerHTML = rows.slice().reverse().map((a) => cardTemplate(a, true)).join('');
    }

    function updateFeed() {
      const sid = streamFilter.value;
      if (!sid) {
        streamFeed.removeAttribute('src');
        streamState.textContent = 'No stream';
        return;
      }
      streamFeed.src = `/api/stream/${encodeURIComponent(sid)}.mjpg?fps=12&t=${Date.now()}`;
      streamState.textContent = sid;
    }

    async function refreshStreams() {
      try {
        const res = await fetch('/api/streams');
        const data = await res.json();
        const streams = data.streams || [];
        const current = streamFilter.value;
        streamFilter.innerHTML = '<option value="">All</option>' + streams.map(s => `<option value="${s}">${s}</option>`).join('');

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

    async function refreshPrivacy() {
      const sid = streamFilter.value;
      if (!sid) {
        btnPrivacy.disabled = true;
        btnPrivacy.textContent = 'Privacy: n/a';
        return;
      }
      try {
        const res = await fetch(`/api/privacy/${encodeURIComponent(sid)}`);
        const state = await res.json();
        privacyEnabled = Boolean(state.enabled);
        btnPrivacy.disabled = !state.toggle_allowed || state.configured_mode === 'none';
        btnPrivacy.textContent = privacyEnabled ? 'Privacy: on' : 'Privacy: OFF';
        btnPrivacy.style.background = privacyEnabled ? '' : '#c92a2a';
        btnPrivacy.style.color = privacyEnabled ? '' : '#fff';
        streamState.textContent = privacyEnabled ? sid : `${sid} · PRIVACY OFF`;
      } catch (_) {
        btnPrivacy.disabled = true;
        btnPrivacy.textContent = 'Privacy: n/a';
      }
    }

    async function togglePrivacy() {
      const sid = streamFilter.value;
      if (!sid) return;
      const next = !privacyEnabled;
      if (!next && !window.confirm('Disable anonymization for this session?')) return;
      const res = await fetch(`/api/privacy/${encodeURIComponent(sid)}?enabled=${next}`, { method: 'POST' });
      if (res.ok) await refreshPrivacy();
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
      await refreshPrivacy();
      lastUpdate.textContent = new Date().toLocaleTimeString();
    }

    function resetAutoRefresh() {
      if (refreshTimerId !== null) {
        clearInterval(refreshTimerId);
      }
      const ms = Math.max(700, Number(refreshMsSel.value || 3000));
      refreshTimerId = setInterval(async () => {
        await refreshAll();
      }, ms);
    }

    btnRefresh.addEventListener('click', refreshAll);
    btnOpenOnly.addEventListener('click', () => {
      openOnly = !openOnly;
      btnOpenOnly.textContent = `Open only: ${openOnly ? 'on' : 'off'}`;
      renderEvents();
    });
    btnSound.addEventListener('click', () => {
      soundOn = !soundOn;
      btnSound.textContent = `Sound: ${soundOn ? 'on' : 'off'}`;
    });
    btnPrivacy.addEventListener('click', togglePrivacy);
    refreshMsSel.addEventListener('change', resetAutoRefresh);
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
      resetAutoRefresh();
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
