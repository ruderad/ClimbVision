"""The interactive viewer: `viewer.html` + `viewer.mp4` in a results folder.

`climbvision view <results>` writes the clip as a browser-playable video (no drawing on
it) and one HTML file that plays it with the scene layers (`scene.py`) drawn live on
top. Layers switch on and off instantly, presets pick a set, and there's slow motion,
frame-by-frame stepping and a timeline under the video (click to jump; hover a move for
its details). The file works offline: open it in any browser; keep it next to
`viewer.mp4`.

Shortcuts: space play / pause · ← → one frame · shift + ← → one second ·
1–6 presets · [ ] slower / faster.
"""

from __future__ import annotations

from pathlib import Path

from climbvision.output.scene import Scene, build_scene


def write_viewer(video_path: str | Path, result_dir: str | Path,
                 scene: Scene | None = None, video: bool = True,
                 watermark: bool = True) -> Path:
    """viewer.html (+ viewer.mp4 unless `video=False` and it exists) in result_dir.
    The watermark is in viewer.mp4 itself, so the viewer and its exports carry it."""
    from climbvision.engine.video import VideoReader, VideoWriter, warn_if_changed
    from climbvision.output.watermark import apply_watermark

    result_dir = Path(result_dir)
    scene = scene or build_scene(result_dir)
    mp4 = result_dir / "viewer.mp4"
    if video or not mp4.exists():
        warn_if_changed(result_dir, video_path)
        import cv2

        with VideoReader(video_path) as reader:
            size = tuple(scene.size) if scene.size and scene.size[0] else (
                reader.info.width, reader.info.height)
            resize = size != (reader.info.width, reader.info.height)
            with VideoWriter(mp4, scene.fps, size, crf=18, audio_from=video_path) as w:
                if not w.browser_ready:
                    print("[climbvision] no ffmpeg found: viewer.mp4 is MPEG-4 and won't play "
                          "in a browser (pip install imageio-ffmpeg)")
                for i, (_, _t, img) in enumerate(reader.frames(step=scene.step)):
                    if i >= len(scene.t):
                        break
                    if resize:
                        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    if watermark:
                        apply_watermark(img)
                    w.write(img)
    data = scene.to_json().replace("</", "<\\/")
    muxer = (Path(__file__).parent / "vendor" / "mp4-muxer.min.js").read_text(encoding="utf-8")
    html = (TEMPLATE.replace("__TITLE__", f"{scene.name} — ClimbVision")
            .replace("__MUXER__", muxer).replace("__DATA__", data))
    out = result_dir / "viewer.html"
    out.write_text(html, encoding="utf-8")
    return out


def _range_handler():
    import http.server
    import re

    class Handler(http.server.SimpleHTTPRequestHandler):
        """Static files with byte ranges (Safari won't play a video without them)."""

        def log_message(self, *a, **k):  # quiet
            pass

        def send_head(self):
            rng = self.headers.get("Range")
            path = Path(self.translate_path(self.path))
            m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
            if not m or not path.is_file():
                return super().send_head()
            size = path.stat().st_size
            a = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2) or 0))
            b = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
            b = min(b, size - 1)
            if a > b:
                self.send_error(416)
                return None
            f = path.open("rb")
            f.seek(a)
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.send_header("Content-Length", str(b - a + 1))
            self.end_headers()
            self._remaining = b - a + 1
            return f

        def copyfile(self, source, outputfile):
            left = getattr(self, "_remaining", None)
            if left is None:
                return super().copyfile(source, outputfile)
            while left > 0:
                chunk = source.read(min(1 << 16, left))
                if not chunk:
                    break
                outputfile.write(chunk)
                left -= len(chunk)
            self._remaining = None

    return Handler


_RangeHandler = _range_handler()


def serve(pages: list[Path], port: int = 0) -> None:
    """Serve the folder holding the viewers on localhost, open each, until Ctrl+C. Over
    http the page and its video share an origin, so export reads the video directly."""
    import functools
    import http.server
    import os
    import threading
    import webbrowser

    root = Path(os.path.commonpath([str(p.parent) for p in pages]))
    handler = functools.partial(_RangeHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    for p in pages:
        webbrowser.open(url + p.relative_to(root).as_posix())
    print(f"[climbvision] serving {root} at {url} — Ctrl+C to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#111214;--panel:#1b1d21;--line:#2c2f35;--text:#e8e8ea;--dim:#9a9ca3;--accent:#5ad2ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0 8px 0 0;font-weight:600}
.group{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.group .lbl{color:var(--dim);font-size:12px;margin-right:2px}
button,.chip{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 10px;font:inherit;cursor:pointer}
button:hover,.chip:hover{border-color:#4a4e57}
button.on,.chip.on{background:#23384a;border-color:var(--accent);color:#fff}
.chip input{display:none}
main{display:flex;flex-direction:column;align-items:center;padding:12px 16px;gap:10px}
#stage{position:relative;max-width:100%;background:#000;line-height:0}
#stage video,#stage canvas{display:block;max-width:100%;max-height:calc(100vh - 340px);width:auto;height:auto}
#stage canvas{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
#controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:center}
#time{font-variant-numeric:tabular-nums;color:var(--dim);min-width:120px;text-align:center}
#tlwrap{width:100%;max-width:1100px;position:relative}
#tl{width:100%;height:auto;display:block;cursor:pointer;background:var(--panel);border-radius:6px}
#tip{position:absolute;pointer-events:none;background:#000d;border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:12px;display:none;white-space:nowrap}
.help{color:var(--dim);font-size:12px;text-align:center}
#export{background:#23384a;border-color:var(--accent)}
dialog{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:18px 20px;max-width:min(520px,92vw)}
dialog::backdrop{background:#0009}
dialog h2{font-size:16px;margin:0 0 12px}
dialog .row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
dialog .row > span:first-child{min-width:92px;color:var(--dim)}
dialog input[type=number]{width:84px;background:#111;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 6px;font:inherit}
dialog select{background:#111;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 6px;font:inherit}
dialog .note{color:var(--dim);font-size:12px;margin:8px 0}
dialog progress{width:100%;height:10px}
dialog code{display:block;white-space:pre-wrap;word-break:break-all;background:#111;border:1px solid var(--line);border-radius:6px;padding:8px;font-size:12px;margin-top:6px}
.actions{display:flex;gap:8px;justify-content:flex-end;margin-top:14px;flex-wrap:wrap}
@media (max-width:600px){header{padding:8px}h1{width:100%}}
</style></head>
<body>
<header>
  <h1 id="name"></h1>
  <div class="group" id="presets"><span class="lbl">Preset</span></div>
  <div class="group" id="layers"><span class="lbl">Layers</span></div>
</header>
<main>
  <div id="stage"><video id="v" src="viewer.mp4" playsinline preload="auto"></video><canvas id="c"></canvas></div>
  <div id="controls">
    <button id="back1" title="one frame back (←)">◀︎ frame</button>
    <button id="play" title="space">Play</button>
    <button id="fwd1" title="one frame on (→)">frame ▶︎</button>
    <span class="group" id="speeds"><span class="lbl">Speed</span></span>
    <span id="time"></span>
    <button id="export" title="save the clip with the layers you see as an MP4">Export MP4…</button>
  </div>
  <div id="tlwrap"><canvas id="tl"></canvas><div id="tip"></div></div>
  <div class="help">space play/pause · ← → frame · shift+← → 1 s · [ ] speed · 1–6 presets · click the timeline to jump</div>
</main>
<dialog id="xd">
  <h2>Export the clip with these overlays</h2>
  <div class="note" id="xlayers"></div>
  <div class="row"><span>From</span><input type="number" id="xa" step="0.05" min="0"> s <button id="xa_now">now</button>
    <span style="min-width:auto;color:var(--dim)">to</span><input type="number" id="xb" step="0.05" min="0"> s <button id="xb_now">now</button>
    <button id="xall">whole clip</button></div>
  <div class="row"><span>Size</span><select id="xsize"><option value="1">original</option><option value="1080">1080p</option><option value="720">720p</option></select></div>
  <div class="row"><span>Speed</span><select id="xspeed"><option value="1">real time</option><option value="0.5">0.5× (slow motion)</option><option value="0.25">0.25×</option></select></div>
  <div class="row"><span>Timeline</span><label><input type="checkbox" id="xtl"> strip under the video</label></div>
  <div class="row"><span>Sound</span><label><input type="checkbox" id="xsnd" checked> the clip's sound (real time only)</label></div>
  <div id="xprog" style="display:none"><progress id="xbar" max="1" value="0"></progress><div class="note" id="xmsg"></div></div>
  <div class="note" id="xerr"></div>
  <details class="note"><summary>…or from the terminal</summary><code id="xcmd"></code></details>
  <div class="actions"><input type="file" id="xfile" accept="video/mp4" style="display:none">
    <button id="xpick" style="display:none">Choose viewer.mp4…</button>
    <button id="xcancel">Close</button><button id="xgo" class="on">Export</button></div>
</dialog>
<script>__MUXER__</script>
<script type="application/json" id="scene">__DATA__</script>
<script>
const S = JSON.parse(document.getElementById('scene').textContent);
const v = document.getElementById('v'), c = document.getElementById('c'), ctx = c.getContext('2d');
const tl = document.getElementById('tl'), tctx = tl.getContext('2d'), tip = document.getElementById('tip');
const ORDER = ["support","trail","contacts","effort","forces","skeleton","arms","angles","arc","swing","com","speed","velocity","info"];
const LABELS = {skeleton:"Skeleton",arms:"Bent arms",angles:"Joint angles",contacts:"Holds",support:"Base of support",com:"COM (balance)",
  trail:"COM trail",speed:"COM (speed)",forces:"Forces",effort:"Joint effort",velocity:"Velocity",arc:"Dyno arc",swing:"Swing",info:"Info text",timeline:"Timeline"};
const HELP = {skeleton:"bones, spine, joints; grey = hidden or guessed",arms:"holding arms orange when bent",
  contacts:"rings: limbs on holds; dashed: smears (bare wall, friction); grey flag?: still, not touching",support:"hull of the limbs on holds; plumb line from the COM",
  com:"green inside the base of support, red outside",trail:"COM path, last 2 s",
  forces:"load on each hand / foot, % body weight; arrow = force the hold gives the body; hollow ~ = on a smear (estimate)",
  velocity:"where the COM is heading (0.25 s ahead)",info:"time, contacts, latest move",
  timeline:"contacts per limb, moves (red = dyno), rests (green), feet cut (red), move phases, load",
  angles:"elbow, hip, knee angles (180 = straight); bent holding elbows orange",
  speed:"COM coloured by speed: blue still, white 1 m/s, red 2 m/s",
  effort:"how hard each joint works (net joint moment, Nm/kg): bigger and redder = harder; core at the hips",
  arc:"dynamic moves: free-flight path from the peak speed, the deadpoint, and when the hand caught",
  swing:"feet off: how fast the body turns about the hands"};
const STORE = "climbvision.viewer.layers";
let on = new Set(S.presets[S.default_preset]);
try { const s = JSON.parse(localStorage.getItem(STORE)); if (Array.isArray(s)) on = new Set(s); } catch (e) {}
document.getElementById('name').textContent = S.name;
document.title = S.name + " — ClimbVision";
const W = S.size[0], H = S.size[1], unit = Math.max(W, H) / 1000, N = S.t.length;
c.width = W; c.height = H;

// ---------- controls
const presetsEl = document.getElementById('presets'), layersEl = document.getElementById('layers');
const pnames = Object.keys(S.presets);
pnames.forEach((p, k) => { const b = document.createElement('button'); b.textContent = p; b.title = (k+1) + ": " + S.presets[p].join(", ");
  b.onclick = () => { on = new Set(S.presets[p]); sync(); }; b.dataset.p = p; presetsEl.appendChild(b); });
S.layers.forEach(name => { const l = document.createElement('label'); l.className = 'chip'; l.title = HELP[name] || "";
  l.innerHTML = '<input type="checkbox">' + (LABELS[name] || name); l.dataset.n = name;
  l.onclick = (e) => { e.preventDefault(); on.has(name) ? on.delete(name) : on.add(name); sync(); };
  layersEl.appendChild(l); });
const speedsEl = document.getElementById('speeds'), SPEEDS = [0.1, 0.25, 0.5, 1];
SPEEDS.forEach(s => { const b = document.createElement('button'); b.textContent = s + "×"; b.dataset.s = s;
  b.onclick = () => { v.playbackRate = s; sync(); }; speedsEl.appendChild(b); });
function sync() {
  try { localStorage.setItem(STORE, JSON.stringify([...on])); } catch (e) {}
  for (const l of layersEl.querySelectorAll('.chip')) l.classList.toggle('on', on.has(l.dataset.n));
  for (const b of presetsEl.querySelectorAll('button')) {
    const s = new Set(S.presets[b.dataset.p]); b.classList.toggle('on', s.size === on.size && [...s].every(x => on.has(x))); }
  for (const b of speedsEl.querySelectorAll('button')) b.classList.toggle('on', +b.dataset.s === v.playbackRate);
  document.getElementById('tlwrap').style.display = on.has('timeline') && S.timeline.rows.length ? '' : 'none';
  draw(); }
const playBtn = document.getElementById('play');
playBtn.onclick = () => v.paused ? v.play() : v.pause();
v.onplay = () => playBtn.textContent = "Pause"; v.onpause = () => playBtn.textContent = "Play";
v.onclick = () => v.paused ? v.play() : v.pause();
function stepFrames(k) { v.pause(); const i = Math.min(N - 1, Math.max(0, frameIndex() + k)); v.currentTime = (i + 0.5) / S.fps; }
document.getElementById('back1').onclick = () => stepFrames(-1);
document.getElementById('fwd1').onclick = () => stepFrames(1);
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === ' ') { e.preventDefault(); v.paused ? v.play() : v.pause(); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); stepFrames(e.shiftKey ? -Math.round(S.fps) : -1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); stepFrames(e.shiftKey ? Math.round(S.fps) : 1); }
  else if (e.key === '[' || e.key === ']') { let k = SPEEDS.indexOf(v.playbackRate); k = Math.max(0, Math.min(SPEEDS.length - 1, k + (e.key === ']' ? 1 : -1)));
    v.playbackRate = SPEEDS[k < 0 ? 3 : k]; sync(); }
  else if (/^[1-9]$/.test(e.key) && pnames[+e.key - 1]) { on = new Set(S.presets[pnames[+e.key - 1]]); sync(); }
});

// ---------- drawing the scene shapes (same meaning as render.py)
let mediaTime = 0;
function frameIndex() { return Math.max(0, Math.min(N - 1, Math.floor(mediaTime * S.fps + 1e-3))); }
function lw(w) { return Math.max(1, w * unit); }
function shape(g, s, info) {
  const k = s[0];
  g.setLineDash([]); g.globalAlpha = 1;
  if (k === "l") { const [, x0, y0, x1, y1, col, w, dashed] = s; g.strokeStyle = col; g.lineWidth = lw(w);
    if (dashed) g.setLineDash([6 * unit, 6 * unit]); g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); }
  else if (k === "p" || k === "g") { const xy = s[1]; g.beginPath(); g.moveTo(xy[0], xy[1]);
    for (let j = 2; j < xy.length; j += 2) g.lineTo(xy[j], xy[j + 1]);
    if (k === "g") { g.closePath(); if (s[4] > 0) { g.globalAlpha = s[4]; g.fillStyle = s[2]; g.fill(); g.globalAlpha = 1; } }
    g.strokeStyle = s[2]; g.lineWidth = lw(s[3]); g.lineCap = "round"; g.lineJoin = "round"; g.stroke(); }
  else if (k === "q") { const [, xy, cols, w] = s; g.lineWidth = lw(w); g.lineCap = "round";
    for (let j = 0; j < cols.length; j++) { g.strokeStyle = cols[j]; g.beginPath(); g.moveTo(xy[2*j], xy[2*j+1]); g.lineTo(xy[2*j+2], xy[2*j+3]); g.stroke(); } }
  else if (k === "c") { const [, x, y, r, col, w, fill] = s; g.beginPath(); g.arc(x, y, Math.max(2, r * unit), 0, 2 * Math.PI);
    if (fill) { g.globalAlpha = Math.min(1, fill); g.fillStyle = col; g.fill(); g.globalAlpha = 1; }
    if (w) { g.strokeStyle = col; g.lineWidth = lw(w); g.stroke(); } }
  else if (k === "d") { const [, x, y, r, col, w] = s; g.strokeStyle = col; g.lineWidth = lw(w);
    const rr = Math.max(2, r * unit); g.setLineDash([rr * 0.44, rr * 0.35]); g.beginPath(); g.arc(x, y, rr, 0, 2 * Math.PI); g.stroke(); }
  else if (k === "a") { const [, x0, y0, x1, y1, col, w] = s; const L = Math.hypot(x1 - x0, y1 - y0); if (L < 2) return;
    g.strokeStyle = col; g.fillStyle = col; g.lineWidth = lw(w); g.lineCap = "round";
    const h = Math.min(0.4 * L, 18 * unit), a = Math.atan2(y1 - y0, x1 - x0);
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1 - h * 0.6 * Math.cos(a), y1 - h * 0.6 * Math.sin(a)); g.stroke();
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - h * Math.cos(a - 0.45), y1 - h * Math.sin(a - 0.45));
    g.lineTo(x1 - h * Math.cos(a + 0.45), y1 - h * Math.sin(a + 0.45)); g.closePath(); g.fill(); }
  else if (k === "t") { const [, x, y, text, col, size] = s; g.font = `600 ${Math.round(22 * size * unit)}px system-ui,sans-serif`;
    const m = g.measureText(text), hgt = 22 * size * unit, pad = hgt * 0.25;
    g.globalAlpha = 0.55; g.fillStyle = "#000"; g.fillRect(x - pad, y - hgt - pad * 0.2, m.width + 2 * pad, hgt + 1.4 * pad);
    g.globalAlpha = 1; g.fillStyle = col; g.fillText(text, x, y); }
  else if (k === "T" && info) info.push(s);
}
function infoBox(g, lines) {
  if (!lines.length) return; lines.sort((a, b) => a[1] - b[1]);
  const fs = Math.round(26 * unit), lh = fs * 1.35, x = 15 * unit + 5, y0 = 40 * unit + 10, pad = 8 * unit;
  g.font = `600 ${fs}px system-ui,sans-serif`;
  const tw = Math.max(...lines.map(s => g.measureText(s[2]).width));
  g.globalAlpha = 0.8; g.fillStyle = "#000"; g.fillRect(x - pad, y0 - fs - pad, tw + 2 * pad, lh * lines.length + pad);
  g.globalAlpha = 1; lines.forEach((s, k) => { g.fillStyle = s[3]; g.fillText(s[2], x, y0 + k * lh); });
}
function paintOverlays(g, i) {
  const info = [];
  for (const name of ORDER) if (on.has(name) && S.frames[name]) for (const s of S.frames[name][i]) shape(g, s, info);
  infoBox(g, info);
}
function draw() {
  const i = frameIndex();
  ctx.clearRect(0, 0, W, H);
  paintOverlays(ctx, i);
  document.getElementById('time').textContent = `${(S.t[i] ?? 0).toFixed(2)} s · frame ${i + 1}/${N}`;
  drawTimeline(S.t[i] ?? 0);
}

// ---------- timeline
const TL = S.timeline, rowH = 22, T0 = 8, left = 70, right = 10;
const LOADH = TL.load ? 46 : 0, PH = (TL.phases && TL.phases.length) ? rowH : 0;
const PHC = {push: "#ff8c00", reach: "#5ad2ff", catch: "#ff5050"};
const tlH = T0 + rowH * Math.max(1, TL.rows.length) + PH + LOADH + 24;
function tlSize() { const w = tl.clientWidth || 800, dpr = window.devicePixelRatio || 1;
  tl.width = w * dpr; tl.height = tlH * dpr; tl.style.height = tlH + "px"; tctx.setTransform(dpr, 0, 0, dpr, 0, 0); return w; }
let tlW = 800;
function X(s) { return left + (tlW - left - right) * s / Math.max(TL.duration, 1e-6); }
function drawTimeline(t) { if (TL.rows.length) { tctx.clearRect(0, 0, tlW, tlH); paintTimeline(tctx, tlW, t); } }
function paintTimeline(g, w, t) {
  const X = s => left + (w - left - right) * s / Math.max(TL.duration, 1e-6);
  const y1 = T0 + rowH * TL.rows.length;
  for (const [a, b, kind] of TL.bands || []) { g.fillStyle = kind === "rest" ? "#3c8c3c55" : "#c8323255"; g.fillRect(X(a), T0, Math.max(1, X(b) - X(a)), y1 - T0); }
  g.font = "12px system-ui,sans-serif";
  TL.rows.forEach((row, r) => { const y = T0 + r * rowH; g.fillStyle = "#9a9ca3"; g.fillText(row.name, 6, y + 15);
    for (const [a, b, kind] of row.spans) { g.globalAlpha = kind === "smear" ? 0.4 : 1;
      g.fillStyle = kind === "ground" ? "#555" : row.colour; g.fillRect(X(a), y + 4, Math.max(1, X(b) - X(a)), rowH - 8); }
    g.globalAlpha = 1; });
  const idx = Object.fromEntries(TL.rows.map((r, k) => [r.limb, k]));
  for (const m of TL.moves || []) { const k = idx[m.limb]; if (k === undefined) continue; const y = T0 + k * rowH + rowH / 2;
    const hot = m.tags.includes("dyno") || m.tags.includes("jump"); g.strokeStyle = hot ? "#ff5050" : "#eee"; g.lineWidth = 2;
    g.beginPath(); g.moveTo(X(m.a), y); g.lineTo(Math.max(X(m.a) + 1, X(m.b)), y); g.stroke();
    g.fillStyle = hot ? "#ff5050" : "#ddd"; g.fillText(m.n, X(m.b) + 2, y - 3); }
  if (PH) { g.fillStyle = "#9a9ca3"; g.fillText("phase", 6, y1 + 15);
    for (const [a, b, name] of TL.phases) { g.fillStyle = PHC[name] || "#999"; g.fillRect(X(a), y1 + 6, Math.max(1, X(b) - X(a)), rowH - 12); } }
  if (TL.load) {  // load on the hands and the feet, % body weight (100 % dotted)
    const ly0 = y1 + PH + 4, lh = LOADH - 8, top = 150, n = S.t.length;
    const Y = v => ly0 + lh - lh * Math.min(v, top) / top;
    g.strokeStyle = "#666"; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(left, Y(100)); g.lineTo(w - right, Y(100)); g.stroke(); g.setLineDash([]);
    for (const [key, col] of [["hands", "#b07aff"], ["feet", "#3cd25a"]]) { const arr = TL.load[key]; g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath(); let pen = false;
      for (let i = 0; i < n; i++) { const val = arr[i]; if (val == null) { pen = false; continue; } const x = X(S.t[i]), y = Y(val); pen ? g.lineTo(x, y) : g.moveTo(x, y); pen = true; } g.stroke(); }
    g.fillStyle = "#9a9ca3"; g.fillText("load", 6, ly0 + 14); g.fillStyle = "#b07aff"; g.fillText("hands", 6, ly0 + 28); g.fillStyle = "#3cd25a"; g.fillText("feet", 6, ly0 + 40);
  }
  // seconds
  const y2 = y1 + PH + LOADH;
  g.fillStyle = "#777"; const stepS = TL.duration > 60 ? 10 : TL.duration > 20 ? 5 : 1;
  for (let s = 0; s <= TL.duration; s += stepS) { g.fillRect(X(s), y2 + 2, 1, 5); g.fillText(s + "s", X(s) + 2, y2 + 16); }
  g.fillStyle = "#fff"; g.fillRect(X(t) - 1, 2, 2, tlH - 4);
}
function seekFromEvent(e) { const r = tl.getBoundingClientRect(); const s = (e.clientX - r.left - left) / (tlW - left - right) * TL.duration;
  v.currentTime = Math.max(0, Math.min(TL.duration - 1e-3, s)); mediaTime = v.currentTime; draw(); }
let dragging = false;
tl.addEventListener('pointerdown', e => { dragging = true; tl.setPointerCapture(e.pointerId); seekFromEvent(e); });
tl.addEventListener('pointermove', e => { if (dragging) seekFromEvent(e); hover(e); });
tl.addEventListener('pointerup', () => dragging = false);
tl.addEventListener('pointerleave', () => tip.style.display = 'none');
function hover(e) {
  const r = tl.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const k = Math.floor((y - T0) / rowH), row = TL.rows[k]; let best = null;
  const yl = T0 + rowH * TL.rows.length;
  if (PH && y > yl && y < yl + PH) { const s = (x - left) / (tlW - left - right) * TL.duration;
    const ph = TL.phases.find(([a, b]) => s >= a && s <= b);
    if (ph) { tip.innerHTML = `<b>${ph[2]}</b> ${(ph[1] - ph[0]).toFixed(2)} s`; tip.style.display = 'block';
      tip.style.left = Math.min(x + 12, r.width - 200) + 'px'; tip.style.top = (y + 12) + 'px'; return; } }
  if (TL.load && y > yl + PH && y < yl + PH + LOADH) { const s = (x - left) / (tlW - left - right) * TL.duration;
    let i = 0; while (i < S.t.length - 1 && S.t[i] < s) i++; const h = TL.load.hands[i], f = TL.load.feet[i];
    if (h != null) { tip.innerHTML = `${S.t[i].toFixed(2)} s<br>hands ${h}% · feet ${f}% body weight`; tip.style.display = 'block';
      tip.style.left = Math.min(x + 12, r.width - 200) + 'px'; tip.style.top = (y + 12) + 'px'; return; } }
  if (row) for (const m of TL.moves || []) if (m.limb === row.limb && x >= X(m.a) - 4 && x <= X(m.b) + 12) best = m;
  if (!best) { tip.style.display = 'none'; return; }
  tip.innerHTML = `<b>move ${best.n}</b> · ${row.name} · ${(best.b - best.a).toFixed(2)} s` + (best.tags.length ? `<br>${best.tags.join(", ")}` : "") +
    (best.load != null ? `<br>catch load ${best.load}% body weight` : "");
  tip.style.display = 'block'; tip.style.left = Math.min(x + 12, r.width - 200) + 'px'; tip.style.top = (y + 12) + 'px';
}

// ---------- the clock: follow the video's frames
function tick(now, meta) { mediaTime = meta ? meta.mediaTime : v.currentTime; draw();
  if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(tick); }
if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(tick);
else (function loop() { mediaTime = v.currentTime; draw(); requestAnimationFrame(loop); })();
v.addEventListener('seeked', () => { mediaTime = v.currentTime; draw(); });
v.addEventListener('loadeddata', () => { mediaTime = v.currentTime; draw(); });
window.addEventListener('resize', () => { tlW = tlSize(); draw(); });
tlW = tlSize(); v.playbackRate = 1; sync();

// ---------- export: the frames you see, with the layers you chose, as an MP4
// Drawn frame by frame (seek, draw the video and the layers, encode with WebCodecs),
// so it's exact, not a screen recording. Chrome, Edge and Safari 16.4+.
const xd = document.getElementById('xd'), $ = id => document.getElementById(id);
let exporting = false, cancelled = false;
const lastPreset = () => { for (const p of pnames) { const s = new Set(S.presets[p]); if (s.size === on.size && [...s].every(x => on.has(x))) return p; } return null; };
function layerList() { return S.layers.filter(n => on.has(n)); }
function cliCommand() {
  const p = lastPreset(), base = `climbvision render outputs/${S.name}`;
  return p ? `${base} --preset ${p}` : `${base} --layers ${layerList().join(",")} --out annotated_custom.mp4`;
}
$('export').onclick = () => {
  v.pause(); $('xlayers').textContent = "Layers: " + (layerList().map(n => LABELS[n] || n).join(", ") || "none (the plain clip)");
  if (!$('xa').value) { $('xa').value = 0; $('xb').value = TL.duration ? TL.duration.toFixed(2) : (N / S.fps).toFixed(2); }
  $('xtl').checked = on.has('timeline') && TL.rows.length > 0; $('xtl').disabled = !TL.rows.length;
  $('xcmd').textContent = cliCommand(); $('xerr').textContent = ""; $('xprog').style.display = "none";
  if (!window.VideoEncoder) $('xerr').textContent = "This browser can't encode video (needs Chrome, Edge or Safari 16.4+). Use the terminal command below.";
  xd.showModal(); };
$('xa_now').onclick = () => $('xa').value = v.currentTime.toFixed(2);
$('xb_now').onclick = () => $('xb').value = v.currentTime.toFixed(2);
$('xall').onclick = () => { $('xa').value = 0; $('xb').value = (N / S.fps).toFixed(2); };
$('xcancel').onclick = () => { if (exporting) cancelled = true; else xd.close(); };
$('xpick').onclick = () => $('xfile').click();
$('xfile').onchange = () => { const f = $('xfile').files[0]; if (!f) return; const t = v.currentTime;
  v.src = URL.createObjectURL(f); v.addEventListener('loadeddata', () => { v.currentTime = t; $('xpick').style.display = "none";
    $('xerr').textContent = "Loaded. Press Export again."; }, {once: true}); };
// seek and wait until the frame is there: 'seeked', then the next presented frame when the
// browser reports it (requestVideoFrameCallback), with a short fallback in case it doesn't
const seekTo = t => new Promise(res => {
  let ok = false; const fin = () => { if (!ok) { ok = true; res(); } };
  v.addEventListener('seeked', () => { if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(fin); setTimeout(fin, 120); }, {once: true});
  if (Math.abs(v.currentTime - t) < 1e-4) fin(); else v.currentTime = t; });
async function pickCodec(w, h, fps, bitrate) {
  for (const [codec, mux] of [["avc1.640033", "avc"], ["avc1.4d0033", "avc"], ["avc1.42003e", "avc"], ["vp09.00.40.08", "vp9"]]) {
    const cfg = {codec, width: w, height: h, bitrate, framerate: fps};
    try { const r = await VideoEncoder.isConfigSupported(cfg); if (r.supported) return [cfg, mux]; } catch (e) {}
  }
  return [null, null];
}
// the sound of [t0, t1) of the video as planar float PCM, and an encoder config for it
async function prepareSound(t0, t1) {
  const buf = await (await fetch(v.currentSrc)).arrayBuffer();
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  let ab;
  try { ab = await ac.decodeAudioData(buf); } catch (e) { return null; }  // no audio track
  finally { ac.close(); }
  if (!ab || !ab.length) return null;
  const rate = ab.sampleRate, ch = Math.min(2, ab.numberOfChannels);
  const a = Math.max(0, Math.round(t0 * rate)), b = Math.min(ab.length, Math.round(t1 * rate));
  if (b <= a) return null;
  const planes = []; for (let c2 = 0; c2 < ch; c2++) planes.push(ab.getChannelData(c2).slice(a, b));
  for (const [codec, mux] of [["mp4a.40.2", "aac"], ["opus", "opus"]]) {
    const r2 = codec === "opus" ? 48000 : rate;
    const cfg = {codec, sampleRate: r2, numberOfChannels: ch, bitrate: 160000};
    try { if ((await AudioEncoder.isConfigSupported(cfg)).supported) {
      if (r2 !== rate) return {cfg, mux, rate: r2, ch, planes: planes.map(p => resample(p, rate, r2))};
      return {cfg, mux, rate, ch, planes}; } } catch (e) {}
  }
  throw new Error("no sound encoder in this browser");
}
function resample(x, from, to) {  // linear: good enough for a voice / a gym's sound
  const n = Math.round(x.length * to / from), out = new Float32Array(n);
  for (let i = 0; i < n; i++) { const p = i * from / to, k = Math.floor(p), f = p - k; out[i] = (x[k] || 0) * (1 - f) + (x[k + 1] || 0) * f; }
  return out;
}
$('xgo').onclick = async () => {
  if (exporting || !window.VideoEncoder) return;
  const fps = S.fps, i0 = Math.max(0, Math.floor(+$('xa').value * fps)), i1 = Math.min(N - 1, Math.ceil(+$('xb').value * fps) - 1);
  if (!(i1 > i0)) { $('xerr').textContent = "The end must be after the start."; return; }
  const sz = +$('xsize').value, sc = sz === 1 ? 1 : Math.min(1, sz / Math.min(W, H));
  const ow = Math.round(W * sc / 2) * 2, oh = Math.round(H * sc / 2) * 2;
  const withTL = $('xtl').checked && TL.rows.length > 0, k = ow / 1000;
  const bandH = withTL ? Math.round(tlH * k / 2) * 2 : 0, total = oh + bandH;
  const outFps = fps * +$('xspeed').value;  // slow motion: same frames, played slower
  const cv = document.createElement('canvas'); cv.width = ow; cv.height = total; const g = cv.getContext('2d');
  const [cfg, mux] = await pickCodec(ow, total, outFps, Math.round(Math.min(40e6, 10e6 * ow * total / (1080 * 1920))));
  if (!cfg) { $('xerr').textContent = "No video encoder for this size in this browser; try a smaller size, or the terminal command."; return; }
  // the clip's sound for [t0, t1): decoded and encoded in the browser (AAC, else Opus),
  // before the video, so a problem with the sound only drops the sound
  let snd = null, sndNote = "", aChunks = [];
  if ($('xsnd').checked && +$('xspeed').value === 1 && window.AudioEncoder) {
    $('xprog').style.display = ""; $('xmsg').textContent = "reading the sound…";
    try {
      snd = await prepareSound(i0 / fps, (i1 + 1) / fps);
      if (!snd) sndNote = " (the clip has no sound)";
      else {
        let aFail = null;
        const aenc = new AudioEncoder({output: (chunk, meta) => aChunks.push([chunk, meta]), error: e => aFail = e});
        aenc.configure(snd.cfg);
        const step = 1024, n = snd.planes[0].length;
        for (let o = 0; o < n; o += step) {
          const len = Math.min(step, n - o), data = new Float32Array(len * snd.ch);
          for (let c2 = 0; c2 < snd.ch; c2++) data.set(snd.planes[c2].subarray(o, o + len), c2 * len);
          const ad = new AudioData({format: "f32-planar", sampleRate: snd.rate, numberOfFrames: len, numberOfChannels: snd.ch,
            timestamp: Math.round(o * 1e6 / snd.rate), data});
          aenc.encode(ad); ad.close();
        }
        await aenc.flush(); aenc.close();
        if (aFail) throw aFail;
      }
    } catch (e) { snd = null; aChunks = []; sndNote = " (without sound: " + (e && e.message ? e.message : e) + ")"; }
  } else if ($('xsnd').checked && +$('xspeed').value !== 1) sndNote = " (no sound in slow motion)";
  else if ($('xsnd').checked && !window.AudioEncoder) sndNote = " (this browser can't encode sound)";
  const target = new Mp4Muxer.ArrayBufferTarget();
  const muxer = new Mp4Muxer.Muxer({target, video: {codec: mux, width: ow, height: total},
    ...(snd ? {audio: {codec: snd.mux, sampleRate: snd.rate, numberOfChannels: snd.ch}} : {}), fastStart: "in-memory"});
  for (const [chunk, meta] of aChunks) muxer.addAudioChunk(chunk, meta);
  let failed = null;
  const enc = new VideoEncoder({output: (chunk, meta) => muxer.addVideoChunk(chunk, meta), error: e => failed = e});
  enc.configure(cfg);
  exporting = true; cancelled = false; $('xgo').disabled = true; $('xcancel').textContent = "Cancel";
  $('xprog').style.display = ""; $('xerr').textContent = "";
  const keep = v.currentTime, wasMuted = v.muted; v.pause();
  try {
    for (let i = i0; i <= i1; i++) {
      if (cancelled || failed) break;
      await seekTo((i + 0.5) / fps);
      if (Math.abs(v.currentTime - (i + 0.5) / fps) > 1 / fps) throw new Error("the video can't be seeked here (served without byte ranges?) — use climbvision view outputs/" + S.name + " --open, or pick viewer.mp4");
      g.fillStyle = "#121212"; g.fillRect(0, 0, ow, total);
      g.drawImage(v, 0, 0, ow, oh);
      g.save(); g.scale(ow / W, oh / H); paintOverlays(g, i); g.restore();
      if (withTL) { g.save(); g.translate(0, oh); g.scale(k, k); g.fillStyle = "#1b1d21"; g.fillRect(0, 0, 1000, tlH); paintTimeline(g, 1000, S.t[i]); g.restore(); }
      const frame = new VideoFrame(cv, {timestamp: Math.round((i - i0) * 1e6 / outFps), duration: Math.round(1e6 / outFps)});
      enc.encode(frame, {keyFrame: (i - i0) % Math.max(1, Math.round(outFps * 2)) === 0}); frame.close();
      while (enc.encodeQueueSize > 6) await new Promise(r => setTimeout(r, 5));
      $('xbar').value = (i - i0 + 1) / (i1 - i0 + 1); $('xmsg').textContent = `frame ${i - i0 + 1} of ${i1 - i0 + 1}`;
    }
    if (failed) throw failed;
    if (!cancelled) {
      await enc.flush(); muxer.finalize();
      const blob = new Blob([target.buffer], {type: "video/mp4"});
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
      a.download = `${S.name}_${lastPreset() || "custom"}${+$('xspeed').value < 1 ? "_slow" : ""}.mp4`; document.body.appendChild(a); a.click(); a.remove();
      $('xmsg').textContent = `Saved ${a.download} (${(blob.size / 1e6).toFixed(1)} MB) to your downloads${sndNote}.`;
    } else $('xmsg').textContent = "Cancelled.";
  } catch (e) {
    if (e && e.name === "SecurityError") {
      $('xerr').innerHTML = "The browser blocks reading the video of a page opened from disk. Pick <b>viewer.mp4</b> (it's next to this page) once, then Export again — or run <i>climbvision view outputs/" + S.name + " --open</i>, which serves the page so this isn't needed.";
      $('xpick').style.display = "";
    } else $('xerr').textContent = "Export failed: " + (e && e.message ? e.message : e);
  } finally {
    try { if (enc.state !== "closed") enc.close(); } catch (e) {}
    exporting = false; $('xgo').disabled = false; $('xcancel').textContent = "Close"; v.muted = wasMuted;
    await seekTo(keep); mediaTime = v.currentTime; draw();
  }
};
</script></body></html>
"""
