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
                 scene: Scene | None = None, video: bool = True) -> Path:
    """viewer.html (+ viewer.mp4 unless `video=False` and it exists) in result_dir."""
    from climbvision.engine.video import VideoReader, VideoWriter, warn_if_changed

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
            with VideoWriter(mp4, scene.fps, size, crf=18) as w:
                if not w.browser_ready:
                    print("[climbvision] no ffmpeg found: viewer.mp4 is MPEG-4 and won't play "
                          "in a browser (pip install imageio-ffmpeg)")
                for i, (_, _t, img) in enumerate(reader.frames(step=scene.step)):
                    if i >= len(scene.t):
                        break
                    if resize:
                        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    w.write(img)
    data = scene.to_json().replace("</", "<\\/")
    html = TEMPLATE.replace("__TITLE__", f"{scene.name} — ClimbVision").replace(
        "__DATA__", data)
    out = result_dir / "viewer.html"
    out.write_text(html, encoding="utf-8")
    return out


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
  </div>
  <div id="tlwrap"><canvas id="tl"></canvas><div id="tip"></div></div>
  <div class="help">space play/pause · ← → frame · shift+← → 1 s · [ ] speed · 1–6 presets · click the timeline to jump</div>
</main>
<script type="application/json" id="scene">__DATA__</script>
<script>
const S = JSON.parse(document.getElementById('scene').textContent);
const v = document.getElementById('v'), c = document.getElementById('c'), ctx = c.getContext('2d');
const tl = document.getElementById('tl'), tctx = tl.getContext('2d'), tip = document.getElementById('tip');
const ORDER = ["support","trail","contacts","effort","forces","skeleton","arms","angles","arc","swing","com","speed","velocity","info"];
const LABELS = {skeleton:"Skeleton",arms:"Bent arms",angles:"Joint angles",contacts:"Holds",support:"Base of support",com:"COM (balance)",
  trail:"COM trail",speed:"COM (speed)",forces:"Forces",effort:"Joint effort",velocity:"Velocity",arc:"Dyno arc",swing:"Swing",info:"Info text",timeline:"Timeline"};
const HELP = {skeleton:"bones, spine, joints; grey = hidden or guessed",arms:"holding arms orange when bent",
  contacts:"rings: limbs on holds; dashed: smears",support:"hull of the limbs on holds; plumb line from the COM",
  com:"green inside the base of support, red outside",trail:"COM path, last 2 s",
  forces:"load on each hand / foot, % body weight; arrow = force the hold gives the body",
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
function shape(s, info) {
  const k = s[0];
  ctx.setLineDash([]); ctx.globalAlpha = 1;
  if (k === "l") { const [, x0, y0, x1, y1, col, w, dashed] = s; ctx.strokeStyle = col; ctx.lineWidth = lw(w);
    if (dashed) ctx.setLineDash([6 * unit, 6 * unit]); ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }
  else if (k === "p" || k === "g") { const xy = s[1]; ctx.beginPath(); ctx.moveTo(xy[0], xy[1]);
    for (let j = 2; j < xy.length; j += 2) ctx.lineTo(xy[j], xy[j + 1]);
    if (k === "g") { ctx.closePath(); if (s[4] > 0) { ctx.globalAlpha = s[4]; ctx.fillStyle = s[2]; ctx.fill(); ctx.globalAlpha = 1; } }
    ctx.strokeStyle = s[2]; ctx.lineWidth = lw(s[3]); ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.stroke(); }
  else if (k === "q") { const [, xy, cols, w] = s; ctx.lineWidth = lw(w); ctx.lineCap = "round";
    for (let j = 0; j < cols.length; j++) { ctx.strokeStyle = cols[j]; ctx.beginPath(); ctx.moveTo(xy[2*j], xy[2*j+1]); ctx.lineTo(xy[2*j+2], xy[2*j+3]); ctx.stroke(); } }
  else if (k === "c") { const [, x, y, r, col, w, fill] = s; ctx.beginPath(); ctx.arc(x, y, Math.max(2, r * unit), 0, 2 * Math.PI);
    if (fill) { ctx.globalAlpha = Math.min(1, fill); ctx.fillStyle = col; ctx.fill(); ctx.globalAlpha = 1; }
    if (w) { ctx.strokeStyle = col; ctx.lineWidth = lw(w); ctx.stroke(); } }
  else if (k === "d") { const [, x, y, r, col, w] = s; ctx.strokeStyle = col; ctx.lineWidth = lw(w);
    const rr = Math.max(2, r * unit); ctx.setLineDash([rr * 0.44, rr * 0.35]); ctx.beginPath(); ctx.arc(x, y, rr, 0, 2 * Math.PI); ctx.stroke(); }
  else if (k === "a") { const [, x0, y0, x1, y1, col, w] = s; const L = Math.hypot(x1 - x0, y1 - y0); if (L < 2) return;
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = lw(w); ctx.lineCap = "round";
    const h = Math.min(0.4 * L, 18 * unit), a = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - h * 0.6 * Math.cos(a), y1 - h * 0.6 * Math.sin(a)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - h * Math.cos(a - 0.45), y1 - h * Math.sin(a - 0.45));
    ctx.lineTo(x1 - h * Math.cos(a + 0.45), y1 - h * Math.sin(a + 0.45)); ctx.closePath(); ctx.fill(); }
  else if (k === "t") { const [, x, y, text, col, size] = s; ctx.font = `600 ${Math.round(22 * size * unit)}px system-ui,sans-serif`;
    const m = ctx.measureText(text), hgt = 22 * size * unit, pad = hgt * 0.25;
    ctx.globalAlpha = 0.55; ctx.fillStyle = "#000"; ctx.fillRect(x - pad, y - hgt - pad * 0.2, m.width + 2 * pad, hgt + 1.4 * pad);
    ctx.globalAlpha = 1; ctx.fillStyle = col; ctx.fillText(text, x, y); }
  else if (k === "T" && info) info.push(s);
}
function infoBox(lines) {
  if (!lines.length) return; lines.sort((a, b) => a[1] - b[1]);
  const fs = Math.round(26 * unit), lh = fs * 1.35, x = 15 * unit + 5, y0 = 40 * unit + 10, pad = 8 * unit;
  ctx.font = `600 ${fs}px system-ui,sans-serif`;
  const tw = Math.max(...lines.map(s => ctx.measureText(s[2]).width));
  ctx.globalAlpha = 0.8; ctx.fillStyle = "#000"; ctx.fillRect(x - pad, y0 - fs - pad, tw + 2 * pad, lh * lines.length + pad);
  ctx.globalAlpha = 1; lines.forEach((s, k) => { ctx.fillStyle = s[3]; ctx.fillText(s[2], x, y0 + k * lh); });
}
function draw() {
  const i = frameIndex();
  ctx.clearRect(0, 0, W, H);
  const info = [];
  for (const name of ORDER) if (on.has(name) && S.frames[name]) for (const s of S.frames[name][i]) shape(s, info);
  infoBox(info);
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
function drawTimeline(t) {
  if (!TL.rows.length) return;
  const w = tlW; tctx.clearRect(0, 0, w, tlH);
  const y1 = T0 + rowH * TL.rows.length;
  for (const [a, b, kind] of TL.bands || []) { tctx.fillStyle = kind === "rest" ? "#3c8c3c55" : "#c8323255"; tctx.fillRect(X(a), T0, Math.max(1, X(b) - X(a)), y1 - T0); }
  tctx.font = "12px system-ui,sans-serif";
  TL.rows.forEach((row, r) => { const y = T0 + r * rowH; tctx.fillStyle = "#9a9ca3"; tctx.fillText(row.name, 6, y + 15);
    for (const [a, b, kind] of row.spans) { tctx.globalAlpha = kind === "smear" ? 0.4 : 1;
      tctx.fillStyle = kind === "ground" ? "#555" : row.colour; tctx.fillRect(X(a), y + 4, Math.max(1, X(b) - X(a)), rowH - 8); }
    tctx.globalAlpha = 1; });
  const idx = Object.fromEntries(TL.rows.map((r, k) => [r.limb, k]));
  for (const m of TL.moves || []) { const k = idx[m.limb]; if (k === undefined) continue; const y = T0 + k * rowH + rowH / 2;
    const hot = m.tags.includes("dyno") || m.tags.includes("jump"); tctx.strokeStyle = hot ? "#ff5050" : "#eee"; tctx.lineWidth = 2;
    tctx.beginPath(); tctx.moveTo(X(m.a), y); tctx.lineTo(Math.max(X(m.a) + 1, X(m.b)), y); tctx.stroke();
    tctx.fillStyle = hot ? "#ff5050" : "#ddd"; tctx.fillText(m.n, X(m.b) + 2, y - 3); }
  if (PH) { tctx.fillStyle = "#9a9ca3"; tctx.fillText("phase", 6, y1 + 15);
    for (const [a, b, name] of TL.phases) { tctx.fillStyle = PHC[name] || "#999"; tctx.fillRect(X(a), y1 + 6, Math.max(1, X(b) - X(a)), rowH - 12); } }
  if (TL.load) {  // load on the hands and the feet, % body weight (100 % dotted)
    const ly0 = y1 + PH + 4, lh = LOADH - 8, top = 150, n = S.t.length;
    const Y = v => ly0 + lh - lh * Math.min(v, top) / top;
    tctx.strokeStyle = "#666"; tctx.setLineDash([3, 3]); tctx.beginPath(); tctx.moveTo(left, Y(100)); tctx.lineTo(w - right, Y(100)); tctx.stroke(); tctx.setLineDash([]);
    for (const [key, col] of [["hands", "#b07aff"], ["feet", "#3cd25a"]]) { const arr = TL.load[key]; tctx.strokeStyle = col; tctx.lineWidth = 1.5; tctx.beginPath(); let pen = false;
      for (let i = 0; i < n; i++) { const val = arr[i]; if (val == null) { pen = false; continue; } const x = X(S.t[i]), y = Y(val); pen ? tctx.lineTo(x, y) : tctx.moveTo(x, y); pen = true; } tctx.stroke(); }
    tctx.fillStyle = "#9a9ca3"; tctx.fillText("load", 6, ly0 + 14); tctx.fillStyle = "#b07aff"; tctx.fillText("hands", 6, ly0 + 28); tctx.fillStyle = "#3cd25a"; tctx.fillText("feet", 6, ly0 + 40);
  }
  // seconds
  const y2 = y1 + PH + LOADH;
  tctx.fillStyle = "#777"; const stepS = TL.duration > 60 ? 10 : TL.duration > 20 ? 5 : 1;
  for (let s = 0; s <= TL.duration; s += stepS) { tctx.fillRect(X(s), y2 + 2, 1, 5); tctx.fillText(s + "s", X(s) + 2, y2 + 16); }
  tctx.fillStyle = "#fff"; tctx.fillRect(X(t) - 1, 2, 2, tlH - 4);
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
</script></body></html>
"""
