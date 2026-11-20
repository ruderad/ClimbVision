"""The marking page for `climbvision wall mark`: pair landmarks of a wall scan with the
same spots on a clip's wall picture, for `climbvision wall fit`.

Left: the scan's front view with numbered landmarks (holds and volumes). Right: the clip's
`wall.jpg`. Click a number on the left, then the same spot on the right; repeat for 5-8
landmarks spread over the wall (high, low, left, right). "Download" saves
`wall_marks.csv` (landmark, x, y in wall.jpg pixels).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2

from climbvision.engine.wallmodel import WallModel, front_view


def _b64(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def write_mark_page(model: WallModel, result_dir: str | Path) -> Path:
    result_dir = Path(result_dir)
    wall = cv2.imread(str(result_dir / "wall.jpg"))
    if wall is None:
        raise FileNotFoundError(f"{result_dir / 'wall.jpg'} (run the camera layer first)")
    front, info = front_view(model.P, model.C, model.landmarks)
    lms = [{"id": d["id"], "u": (d["xyz"][0] - info["x0"]) * info["res"],
            "v": (info["z1"] - d["xyz"][2]) * info["res"],
            "h": round(d["xyz"][2], 2)} for d in model.landmarks]
    data = {"wall": model.name, "clip": result_dir.name, "landmarks": lms,
            "front": _b64(front), "photo": _b64(wall)}
    html = PAGE.replace("__DATA__", json.dumps(data))
    out = result_dir / "wall_marks.html"
    out.write_text(html, encoding="utf-8")
    return out


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mark the wall — ClimbVision</title>
<style>
:root{--bg:#0B1C2C;--fg:#F4F8FA;--muted:#8EC3DA;--accent:#F0A58A}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{padding:10px 16px;border-bottom:1px solid #2E4A62}
h1{font-size:16px;margin:0 0 4px} p{margin:2px 0;color:var(--muted)}
main{display:flex;gap:12px;padding:12px;align-items:flex-start;flex-wrap:wrap}
.pane{position:relative;flex:1 1 420px;min-width:300px} .pane:first-child{flex:1.7 1 560px}
.pane h2{font-size:13px;margin:0 0 6px;color:var(--muted)}
canvas{width:100%;height:auto;display:block;border-radius:6px;cursor:crosshair;background:#000}
aside{flex:0 0 240px} table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{padding:3px 6px;border-bottom:1px solid #2E4A62;text-align:left}
button{background:#2E4A62;color:var(--fg);border:0;border-radius:6px;padding:7px 12px;margin:4px 4px 0 0;cursor:pointer}
button.primary{background:var(--accent);color:var(--bg);font-weight:600}
#state{color:var(--accent);margin-top:6px;min-height:1.4em}
</style></head><body>
<header><h1>Mark the wall: <span id="t"></span></h1>
<p>1. Click a numbered landmark on the scan (left). 2. Click the same spot on the video's wall picture (right). Do 5–8, spread out: high, low, left, right.</p>
<p>Then <b>Download</b> and run <code>climbvision wall fit <span id="w"></span> outputs/<span id="c"></span> --marks wall_marks.csv</code> (or put the file in the clip folder).</p></header>
<main>
<div class="pane"><h2>Wall scan (front view)</h2><canvas id="scan"></canvas></div>
<div class="pane"><h2>Video wall picture (wall.jpg)</h2><canvas id="photo"></canvas></div>
<aside><table id="tbl"><thead><tr><th>#</th><th>height</th><th>x</th><th>y</th><th></th></tr></thead><tbody></tbody></table>
<div id="state"></div>
<button id="undo">Undo</button><button id="clear">Clear</button><button class="primary" id="dl">Download</button></aside>
</main>
<script>
const D = __DATA__;
document.getElementById('t').textContent = D.wall + ' ↔ ' + D.clip;
document.getElementById('w').textContent = D.wall; document.getElementById('c').textContent = D.clip;
const pairs = []; let sel = null;
function load(src){ return new Promise(r=>{const i=new Image(); i.onload=()=>r(i); i.src=src;}); }
function pos(c, e){ const r=c.getBoundingClientRect(); return [(e.clientX-r.left)*c.width/r.width, (e.clientY-r.top)*c.height/r.height]; }
Promise.all([load(D.front), load(D.photo)]).then(([fi, pi])=>{
  const sc=document.getElementById('scan'), ph=document.getElementById('photo');
  sc.width=fi.width; sc.height=fi.height; ph.width=pi.width; ph.height=pi.height;
  const s=sc.getContext('2d'), p=ph.getContext('2d');
  function draw(){
    s.drawImage(fi,0,0);
    for(const l of D.landmarks){ const done=pairs.find(q=>q.id===l.id);
      s.beginPath(); s.arc(l.u,l.v,13,0,7); s.lineWidth=3;
      s.strokeStyle = l.id===sel ? '#F0A58A' : done ? '#7CFC00' : 'rgba(0,0,0,0)'; s.stroke(); }
    p.drawImage(pi,0,0);
    for(const q of pairs){ p.beginPath(); p.arc(q.x,q.y,7,0,7); p.lineWidth=3; p.strokeStyle='#7CFC00'; p.stroke();
      p.font='bold 16px system-ui'; p.fillStyle='#000'; p.fillText(q.id,q.x+9,q.y+5); p.fillStyle='#7CFC00'; p.fillText(q.id,q.x+8,q.y+4); }
    const tb=document.querySelector('#tbl tbody'); tb.innerHTML='';
    for(const q of pairs){ const l=D.landmarks.find(l=>l.id===q.id);
      tb.insertAdjacentHTML('beforeend', `<tr><td>${q.id}</td><td>${l.h} m</td><td>${q.x.toFixed(0)}</td><td>${q.y.toFixed(0)}</td><td><a href="#" data-id="${q.id}">✕</a></td></tr>`); }
    document.getElementById('state').textContent = sel ? `Now click landmark ${sel} on the wall picture` :
      (pairs.length < 5 ? `${pairs.length} marked — at least 5 needed` : `${pairs.length} marked — ready`);
  }
  sc.onclick = e=>{ const [x,y]=pos(sc,e); let best=null, bd=1e9;
    for(const l of D.landmarks){ const d=Math.hypot(l.u-x,l.v-y); if(d<bd){bd=d;best=l;} }
    if(best && bd<40){ sel=best.id; draw(); } };
  ph.onclick = e=>{ if(!sel) return; const [x,y]=pos(ph,e);
    const i=pairs.findIndex(q=>q.id===sel); const q={id:sel,x,y}; if(i>=0) pairs[i]=q; else pairs.push(q); sel=null; draw(); };
  document.getElementById('tbl').onclick = e=>{ const id=+e.target.dataset.id; if(id){ e.preventDefault(); const i=pairs.findIndex(q=>q.id===id); if(i>=0) pairs.splice(i,1); draw(); } };
  document.getElementById('undo').onclick = ()=>{ pairs.pop(); draw(); };
  document.getElementById('clear').onclick = ()=>{ pairs.length=0; sel=null; draw(); };
  document.getElementById('dl').onclick = ()=>{
    const csv = 'landmark,x,y\n' + pairs.map(q=>`${q.id},${q.x.toFixed(1)},${q.y.toFixed(1)}`).join('\n') + '\n';
    const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'})); a.download='wall_marks.csv'; a.click(); };
  draw();
});
</script></body></html>
"""
