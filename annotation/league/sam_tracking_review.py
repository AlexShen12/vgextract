"""Source-synchronized SAM pilot playback with exact native-frame mask overlays."""
from pathlib import Path

from .core import digest, read_json


PAGE = """<!doctype html><meta charset="utf-8"><title>SAM tracking pilot</title>
<style>body{background:#151b26;color:#eef;font:16px system-ui;margin:24px;max-width:1200px}
button,select{font:inherit;margin:8px;padding:6px;background:#293448;color:white;border:1px solid #789}
#screen{position:relative;width:100%;aspect-ratio:16/9}video,canvas{position:absolute;width:100%;height:100%}
canvas{pointer-events:none}pre{white-space:pre-wrap}a{color:#9cf}</style>
<h1>Selected-player SAM tracking</h1><p>Experimental development evidence. Champion naming comes from existing HUD portrait evidence.</p>
<select id="clip" aria-label="Clip"></select><select id="arm" aria-label="Seed method"><option value="verified">Verified source seed</option><option value="automatic">Automatic body seed</option></select>
<button id="play">Play clip</button><button id="refresh">Refresh results</button><p id="state"></p>
<div id="screen"><video id="video" controls src="/video"></video><canvas id="mask" width="1920" height="1080"></canvas></div>
<p id="cue"></p><details><summary>Run counts and resources</summary><pre id="summary"></pre></details>
<script>
const $=id=>document.getElementById(id),video=$('video'),canvas=$('mask'),ctx=canvas.getContext('2d');
let data,clip,arm,maskURL='',generation=0;const pictures=new Map();
async function load(){data=await (await fetch('/data')).json();const previous=$('clip').value;$('clip').replaceChildren();
for(const c of data.clips){const o=new Option(`${c.id} · ${c.start.toFixed(2)}–${c.end.toFixed(2)}s`,c.id);$('clip').add(o)}
if(previous)$('clip').value=previous;$('summary').textContent=JSON.stringify(data.summary,null,2);select()}
function select(){generation++;clip=data.clips.find(c=>c.id===$('clip').value);arm=clip.arms[$('arm').value];maskURL='';ctx.clearRect(0,0,1920,1080);
$('state').textContent=`${arm.run_status} · ${arm.seed?.champion||'the selected champion'} · ${arm.status}`;
video.currentTime=clip.start;$('cue').textContent=arm.seed?`Seed at ${arm.seed.time.toFixed(3)}s · ${arm.seed.source}`:'No supported seed';}
async function draw(){if(!clip||!arm)return;const f=arm.masks.reduce((best,f)=>Math.abs(f.time-video.currentTime)<Math.abs((best?.time??1e9)-video.currentTime)?f:best,null);
const url=f&&Math.abs(f.time-video.currentTime)<.04?f.url:'';if(url===maskURL)return;maskURL=url;const revision=generation;ctx.clearRect(0,0,1920,1080);if(!url)return;
let image=pictures.get(url);if(!image){image=new Image();image.src=url;await image.decode();pictures.set(url,image)}
if(revision!==generation||url!==maskURL)return;ctx.drawImage(image,0,0);const pixels=ctx.getImageData(0,0,1920,1080);
for(let i=0;i<pixels.data.length;i+=4){const visible=pixels.data[i]>127;pixels.data[i]=40;pixels.data[i+1]=255;pixels.data[i+2]=130;pixels.data[i+3]=visible?110:0}ctx.putImageData(pixels,0,0);}
video.addEventListener('timeupdate',()=>{if(clip&&video.currentTime>=clip.end){video.pause();video.currentTime=clip.end-.001}draw()});
function tick(){draw();requestAnimationFrame(tick)}tick();
$('clip').onchange=select;$('arm').onchange=select;$('refresh').onclick=load;$('play').onclick=async()=>{video.currentTime=clip.start;await video.play()};load().catch(e=>$('state').textContent=e.message);
</script>"""


def create_app(prepared_path):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    prepared_path = Path(prepared_path)
    app = FastAPI()
    assets = {}

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE

    @app.get("/video")
    def video():
        return FileResponse(read_json(prepared_path)["source"]["path"], media_type="video/mp4")

    @app.get("/asset/{key}")
    def asset(key: str):
        if key not in assets:
            raise HTTPException(404)
        return FileResponse(assets[key])

    @app.get("/data")
    def data():
        prepared = read_json(prepared_path)
        root = Path(prepared["root"])
        clips = []
        for clip in prepared["clips"]:
            row = {k: clip[k] for k in ("id", "start", "end")}
            row["arms"] = {}
            for arm, info in clip["arms"].items():
                result = root / arm / (clip["id"] + "-complete.json")
                if not result.exists():
                    result = root / arm / (clip["id"] + "-partial.json")
                frames = read_json(result)["frames"] if result.exists() else []
                masks = []
                for frame in frames:
                    index = frame["frame_index"]
                    if not 0 <= index < len(info.get("frames", [])):
                        continue
                    for mask in frame["masks"]:
                        if mask["object_id"] != 1:
                            continue
                        path = Path(mask["mask_path"])
                        key = digest(str(path))[:20]
                        if path.is_file() and path.resolve().is_relative_to(root.resolve()):
                            assets[key] = path
                            masks.append({"time": info["frames"][index]["time"], "url": "/asset/" + key})
                report = root / arm / "report.json"
                row["arms"][arm] = {"status": info["status"], "seed": info.get("seed"), "masks": masks,
                    "run_status": read_json(report)["status"] if report.exists() else "pending"}
            clips.append(row)
        summary = root / "summary.json"
        return {"clips": clips, "summary": read_json(summary) if summary.exists() else {"status": "pending"}}

    return app


def serve(prepared_path, port):
    import uvicorn
    uvicorn.run(create_app(prepared_path), host="127.0.0.1", port=port)
    return 0
