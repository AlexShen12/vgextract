"""Pinned asset conversion and CPU renders. Run Blender script with --blender-worker."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def camera_elevations(policy):
    elevations = policy.get("elevations", [45, 60])
    if (len(elevations) != 2 or len(set(elevations)) != 2
            or any(type(value) not in (int, float) or not 15 <= value <= 85 for value in elevations)):
        raise ValueError("Render policy needs two distinct camera elevations between 15 and 85 degrees")
    if elevations != [45, 60] and not policy.get("development_evidence"):
        raise ValueError("Calibrated elevations need development evidence")
    return elevations


def render_mask(image):
    if "A" not in image.getbands():
        raise ValueError("Rendered reference needs a foreground alpha channel")
    mask = image.getchannel("A").point(lambda value: 255 if value > 127 else 0)
    box = mask.getbbox()
    if box is None:
        raise ValueError("Blank rendered skin")
    if box[0] == 0 or box[1] == 0 or box[2] == image.width or box[3] == image.height:
        raise ValueError("Rendered skin clips the camera bounds")
    return mask


def render_assets(manifest_path, converter, blender, output):
    from .core import atomic_json, digest, read_json
    from .media import file_hash
    manifest = read_json(Path(manifest_path)); output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    render_policy = manifest.get("render_policy", {})
    elevations = camera_elevations(render_policy)
    report_path = output / "render-report.json"
    if report_path.exists() and read_json(report_path)["manifest_hash"] != digest(manifest):
        raise ValueError("Render inputs changed. Use a new output directory.")
    lock_path=output/"tool-lock.json"
    tools = read_json(lock_path) if lock_path.exists() else manifest.get("tools")
    if tools is None:
        tools={name:{"path":str(Path(path).resolve()),"sha256":file_hash(Path(path))}
               for name,path in (("lol2gltf",converter),("blender",blender))}
        atomic_json(lock_path,tools)
    for name, path in (("lol2gltf", converter), ("blender", blender)):
        if file_hash(Path(path)) != tools[name]["sha256"]:
            raise ValueError("Renderer executable differs from its frozen hash")
    results = []
    for skin in manifest["skins"]:
        target = output/(skin["champion"]+"-"+str(skin["skin_id"]))
        target.mkdir(parents=True, exist_ok=True)
        try:
            files = {entry["role"]: Path(entry["path"]).resolve() for entry in skin["files"] if entry["role"] in {"mesh", "skeleton"}}
            for entry in skin["files"]:
                if file_hash(Path(entry["path"])) != entry["sha256"]:
                    raise ValueError("Changed model/texture/animation asset")
            animations = [entry for entry in skin["files"] if entry["role"] == "animation"]
            animation_dir = target/"animations"; animation_dir.mkdir(exist_ok=True)
            from shutil import copy2
            for entry in animations:
                copy2(entry["path"], animation_dir/Path(entry["path"]).name)
            textures = [entry for entry in skin["files"] if entry["role"] == "texture"]
            command = [str(converter), "skn2gltf", "-m", str(files["mesh"]), "-s", str(files["skeleton"]),
                       "-g", str(target/"model.glb"), "-a", str(animation_dir)]
            subprocess.run(command, check=True, timeout=300, capture_output=True)
            if textures:
                if any(e["material"]=="__auto_single__" for e in textures):
                    import struct
                    raw=(target/"model.glb").read_bytes();length,kind=struct.unpack_from("<II",raw,12)
                    if kind!=0x4e4f534a:raise ValueError("Converted GLB has no JSON chunk")
                    gltf=json.loads(raw[20:20+length]);materials=gltf.get("materials",[])
                    if len(textures)!=1 or len(materials)!=1 or not materials[0].get("name"):
                        raise ValueError("Multiple skin materials require explicit verified texture bindings")
                    textures=[{**textures[0],"material":materials[0]["name"]}]
                textured=command+["--materials",*[e["material"] for e in textures],"--textures",*[e["path"] for e in textures]]
                subprocess.run(textured,check=True,timeout=300,capture_output=True)
            subprocess.run([str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
                            "--python", str(Path(__file__).resolve()), "--", "--blender-worker",
                            "--input", str(target/"model.glb"), "--output", str(target), "--elevations", *map(str, elevations)],
                           check=True, timeout=7200, capture_output=True)
            rendered = json.loads((target/"renders.json").read_text())
            for entry in rendered:
                from PIL import Image
                with Image.open(target/entry["path"]) as image:
                    mask = render_mask(image)
                    mask_path=target/(Path(entry["path"]).stem+"-mask.png");mask.save(mask_path)
                entry.update(champion=skin["champion"], skin_id=str(skin["skin_id"]), reference_kind="rendered",
                             verified=False, reviewer=None, sha256=file_hash(target/entry["path"]),
                             mask_path=str(mask_path),mask_sha256=file_hash(mask_path),
                             model_sha256=file_hash(target/"model.glb"), source_assets=skin["files"])
                entry["path"] = str(target/entry["path"])
                entry["id"] = digest(entry)[:20]
            available_poses = {entry["pose_kind"] for entry in rendered}
            results.append({"champion": skin["champion"], "skin_id": skin["skin_id"], "status": "needs_visual_audit",
                            "available_poses": sorted(available_poses),
                            "missing_poses": sorted({"idle", "movement", "attack", "cast"} - available_poses),
                            "renders": rendered})
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            results.append({"champion": skin["champion"], "skin_id": skin["skin_id"], "status": "unsupported", "error": str(exc)})
        atomic_json(report_path, {"tools": tools, "manifest_hash": digest(manifest), "render_policy": render_policy, "skins": results})
    return results


def audit(report_path,review_path,output):
    from .core import atomic_json,read_json
    from .media import file_hash
    report=read_json(Path(report_path));reviews=read_json(Path(review_path));gallery=[]
    for review in reviews:
        if review.get("approved") is not True or not review.get("reviewer") or review.get("reviewer_kind") not in {"human","model"}:
            raise ValueError("Explicit source render audit required")
        skin=next(s for s in report["skins"] if s["champion"]==review["champion"] and str(s["skin_id"])==str(review["skin_id"]))
        if skin["status"]!="needs_visual_audit":raise ValueError("Unsupported skin cannot enter gallery")
        for entry in skin["renders"]:
            if file_hash(Path(entry["path"]))!=entry["sha256"] or file_hash(Path(entry["mask_path"]))!=entry["mask_sha256"]:
                raise ValueError("Rendered skin changed after audit")
            gallery.append({**entry,"verified":review["reviewer_kind"]=="human","source_reviewed":True,"identity_approved":True,
                            "reviewer":review["reviewer"],"reviewer_kind":review["reviewer_kind"],
                            "identity_evidence":{"render_audit":review,"tools":report["tools"]}})
    atomic_json(Path(output),gallery);return gallery


def blender_worker(source, output, elevations=(45, 60)):
    import bpy
    from mathutils import Vector
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(source))
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"; scene.cycles.device = "CPU"; scene.cycles.samples = 8
    scene.render.resolution_x = scene.render.resolution_y = 128; scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"; scene.render.image_settings.color_mode = "RGBA"
    scene.world.color = (.8,.8,.8)
    meshes = [o for o in scene.objects if o.type == "MESH"]
    if not meshes:
        raise ValueError("No renderable skin mesh")
    corners = [obj.matrix_world @ Vector(v) for obj in meshes for v in obj.bound_box]
    center = sum(corners, Vector())/len(corners)
    extent = max((v-center).length for v in corners)
    bpy.ops.object.light_add(type="AREA", location=center+Vector((0,0,extent*4)))
    bpy.context.object.data.energy = 400; bpy.context.object.data.shape = "DISK"; bpy.context.object.data.size = extent*4
    bpy.ops.object.camera_add(); camera = bpy.context.object; camera.data.type = "ORTHO"; camera.data.ortho_scale = extent*2.6
    scene.camera = camera
    armatures = [o for o in scene.objects if o.type == "ARMATURE"]
    actions = list(bpy.data.actions)
    selected = []
    for kind, terms in (("idle", ("idle",)), ("movement", ("run", "walk")), ("attack", ("attack",)), ("cast", ("spell", "cast"))):
        matches = [a for a in actions if any(term in a.name.casefold() for term in terms)]
        selected.extend((kind,a) for a in matches)
    if not selected:
        raise ValueError("No supported animated poses; static bind pose is not an animation gallery")
    records = []
    for action_index, (kind, action) in enumerate(selected):
        for armature in armatures:
            armature.animation_data_create(); armature.animation_data.action = action
        lower, upper = action.frame_range
        for phase in range(4):
            scene.frame_set(round(lower+(upper-lower)*phase/4))
            depsgraph = bpy.context.evaluated_depsgraph_get()
            evaluated = [obj.evaluated_get(depsgraph) for obj in meshes]
            corners = [obj.matrix_world @ Vector(corner) for obj in evaluated for corner in obj.bound_box]
            center = sum(corners, Vector()) / len(corners)
            extent = max((corner - center).length for corner in corners)
            camera.data.ortho_scale = extent * 2.6
            for elevation in elevations:
                for heading in range(8):
                    theta = heading*math.pi/4; phi=math.radians(elevation)
                    camera.location=center+Vector((math.cos(theta)*math.cos(phi), math.sin(theta)*math.cos(phi), math.sin(phi)))*extent*5
                    camera.rotation_euler=(center-camera.location).to_track_quat("-Z", "Y").to_euler()
                    name=f"{kind}-{action_index}-{phase}-{elevation}-{heading}.png"
                    scene.render.filepath=str(output/name); bpy.ops.render.render(write_still=True)
                    records.append({"path":name, "animation":action.name, "pose_kind":kind, "phase":phase/4,
                                    "elevation":elevation, "heading":heading*45})
    (output/"renders.json").write_text(json.dumps(records, indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blender-worker", action="store_true"); p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--converter", type=Path); p.add_argument("--blender", type=Path)
    p.add_argument("--audit",type=Path,help="Explicit per-skin visual review JSON")
    p.add_argument("--elevations", nargs=2, type=float, default=(45, 60))
    args=p.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)
    if args.blender_worker: blender_worker(args.input,args.output,args.elevations)
    elif args.audit:audit(args.input,args.audit,args.output)
    else: render_assets(args.input,args.converter,args.blender,args.output)


if __name__ == "__main__": main()
