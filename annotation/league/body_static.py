"""Static CommunityDragon OBJ candidates, explicitly separate from animated galleries."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from urllib.request import Request,urlopen


def prepare(cache,requests,dd="16.22.1",cd="16.22"):
    from .core import atomic_json
    from .knowledge import JsonCache,DD,CD
    from .reference_bundle import versions
    versions(dd,cd);cache=Path(cache);metadata=JsonCache(cache/"raw")
    roster,_=metadata.get(f"{DD}/cdn/{dd}/data/en_US/champion.json");records=[];failures=[]
    for request in requests:
        name=request["champion"];number=int(request.get("skin_number",0))
        try:
            skin,source=metadata.get(f"{CD}/{cd}/game/data/characters/{name.lower()}/skins/skin{number}.bin.json")
            skin_id=int(roster["data"][name]["key"])*1000+number
            record=next(v for v in skin.values() if isinstance(v,dict) and v.get("championSkinId")==skin_id)
            props=record["skinMeshProperties"];folder=str(Path(props["simpleSkin"]).with_suffix("")).lower()
            if not folder.startswith("assets/") or ".." in folder:raise ValueError("Invalid exported model path")
            listing,listing_source=metadata.get(f"{CD}/json/{cd}/game/{folder}/")
            names=[e["name"] for e in listing if e.get("type")=="file" and e["name"].lower().endswith(".obj")]
            if len(names)!=1 or Path(names[0]).name!=names[0]:raise ValueError("Static export requires one unambiguous OBJ")
            texture=props["texture"].lower().rsplit(".",1)[0]+".png"
            urls={"mesh":f"{CD}/{cd}/game/{folder}/{names[0]}","texture":f"{CD}/{cd}/game/{texture}"}
            files={}
            for role,url in urls.items():
                path=cache/"files"/(hashlib.sha256(url.encode()).hexdigest()+(".obj" if role=="mesh" else ".png"))
                receipt=path.with_suffix(path.suffix+".json")
                if not receipt.exists():
                    with urlopen(Request(url,headers={"User-Agent":"gameplay-reference-resolver/0.1"}),timeout=40) as response:
                        if response.url!=url:raise ValueError("Static export redirected")
                        raw=response.read(64*1024*1024+1)
                    if len(raw)>64*1024*1024:raise ValueError("Static export too large")
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
                    atomic_json(receipt,{"url":url,"sha256":hashlib.sha256(raw).hexdigest()})
                saved=json.loads(receipt.read_text())
                if hashlib.sha256(path.read_bytes()).hexdigest()!=saved["sha256"]:raise ValueError("Static asset hash changed")
                files[role]={**saved,"path":str(path.resolve())}
            records.append({"champion":name,"skin_id":str(skin_id),"skin_name":record["championSkinName"],"files":files,
                "sources":[source,listing_source],"pose":"static_export","review_status":"unreviewed_candidate","supports_animation":False})
        except (OSError,ValueError,RuntimeError,KeyError,StopIteration) as exc:failures.append({**request,"error":str(exc)})
    result={"reference_version":dd,"communitydragon_version":cd,"models":records,"failures":failures,
            "status":"complete" if not failures else "partial"}
    atomic_json(cache/"manifest.json",result);return result


def worker(manifest,output):
    import bpy
    from mathutils import Vector
    metadata=json.loads(Path(manifest).read_text());models=metadata["models"];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[]
    for model in models:
        for record in model["files"].values():
            if hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()!=record["sha256"]:raise ValueError("Model input changed")
        bpy.ops.object.select_all(action="SELECT");bpy.ops.object.delete(use_global=False)
        bpy.ops.wm.obj_import(filepath=model["files"]["mesh"]["path"])
        objects=[o for o in bpy.context.scene.objects if o.type=="MESH"]
        if not objects:raise ValueError("No static mesh imported")
        material=bpy.data.materials.new("diffuse_export");material.use_nodes=True;nodes=material.node_tree.nodes;nodes.clear()
        texture=nodes.new("ShaderNodeTexImage");texture.image=bpy.data.images.load(model["files"]["texture"]["path"])
        emission=nodes.new("ShaderNodeEmission");transparent=nodes.new("ShaderNodeBsdfTransparent");mix=nodes.new("ShaderNodeMixShader");surface=nodes.new("ShaderNodeOutputMaterial")
        links=material.node_tree.links;links.new(texture.outputs["Color"],emission.inputs["Color"])
        links.new(texture.outputs["Alpha"],mix.inputs[0]);links.new(transparent.outputs[0],mix.inputs[1]);links.new(emission.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],surface.inputs["Surface"])
        for obj in objects:obj.data.materials.clear();obj.data.materials.append(material)
        corners=[obj.matrix_world@Vector(v) for obj in objects for v in obj.bound_box];center=sum(corners,Vector())/len(corners);extent=max((v-center).length for v in corners)
        scene=bpy.context.scene;scene.render.engine="CYCLES";scene.cycles.device="CPU";scene.cycles.samples=4
        scene.render.resolution_x=scene.render.resolution_y=128;scene.render.resolution_percentage=100;scene.render.film_transparent=True
        scene.render.image_settings.file_format="PNG";scene.render.image_settings.color_mode="RGBA"
        bpy.ops.object.camera_add();camera=bpy.context.object;camera.data.type="ORTHO";camera.data.ortho_scale=extent*2.3;scene.camera=camera
        for elevation in (45,60):
            for heading in range(8):
                theta=heading*math.pi/4;phi=math.radians(elevation)
                camera.location=center+Vector((math.cos(theta)*math.cos(phi),math.sin(theta)*math.cos(phi),math.sin(phi)))*extent*5
                camera.rotation_euler=(center-camera.location).to_track_quat("-Z","Y").to_euler()
                name=f"{model['champion']}-{model['skin_id']}-{elevation}-{heading}.png";path=output/name
                scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
                # Blender's bundled Python can derive masks without external packages.
                rendered=bpy.data.images.load(str(path),check_existing=False)
                rgba=list(rendered.pixels[:]);alpha=rgba[3::4]
                if not any(a>.5 for a in alpha):raise ValueError("Blank static render")
                mask_image=bpy.data.images.new(name+"-mask",128,128,alpha=True)
                mask_image.pixels=[component for a in alpha for component in (a,a,a,1)]
                mask_path=output/(path.stem+"-mask.png");mask_image.filepath_raw=str(mask_path);mask_image.file_format="PNG";mask_image.save()
                records.append({"champion":model["champion"],"skin_id":model["skin_id"],"image":str(path),"mask":str(mask_path),
                    "image_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"mask_sha256":hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                    "heading":heading*45,"elevation":elevation,"pose":"static_export","review_status":"unreviewed_candidate"})
                (output/"report.json").write_text(json.dumps({"renders":records,"status":"running","quality_gate_passed":False}))
    (output/"report.json").write_text(json.dumps({"renders":records,"status":"completed_static_candidates","quality_gate_passed":False,
        "supports_animation":False,"unsupported_candidates":metadata["failures"],"model_count":len(models)},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("prepare","worker"));p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)
    if args.command=="prepare":print(json.dumps(prepare(args.output,json.loads(args.input.read_text())),indent=2))
    else:worker(args.input,args.output)


if __name__=="__main__":main()
