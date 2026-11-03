"""Image-reference caching with an explicit asset version, independent of match patch."""
from pathlib import Path
import hashlib
from .core import atomic_json,read_json
from .ability_catalog import cache_abilities
from .knowledge import DD
from .visual_assets import download


def cache_references(cache,version='16.13.1',*,offline=False):
    from PIL import Image
    result=cache_abilities(cache,version,offline=offline)
    catalog=read_json(Path(result['catalog']))
    if catalog['status']!='complete_all':raise ValueError('Complete official ability catalog required before identity experiments')
    root=Path(cache)/'qwen35-references'/version;root.mkdir(parents=True,exist_ok=True)
    assets=[]
    for name,c in catalog['champions'].items():
        source=Path(result['catalog']).parent
        for ability in c['abilities']:
            if ability['slot'] not in 'QWER' or len(ability['slot'])!=1:continue
            icon=ability['icon'];raw=(source/icon['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=icon['sha256']:raise ValueError('Ability icon hash mismatch')
            path=root/'images'/(name+'-'+ability['slot']+'.png');path.parent.mkdir(exist_ok=True);path.write_bytes(raw)
            assets.append({'id':ability['id'],'owner':name,'kind':'spell','path':str(path.relative_to(root)),
                           'sha256':icon['sha256'],'url':icon['url']})
        path=root/'images'/(name+'-portrait.png');url=f'{DD}/cdn/{version}/img/champion/{name}.png'
        if not path.exists():
            if offline:raise ValueError('Portrait not cached for offline use: '+name)
            raw=download(url)
            path.parent.mkdir(exist_ok=True);path.write_bytes(raw)
        with Image.open(path) as image:image.verify()
        assets.append({'id':name,'kind':'champion','path':str(path.relative_to(root)),
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'url':url})
    manifest={'schema_version':1,'reference_version':version,'source_patch':None,'mechanics_enabled':False,'assets':assets}
    atomic_json(root/'manifest.json',manifest)
    return {'status':'complete','manifest':str(root/'manifest.json'),'assets':len(assets),'reference_version':version}
