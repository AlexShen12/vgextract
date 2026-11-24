"""Content-compatible task reuse for explicitly experimental Pass 1 runs only."""
import ast
from copy import deepcopy
from pathlib import Path
from .core import digest, read_json, uid
from .media import file_hash

READERS=('readers.py','numeric_reader.py','vl_reader.py','inventory_reader.py',
         'inventory_references.py','reference_bundle.py','intake_profile.py','local_models.py','intake.py')


def reader_signature(directory):
    parts={name:file_hash(directory/name) for name in READERS}
    module=ast.parse((directory/'hybrid_intake.py').read_text())
    reader=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='_IntakeReader')
    methods=[]
    for method in reader.body:
        if not isinstance(method,ast.FunctionDef) or method.name not in {'read','read_fields','read_vl_groups','cache_path'}:continue
        method.body=[n for n in method.body if not (isinstance(n,ast.Assign) and any(
            isinstance(t,ast.Subscript) and isinstance(t.slice,ast.Constant) and
            t.slice.value=='experimental_reader_signature' for t in n.targets))]
        methods.append(ast.dump(method,include_attributes=False))
    parts['hybrid_reader_methods']=methods
    return digest(parts)


def runtime_contract(signature):
    result=deepcopy(signature)
    hardware=result.get('vl',{}).get('runtime',{})
    for name in ('gpu_uuid','total_memory_bytes'):hardware.pop(name,None)
    return result


def reusable_tasks(store, profile, signature, key, targets):
    active=reader_signature(Path(__file__).parent)
    legacy=None
    bundle=store.root.parent.parent
    ledger=bundle/'code_checksums.json';snapshot=bundle/'code/annotation/league'
    if ledger.exists() and snapshot.exists():
        checks=read_json(ledger)
        if all(checks.get(str((snapshot/name).relative_to(bundle)))==file_hash(snapshot/name)
               for name in (*READERS,'hybrid_intake.py')):legacy=reader_signature(snapshot)
    selected={};sources={}
    for path in sorted(store.path('tasks').glob('*.json'),key=lambda p:(p.stat().st_mtime_ns,p.name)):
        task=read_json(path)
        if task.get('status') not in {'ok','unsupported_layout'} or task.get('profile_hash')!=digest(profile):continue
        if not any(abs(task.get('sample_time',task['start'])-t)<1e-6 for t in targets):continue
        if task.get('experimental_reader_signature',legacy)!=active:continue
        runtime=store.path('intake-runtimes/'+task['runtime_id']+'.json')
        if not runtime.exists() or runtime_contract(read_json(runtime))!=runtime_contract(signature):continue
        identifier=uid('task',key,task['start'])
        if identifier not in selected:
            selected[identifier]=task
            sources[identifier]={'original_task_id':task['id'],'original_run_key':task['run_key'],
                                 'runtime_id':task['runtime_id'],'reader_signature':active}
    return selected,sources
