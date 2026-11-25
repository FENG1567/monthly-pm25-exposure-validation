"""Render the current manuscript display from unchanged registered source panels."""
from pathlib import Path
import csv,gzip,hashlib,json,subprocess,sys

def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def render_all(repo_root,output_dir):
    root=Path(repo_root).resolve();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    registry=json.loads((root/'data/summary/figure_registry.json').read_text(encoding='utf-8-sig'))
    panel_root=(root/'data/summary/figure_panels').resolve()
    inputs={}
    for entry in registry['figures']:
        print('CHECK_FIGURE_INPUTS '+entry['figure'], file=sys.stderr, flush=True)
        sources={}
        for key,rel in entry['panels'].items():
            rel=rel['path'] if isinstance(rel,dict) else rel
            path=(root/rel).resolve()
            if panel_root not in path.parents:raise ValueError('Panel path leaves published panel directory')
            opener=gzip.open if path.suffix=='.gz' else open
            with opener(path,'rt',encoding='utf-8-sig',newline='') as f:
                reader=csv.reader(f);next(reader);count=sum(1 for _ in reader)
            if count!=entry['panel_rows'][key]:raise ValueError(f'Panel row mismatch: {entry["figure"]}:{key}')
            sources[key]=dict(filename=rel,rows=count,sha256=_sha(path))
        inputs[entry['figure']]=sources
    modules=[]
    for script in ['display_main.py','display_supp_first.py','display_supp_last.py']:
        print('RENDER_MODULE '+script, file=sys.stderr, flush=True)
        run=subprocess.run([sys.executable,str(Path(__file__).parent/script),'--repo-root',str(root),'--outdir',str(out)],capture_output=True,text=True)
        if run.returncode:raise RuntimeError(f'{script} failed: {run.stderr[-4000:]}')
        modules.append(json.loads(run.stdout))
    records=[]
    for entry in registry['figures']:
        name=entry['figure'];outputs={}
        for ext in ['pdf','png','svg','tiff']:
            path=out/(name+'.'+ext)
            if not path.is_file():raise FileNotFoundError(path)
            outputs[ext]=dict(filename=path.name,sha256=_sha(path))
        record=dict(figure=name,caption=entry['caption'],panel_sources=inputs[name],panel_rows={k:v['rows'] for k,v in inputs[name].items()},outputs=outputs)
        (out/(name+'_replay_manifest.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2))
        records.append(record)
    manifest=dict(method='source_replay_from_published_panel_csvs',model_refit=False,bootstrap_or_resampling=False,figures=records)
    (out/'figure_replay_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest
