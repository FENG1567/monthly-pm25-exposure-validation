"""Refresh the release checksum inventory after intentional repository edits."""
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','__pycache__','.venv','venv','.pytest_cache','build','dist'}
lines=[]
for path in sorted(ROOT.rglob('*')):
    rel=path.relative_to(ROOT)
    if not path.is_file() or any(p in EXCLUDED or p.endswith('.egg-info') for p in rel.parts):continue
    if rel.as_posix()=='MANIFEST.sha256' or (rel.parts[0]=='results' and len(rel.parts)>1 and rel.parts[1]!='reference') or path.suffix=='.pyc':continue
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    lines.append(digest.hexdigest()+'  '+rel.as_posix())
(ROOT/'MANIFEST.sha256').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('Manifest files:',len(lines))
