"""Restore the exact public source files listed in references/download_manifest.json."""
from pathlib import Path
import hashlib,json,urllib.request

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'references/download_manifest.json').read_text())
for entry in manifest:
    if entry['status']!='verified':continue
    path=root/entry['path']
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temp=path.with_suffix(path.suffix+'.partial')
        with urllib.request.urlopen(entry['url'],timeout=180) as response,temp.open('wb') as out:
            while chunk:=response.read(4*1024*1024):out.write(chunk)
        temp.rename(path)
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=entry['sha256']:raise ValueError(f'Checksum mismatch: {path.name}')
    print('Verified',entry['path'],flush=True)
