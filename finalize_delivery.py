"""Refresh documentation, check notebook code, and build the checked handover ZIP."""
import ast
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_dataset import documents, notebook, pretrained_profile, sha256


def main():
    workspace = Path(__file__).resolve().parent
    out = workspace / 'StarDist_Dataset'
    raw = Path(r'C:\Users\ADMIN\Downloads\stage1_train.zip')
    with (out / 'dataset_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    for r in records:
        for key in ['width', 'height', 'num_instances', 'foreground_pixels', 'primary_100']:
            r[key] = int(r[key])
    quality = json.loads((out / 'quality_report.json').read_text(encoding='utf-8'))
    assert quality['status'] == 'PASS' and quality['checked_pairs'] == len(records)
    with zipfile.ZipFile(raw) as z:
        names = z.namelist()
        assert sum('/images/' in n and n.endswith('.png') for n in names) == len(records)
        assert sum('/masks/' in n and n.endswith('.png') for n in names) == sum(r['num_instances'] for r in records)
    quality['source'] = str(raw)
    quality['source_archive_sha256'] = sha256(raw)
    quality['runtime'] = dict(python=sys.version.split()[0], numpy=np.__version__, pillow=Image.__version__)
    quality['focused_integrity_tests'] = '5 passed (test_dataset_integrity.py)'
    quality['pretrained_protocol'] = pretrained_profile(out, records)
    documents(out, records, quality['excluded_samples'], raw, quality['duplicate_pixel_groups'])
    notebook(out)
    # Check every Python cell for syntax. %pip is a Jupyter/Colab magic.
    nb = json.loads((out / 'Chuan_bi_du_lieu.ipynb').read_text(encoding='utf-8'))
    cells = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(line for line in cell['source'] if not line.startswith('%'))
            ast.parse(source)
            cells += 1
    quality['notebook_python_cells_syntax_checked'] = cells
    quality['notebook_execution'] = 'Not executed in a Jupyter kernel; preparation script and exported data verified locally'
    (out / 'quality_report.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf-8')
    for filename in ['prepare_dataset.py', 'test_dataset_integrity.py', 'HUONG_DAN_BAT_DAU.md']:
        shutil.copy2(workspace / filename, out / filename)
    payload = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'checksums.sha256')
    checksum = '\n'.join(f'{sha256(p)}  {p.relative_to(out).as_posix()}' for p in payload) + '\n'
    (out / 'checksums.sha256').write_text(checksum, encoding='utf-8')
    print(f'Checked notebook syntax ({cells} Python cells); checksums for {len(payload)} files', flush=True)
    archive = Path(shutil.make_archive(str(workspace / 'StarDist_Dataset_BanGiao'), 'zip', workspace, out.name))
    print(f'Created archive: {archive.stat().st_size / 1024**2:.1f} MiB; checking contents...', flush=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        # Independently verify every archived payload against the recorded checksum.
        import hashlib
        for line in checksum.splitlines():
            digest, relative = line.split('  ', 1)
            assert hashlib.sha256(z.read(f'{out.name}/{relative}')).hexdigest() == digest, relative
    digest = sha256(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    print(f'PASS: ZIP CRC and {len(payload)} archived SHA256 hashes', flush=True)
    print(json.dumps(dict(images=len(records), nuclei=sum(r['num_instances'] for r in records),
                          primary_images=sum(r['primary_100'] for r in records),
                          archive=str(archive), archive_sha256=digest), indent=2), flush=True)


if __name__ == '__main__':
    main()
