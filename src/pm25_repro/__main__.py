"""Reproduce published summary tables and plots, or run explicitly supplied inputs."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name,"1")

def repository_root(explicit: str | None = None) -> Path:
    root=Path(explicit).resolve() if explicit else Path(__file__).resolve().parents[2]
    if not (root/'data/summary/table_registry.json').is_file():
        raise ValueError('Use a complete repository checkout, with data/summary, or specify --repo-root.')
    return root

def verify_manifest(root: Path) -> dict:
    path=root/'MANIFEST.sha256'
    if not path.is_file():raise FileNotFoundError('MANIFEST.sha256 is missing')
    checked=0
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        expected,relative=line.split('  ',1)
        target=(root/relative).resolve()
        if not target.is_relative_to(root):raise ValueError('Manifest path leaves repository')
        if not target.is_file():raise FileNotFoundError(relative)
        h=hashlib.sha256()
        with target.open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):h.update(block)
        if h.hexdigest().lower()!=expected.lower():raise ValueError('Checksum mismatch: '+relative)
        checked+=1
    if checked<50:raise ValueError('Unexpectedly incomplete file manifest')
    return {'status':'passed','files_checked':checked}

def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',help='Path to the complete checkout (defaults to this source checkout)')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('verify','check-results','tables','figures','derive','all','self-check'):
        sp=sub.add_parser(name)
        if name in ('tables','figures','derive','all'):sp.add_argument('--out',default='results/generated')
    sp=sub.add_parser('analysis',help='Build or explicitly execute an analytical command with external inputs')
    sp.add_argument('--config',required=True);sp.add_argument('--component',required=True);sp.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    try:
        root=repository_root(args.repo_root)
        result={}
        if args.command in ('verify','all'):result['checksums']=verify_manifest(root)
        if args.command in ('check-results','all'):
            from .checks import verify_results
            result['scientific_summary_checks']=verify_results(root)
        if args.command in ('tables','figures','derive','all'):
            output=Path(args.out);output=output.resolve() if output.is_absolute() else (root/output).resolve()
            if not output.is_relative_to(root/'results') or output.is_relative_to(root/'results/reference'):
                raise ValueError('Output must be inside results/ and outside results/reference/.')
            output.mkdir(parents=True,exist_ok=True)
            if args.command in ('derive','all'):
                from .derive import derive_summaries
                result['derived_summaries']=derive_summaries(root,output/'derived')
            if args.command in ('tables','all'):
                from .tables import build_tables
                result['tables']=build_tables(root,output/'tables')
            if args.command in ('figures','all'):
                from .figures import render_all
                result['figures']=render_all(root,output/'figures')
            result['scope']='Deterministic replay of author-owned summaries; not a new full model fit or station bootstrap.'
            (output/'reproduction_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf8')
        if args.command=='self-check':
            from .analysis import self_check
            result['canonical_algorithm_self_checks']=self_check(root)
        if args.command=='analysis':
            from .analysis import run_analysis
            result['analysis']=run_analysis(root,args.config,args.component,execute=args.execute)
        print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
        if args.command=='self-check':
            return int(result['canonical_algorithm_self_checks'].get('returncode',2))
        if args.command=='analysis':
            return int(result['analysis'].get('returncode',0))
        return 0
    except (ValueError,FileNotFoundError,KeyError,RuntimeError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        return 2

if __name__=='__main__':raise SystemExit(main())
