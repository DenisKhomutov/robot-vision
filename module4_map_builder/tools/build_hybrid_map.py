"""One-command DPVO-pose + ALIKED/LightGlue landmark map pipeline."""
from __future__ import annotations
import argparse, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def run(*args):
    print('+',' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),check=True,cwd=ROOT.parent)

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',required=True)
    ap.add_argument('--video',type=Path,required=True)
    ap.add_argument('--calib',type=Path,required=True)
    ap.add_argument('--stride',type=int,default=8)
    ap.add_argument('--skip',type=int,default=0)
    ap.add_argument('--pair-strides',default='1,2,4,8')
    ap.add_argument('--max-reproj-error',type=float,default=5.0)
    ap.add_argument('--min-parallax-deg',type=float,default=0.2)
    ap.add_argument('--force',action='store_true')
    args=ap.parse_args()
    out=(ROOT/'out'/'dpvo'/args.name).resolve()
    video=args.video.resolve(); calib=args.calib.resolve()
    if out.exists():
        if not args.force: raise SystemExit(f'output exists: {out} (use --force)')
        shutil.rmtree(out)
    py=sys.executable
    run(py,'-m','module4_map_builder.tools.run_dpvo_with_state','--name',args.name,'--video',video,'--calib',calib,'--stride',args.stride,'--skip',args.skip)
    run(py,'-m','module4_map_builder.tools.extract_dpvo_keyframes','--state',out/'dpvo_state.pt','--video',video,'--calib',calib,'--out',out/'images','--stride',args.stride,'--skip',args.skip)
    bank=out/'dpvo_aliked_tracks_bank.npz'
    run(py,'-m','module4_map_builder.tools.build_dpvo_aliked_tracks_bank','--run-dir',out,'--images',out/'images','--trajectory',out/'trajectory_tum.txt','--calib',calib,'--out',bank,'--pair-strides',args.pair_strides,'--min-matches',30,'--min-track-len',2,'--max-reproj-error',args.max_reproj_error,'--min-parallax-deg',args.min_parallax_deg)
    run(py,'-m','module4_map_builder.tools.export_runtime_map','--bank',bank,'--trajectory',out/'trajectory_tum.txt','--images',out/'images','--out-dir',out/'runtime_map')
    for view,black in [('colmap_view',False),('colmap_view_black',True)]:
        cmd=[py,'-m','module4_map_builder.tools.export_dpvo_aliked_bank_colmap','--bank',bank,'--trajectory',out/'trajectory_tum.txt','--images-src',out/'images','--out-dir',out/view]
        if black: cmd.append('--black')
        run(*cmd)
    if shutil.which('colmap'):
        run('colmap','model_analyzer','--path',out/'colmap_view'/'sparse_text')
    print(f'completed hybrid map: {out}')
    return 0
if __name__=='__main__': raise SystemExit(main())
