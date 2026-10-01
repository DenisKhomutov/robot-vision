"""Export fixed DPVO poses and real ALIKED/LightGlue tracks as a COLMAP text model."""
from __future__ import annotations
import argparse, math, os, shutil
from pathlib import Path
import cv2
import numpy as np


def rot_from_q(q):
    q=np.asarray(q,float); q/=np.linalg.norm(q); w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],[2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],[2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])

def qvec_from_R(R):
    m=R; tr=np.trace(m)
    if tr>0:
        s=np.sqrt(tr+1)*2; q=np.array([.25*s,(m[2,1]-m[1,2])/s,(m[0,2]-m[2,0])/s,(m[1,0]-m[0,1])/s])
    elif m[0,0]>m[1,1] and m[0,0]>m[2,2]:
        s=np.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2; q=np.array([(m[2,1]-m[1,2])/s,.25*s,(m[0,1]+m[1,0])/s,(m[0,2]+m[2,0])/s])
    elif m[1,1]>m[2,2]:
        s=np.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2; q=np.array([(m[0,2]-m[2,0])/s,(m[0,1]+m[1,0])/s,.25*s,(m[1,2]+m[2,1])/s])
    else:
        s=np.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2; q=np.array([(m[1,0]-m[0,1])/s,(m[0,2]+m[2,0])/s,(m[1,2]+m[2,1])/s,.25*s])
    q/=np.linalg.norm(q)
    return -q if q[0]<0 else q

def load_poses(path):
    poses={}
    for line in path.read_text().splitlines():
        p=line.split()
        if len(p)<8: continue
        stamp=int(round(float(p[0]))); C=np.asarray(p[1:4],float); qx,qy,qz,qw=map(float,p[4:8])
        poses[stamp]=(C,rot_from_q([qw,qx,qy,qz]))
    return poses

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',type=Path,required=True)
    ap.add_argument('--trajectory',type=Path,required=True)
    ap.add_argument('--images-src',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--black',action='store_true')
    ap.add_argument('--force',action='store_true')
    args=ap.parse_args()
    if args.out_dir.exists() and args.force: shutil.rmtree(args.out_dir)
    if args.out_dir.exists(): raise SystemExit(f'output exists: {args.out_dir} (use --force)')
    sparse=args.out_dir/'sparse_text'; images=args.out_dir/'images'; sparse.mkdir(parents=True); images.mkdir()
    b=np.load(args.bank,allow_pickle=True)
    required={'xyz','reproj','K','image_names','tstamps','obs_offsets','obs_frames','obs_xy'}
    missing=required-set(b.files)
    if missing: raise SystemExit(f'bank missing real tracks: {sorted(missing)}')
    names=[str(x) for x in b['image_names']]; stamps=b['tstamps'].astype(int); xyz=b['xyz']; err=b['reproj']
    off=b['obs_offsets'].astype(int); frames=b['obs_frames'].astype(int); xy=b['obs_xy']; K=b['K']; poses=load_poses(args.trajectory)
    first=cv2.imread(str(args.images_src/names[0])); h,w=first.shape[:2]
    per_image=[[] for _ in names]; tracks=[]; colors=[]; cache={}
    for pid in range(len(xyz)):
        track=[]; samples=[]
        for oi in range(off[pid],off[pid+1]):
            fi=int(frames[oi]); point2d_idx=len(per_image[fi]); per_image[fi].append((xy[oi],pid+1)); track.append((fi+1,point2d_idx))
            if fi not in cache: cache[fi]=cv2.imread(str(args.images_src/names[fi]))
            x,y=np.rint(xy[oi]).astype(int)
            if 0<=x<w and 0<=y<h: samples.append(cache[fi][y,x,::-1])
        tracks.append(track); colors.append(np.median(samples,axis=0).astype(int) if samples else np.array([128]*3))
    (sparse/'cameras.txt').write_text(f'1 PINHOLE {w} {h} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n')
    lines=[]; max_center_error=0.
    for fi,(name,stamp) in enumerate(zip(names,stamps)):
        src=args.images_src/name; dst=images/name
        try: os.link(src,dst)
        except OSError: shutil.copy2(src,dst)
        C,Rcw=poses[int(stamp)]; Rwc=Rcw.T; q=qvec_from_R(Rwc); t=-Rwc@C
        max_center_error=max(max_center_error,float(np.linalg.norm(-rot_from_q(q).T@t-C)))
        lines.append(f'{fi+1} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} {t[0]:.17g} {t[1]:.17g} {t[2]:.17g} 1 {name}')
        lines.append(' '.join(f'{p[0]} {p[1]} {pid}' for p,pid in per_image[fi]))
    if max_center_error>1e-9: raise RuntimeError(f'pose conversion error: {max_center_error}')
    (sparse/'images.txt').write_text('\n'.join(lines)+'\n')
    points=[]
    for i,X in enumerate(xyz):
        c=(0,0,0) if args.black else tuple(map(int,colors[i])); tr=' '.join(f'{iid} {idx}' for iid,idx in tracks[i])
        points.append(f'{i+1} {X[0]} {X[1]} {X[2]} {c[0]} {c[1]} {c[2]} {float(err[i])} {tr}')
    (sparse/'points3D.txt').write_text('\n'.join(points)+'\n')
    print(f'poses={len(names)} points={len(xyz)} observations={sum(map(len,tracks))} max_center_error={max_center_error:.3g}')
    return 0
if __name__=='__main__': raise SystemExit(main())
