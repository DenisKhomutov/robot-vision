"""Extract exactly the source frames used by DPVO keyframes."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import cv2
import numpy as np
from module3_map_builder.dpvo_state import load_state


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--video',type=Path,required=True)
    ap.add_argument('--calib',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stride',type=int,required=True)
    ap.add_argument('--skip',type=int,default=0)
    ap.add_argument('--jpg-quality',type=int,default=95)
    ap.add_argument('--force',action='store_true')
    args=ap.parse_args()
    if args.out.exists() and args.force: shutil.rmtree(args.out)
    if args.out.exists() and any(args.out.iterdir()): raise SystemExit(f'output is not empty: {args.out} (use --force)')
    args.out.mkdir(parents=True,exist_ok=True)
    state=load_state(args.state)
    target=set(map(int,np.asarray(state['graph']['tstamps'])))
    vals=np.loadtxt(args.calib,dtype=float).reshape(-1)
    K=np.array([[vals[0],0,vals[2]],[0,vals[1],vals[3]],[0,0,1.]],float)
    dist=vals[4:]
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened(): raise SystemExit(f'cannot open {args.video}')
    for _ in range(args.skip):
        if not cap.read()[0]: raise SystemExit('video ended during skip')
    timestamp=0; saved=0
    while True:
        frame=None; ok=False
        for _ in range(args.stride):
            ok,frame=cap.read()
            if not ok: break
        if not ok: break
        if timestamp in target:
            if dist.size: frame=cv2.undistort(frame,K,dist)
            h,w=frame.shape[:2]; frame=frame[:h-h%16,:w-w%16]
            dst=args.out/f'frame_{timestamp+1:06d}.jpg'
            if not cv2.imwrite(str(dst),frame,[int(cv2.IMWRITE_JPEG_QUALITY),args.jpg_quality]): raise RuntimeError(dst)
            saved+=1
        timestamp+=1
    cap.release()
    if saved!=len(target): raise SystemExit(f'saved {saved}, expected {len(target)}')
    print(f'saved {saved} DPVO keyframes from {timestamp} processed frames to {args.out}')
    return 0
if __name__=='__main__': raise SystemExit(main())
