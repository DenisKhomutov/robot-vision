"""Build ALIKED/LightGlue landmark tracks using DPVO camera poses.

This is the intended hybrid map builder:
  DPVO -> camera poses only
  ALIKED -> keypoints/descriptors
  LightGlue -> multi-view tracks
  triangulation -> 3D landmarks/runtime descriptor bank
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from module4_map_builder.dpvo_state import load_state


def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--state', type=Path, default=None)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--trajectory', type=Path, required=True)
    ap.add_argument('--calib', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--kpts', type=int, default=4096)
    ap.add_argument('--det-threshold', type=float, default=0.04)
    ap.add_argument('--pair-strides', default='1,2,4')
    ap.add_argument('--min-matches', type=int, default=30)
    ap.add_argument('--min-track-len', type=int, default=2)
    ap.add_argument('--max-track-len', type=int, default=12)
    ap.add_argument('--max-reproj-error', type=float, default=5.0)
    ap.add_argument('--min-parallax-deg', type=float, default=0.3)
    ap.add_argument('--max-landmarks', type=int, default=0)
    return ap.parse_args()


def read_calib(path: Path):
    vals=np.loadtxt(path,dtype=float).reshape(-1)
    fx,fy,cx,cy=vals[:4]
    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.0]],float)
    dist=vals[4:].astype(float) if len(vals)>4 else np.zeros(4)
    return K,dist


def quat_wxyz_to_R(q):
    w,x,y,z=q
    n=math.sqrt(w*w+x*x+y*y+z*z)+1e-12
    w,x,y,z=w/n,x/n,y/n,z/n
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y],
    ],float)


def load_tum(path: Path):
    poses={}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        p=line.split()
        if len(p)<8: continue
        t=int(round(float(p[0])))
        trans=np.array([float(p[1]),float(p[2]),float(p[3])],float)
        qx,qy,qz,qw=[float(v) for v in p[4:8]]
        R=quat_wxyz_to_R([qw,qx,qy,qz])
        T=np.eye(4); T[:3,:3]=R; T[:3,3]=trans
        poses[t]=T
    return poses


def projection_from_c2w(Tcw, K):
    Twc=np.linalg.inv(Tcw)
    return K @ Twc[:3]


def load_image_tensor(path, device, half):
    bgr=cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None: raise RuntimeError(f'cannot read {path}')
    rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    t=torch.from_numpy(rgb.transpose(2,0,1)).to(device)
    return t.half() if half else t


class DSU:
    def __init__(self): self.p={}
    def find(self,x):
        if x not in self.p: self.p[x]=x
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra!=rb: self.p[rb]=ra


def triangulate_track(obs, keypoints, P_by_frame):
    # use best baseline pair: max camera center distance among observations
    frames=[o[0] for o in obs]
    best=None; bestd=-1
    centers={f: np.linalg.inv(P_by_frame[f][1])[:3,3] for f in frames}
    # P_by_frame stores (P, Twc), center = c2w[:3,3] easier but enough
    for i in range(len(frames)):
        for j in range(i+1,len(frames)):
            d=np.linalg.norm(centers[frames[i]]-centers[frames[j]])
            if d>bestd: bestd=d; best=(frames[i],frames[j])
    if best is None: return None
    f0,f1=best
    kp0=keypoints[f0][dict(obs)[f0]]
    kp1=keypoints[f1][dict(obs)[f1]]
    Xh=cv2.triangulatePoints(P_by_frame[f0][0], P_by_frame[f1][0], kp0.reshape(2,1), kp1.reshape(2,1))
    if abs(Xh[3,0])<1e-9: return None
    X=(Xh[:3,0]/Xh[3,0]).astype(float)
    return X


def reproj_errors(X, obs, keypoints, P_by_frame):
    errs=[]; depths=[]
    Xh=np.r_[X,1.0]
    for f,k in obs:
        P,Twc=P_by_frame[f]
        x=P@Xh
        if abs(x[2])<1e-9:
            errs.append(np.inf); depths.append(-np.inf); continue
        uv=x[:2]/x[2]
        errs.append(float(np.linalg.norm(uv-keypoints[f][k])))
        depths.append(float(x[2]))
    return np.array(errs), np.array(depths)


def main():
    args=parse_args()
    run_dir=args.run_dir.resolve()
    state=load_state((args.state or run_dir/'dpvo_state.pt').resolve())
    K,dist=read_calib(args.calib)
    poses_tum=load_tum(args.trajectory)
    tstamps=np.asarray(state['graph']['tstamps']).astype(int)
    n=int(state['graph']['n'])
    frame_names=[f'frame_{int(t)+1:06d}.jpg' for t in tstamps]
    frame_paths=[args.images.resolve()/name for name in frame_names]
    missing=[p for p in frame_paths if not p.exists()]
    if missing: raise SystemExit(f'missing images: {missing[:3]}')
    missing_pose=[int(t) for t in tstamps if int(t) not in poses_tum]
    if missing_pose: raise SystemExit(f'missing poses for tstamps: {missing_pose[:5]}')

    sys.path.insert(0,str(Path.cwd()))
    from lightglue import ALIKED, LightGlue
    from module2_localization.core.model_weights import configure_local_model_weights
    configure_local_model_weights()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    half=device=='cuda'
    extractor=ALIKED(max_num_keypoints=args.kpts, detection_threshold=args.det_threshold).eval().to(device)
    matcher=LightGlue(features='aliked').eval().to(device)
    if half: extractor=extractor.half()

    features=[]; keypoints=[]; descriptors=[]
    print(f'extract {n} keyframes', flush=True)
    for i,p in enumerate(frame_paths):
        img=load_image_tensor(p,device,half)
        with torch.inference_mode(): f=extractor.extract(img)
        # LightGlue runs in float here; keep CPU bank descriptors separately as fp16.
        for _key, _value in list(f.items()):
            if torch.is_tensor(_value) and _value.is_floating_point():
                f[_key] = _value.float()
        features.append(f)
        keypoints.append(f['keypoints'][0].detach().cpu().numpy().astype(np.float32))
        descriptors.append(f['descriptors'][0].detach().cpu().numpy().astype(np.float16))
        if (i+1)%25==0: print(f'features {i+1}/{n}', flush=True)

    dsu=DSU(); match_pairs=0; match_count=0
    strides=[int(s) for s in args.pair_strides.split(',') if s.strip()]
    print('match pairs', strides, flush=True)
    for s in strides:
        for i in range(0,n-s):
            j=i+s
            with torch.inference_mode(): out=matcher({'image0':features[i], 'image1':features[j]})
            m=out['matches'][0].detach().cpu().numpy().astype(np.int32)
            if len(m)<args.min_matches: continue
            match_pairs+=1; match_count+=len(m)
            for a,b in m:
                dsu.union((i,int(a)),(j,int(b)))
        print(f'stride {s} done pairs={match_pairs} matches={match_count}', flush=True)

    groups=defaultdict(list)
    for i,kps in enumerate(keypoints):
        for k in range(len(kps)):
            if (i,k) in dsu.p:
                groups[dsu.find((i,k))].append((i,k))
    raw_tracks=[]
    for obs in groups.values():
        by_frame={}
        for f,k in obs:
            by_frame.setdefault(f,k)
        if len(by_frame)>=args.min_track_len:
            items=sorted(by_frame.items())[:args.max_track_len]
            raw_tracks.append(items)
    print(f'raw_tracks={len(raw_tracks)}', flush=True)

    P_by_frame={}
    route=[]
    for i,t in enumerate(tstamps):
        Tcw=poses_tum[int(t)]
        Twc=np.linalg.inv(Tcw)
        P_by_frame[i]=(K@Twc[:3], Twc)
        route.append(Tcw[:3,3])
    route=np.asarray(route,np.float32)

    xyz=[]; desc=[]; node=[]; track_len=[]; reproj=[]; parallax=[]
    landmark_observations=[]
    for obs in raw_tracks:
        if len(obs)<2: continue
        X=triangulate_track(obs,keypoints,P_by_frame)
        if X is None or not np.isfinite(X).all(): continue
        errs,depths=reproj_errors(X,obs,keypoints,P_by_frame)
        if np.any(depths<=0): continue
        mederr=float(np.median(errs))
        if mederr>args.max_reproj_error: continue
        centers=[]
        for f,_ in obs:
            centers.append(poses_tum[int(tstamps[f])][:3,3])
        centers=np.asarray(centers)
        rays=X[None,:]-centers
        rays=rays/(np.linalg.norm(rays,axis=1,keepdims=True)+1e-9)
        maxang=0.0
        for a in range(len(rays)):
            for b in range(a+1,len(rays)):
                maxang=max(maxang, math.degrees(math.acos(float(np.clip(np.dot(rays[a],rays[b]),-1,1)))))
        if maxang<args.min_parallax_deg: continue
        # descriptor = mean normalized descriptors over observations
        ds=np.stack([descriptors[f][k].astype(np.float32) for f,k in obs])
        d=ds.mean(axis=0); d=d/(np.linalg.norm(d)+1e-9)
        xyz.append(X.astype(np.float32)); desc.append(d.astype(np.float16))
        node.append(int(round(np.median([f for f,_ in obs]))))
        track_len.append(len(obs)); reproj.append(mederr); parallax.append(maxang)
        landmark_observations.append(list(obs))
        if args.max_landmarks and len(xyz)>=args.max_landmarks: break

    if not xyz: raise SystemExit('no landmarks survived filters')
    xyz=np.stack(xyz).astype(np.float32); desc=np.stack(desc).astype(np.float16)
    owner=np.arange(len(xyz),dtype=np.int32); node=np.asarray(node,dtype=np.int32)
    out=args.out or (run_dir/'dpvo_aliked_tracks_bank.npz')
    obs_offsets=[0]
    obs_frames=[]; obs_keypoint_indices=[]; obs_xy=[]
    for observations in landmark_observations:
        for frame_idx,keypoint_idx in observations:
            obs_frames.append(frame_idx)
            obs_keypoint_indices.append(keypoint_idx)
            obs_xy.append(keypoints[frame_idx][keypoint_idx])
        obs_offsets.append(len(obs_frames))
    np.savez_compressed(out, desc=desc, owner=owner, xyz=xyz, node=node, route=route,
                        track_len=np.asarray(track_len,np.int16), reproj=np.asarray(reproj,np.float32),
                        parallax=np.asarray(parallax,np.float32), image_names=np.asarray(frame_names),
                        tstamps=tstamps.astype(np.int64), K=K.astype(np.float64), dist=dist.astype(np.float64),
                        obs_offsets=np.asarray(obs_offsets,np.int64), obs_frames=np.asarray(obs_frames,np.int32),
                        obs_keypoint_indices=np.asarray(obs_keypoint_indices,np.int32), obs_xy=np.asarray(obs_xy,np.float32))
    summary={'out':str(out),'frames':n,'match_pairs':match_pairs,'matches':match_count,'raw_tracks':len(raw_tracks),
             'landmarks':len(xyz),'median_track_len':float(np.median(track_len)),
             'median_reproj':float(np.median(reproj)),'p90_reproj':float(np.percentile(reproj,90)),
             'median_parallax':float(np.median(parallax))}
    out.with_suffix('.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
