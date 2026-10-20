"""Read-only numerical and sampled-video audit of a UR3 LeRobot v3 dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def quantiles(values):
    return dict(zip(("min", "p50", "p95", "p99", "max"),
                    np.quantile(values, [0, .5, .95, .99, 1]).tolist()))


def read_data(root):
    return pa.concat_tables([pq.read_table(p) for p in sorted((root / "data").rglob("*.parquet"))]).sort_by([("index", "ascending")]).to_pydict()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--compare")
    p.add_argument("--videos", action="store_true")
    args = p.parse_args()
    root = Path(args.dataset)
    output = Path(args.output)
    info = json.loads((root / "meta/info.json").read_text())
    raw = read_data(root)
    state, action = np.asarray(raw["observation.state"], dtype=np.float64), np.asarray(raw["action"], dtype=np.float64)
    index, episodes, task_ids = (np.asarray(raw[k]) for k in ("index", "episode_index", "task_index"))
    frames, times = np.asarray(raw["frame_index"]), np.asarray(raw["timestamp"])
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    task_names = {int(row["task_index"]): str(name) for name, row in tasks.iterrows()}
    metadata = pa.concat_tables([pq.read_table(f) for f in sorted((root / "meta/episodes").rglob("*.parquet"))]).to_pylist()
    meta = {int(r["episode_index"]): r for r in metadata}
    issues = []
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Non-finite state or action")
    if state.shape != (len(index), 17) or action.shape != (len(index), 8):
        raise ValueError("Unexpected state/action shape")
    if not np.array_equal(index, np.arange(len(index))): issues.append("Global indices not contiguous")
    if len(index) != info["total_frames"]: issues.append("Frame count mismatch")
    if len(meta) != info["total_episodes"] or len(metadata) != len(meta): issues.append("Episode metadata count mismatch")
    episode_stats, hashes, by_task = [], [], {}
    pairs = []
    for episode in sorted(np.unique(episodes)):
        selected = np.flatnonzero(episodes == episode)
        a, b = int(selected[0]), int(selected[-1]) + 1
        m = meta[int(episode)]
        ts = np.unique(task_ids[selected])
        if len(ts) != 1: issues.append(f"Multiple tasks in episode {episode}")
        task = int(ts[0])
        by_task.setdefault(task, []).append(int(episode))
        if m["tasks"] != [task_names[task]]: issues.append(f"Task text mismatch in episode {episode}")
        if not np.array_equal(selected, np.arange(a, b)): issues.append(f"Noncontiguous episode {episode}")
        if (a, b, b-a) != (m["dataset_from_index"], m["dataset_to_index"], m["length"]): issues.append(f"Metadata range mismatch episode {episode}")
        if not np.array_equal(frames[selected], np.arange(b-a)): issues.append(f"Frame index mismatch episode {episode}")
        if not np.allclose(times[selected], np.arange(b-a) / info["fps"], atol=1e-4): issues.append(f"Timestamp mismatch episode {episode}")
        u = action[a:b]
        still = (np.linalg.norm(u[:, :3], axis=1) < .0001) & (np.linalg.norm(u[:, 3:6], axis=1) < np.sin(np.deg2rad(.05)/2))
        episode_stats.append({"episode": int(episode), "task_id": task, "task": task_names[task], "frames": b-a,
                              "duration_s": (b-a)/info["fps"], "translation_path_m": float(np.linalg.norm(np.diff(state[a:b, 6:9], axis=0), axis=1).sum()),
                              "gripper_state_range": [float(state[a:b, -1].min()), float(state[a:b, -1].max())],
                              "gripper_command_range": [float(u[:, -1].min()), float(u[:, -1].max())],
                              "stationary_pose_fraction": float(still.mean())})
        hashes.append(hashlib.sha256(np.concatenate([state[a:b], u], axis=1).astype(np.float32).tobytes()).hexdigest())
        pairs.extend(range(a, b-1))
    if len(set(hashes)) != len(hashes): issues.append("Exact duplicate episodes")
    rng = random.Random(42)
    train, validation = [], []
    for task, values in sorted(by_task.items()):
        shuffled = list(values); rng.shuffle(shuffled)
        nval = max(1, round(len(shuffled)*.1))
        validation.extend(shuffled[:nval]); train.extend(shuffled[nval:])
    train, validation = sorted(train), sorted(validation)
    training = np.isin(episodes, train)
    q = state[:, 12:16]
    qnorm = np.linalg.norm(q, axis=1)
    anorm = np.linalg.norm(action[:, 3:7], axis=1)
    if min(qnorm.min(), anorm.min()) < 1e-8: raise ValueError("Zero quaternion")
    q = q/qnorm[:, None]
    ref = q[training][0]
    dot = q@ref
    canonical = q*np.where(dot[:, None] < 0, -1., 1.)
    pairs = np.asarray(pairs)
    rotvec_rotation = Rotation.from_rotvec(state[:, 9:12])
    quaternion_rotation = Rotation.from_quat(q)
    consistency = np.rad2deg((rotvec_rotation.inv()*quaternion_rotation).magnitude())
    dq = action[:, 3:7]/anorm[:, None]
    angles = np.rad2deg(Rotation.from_quat(dq).magnitude())
    state_motion = state[pairs+1, 6:9]-state[pairs, 6:9]
    correlations = []
    for lag in range(-5, 11):
        j = pairs + lag
        valid = (j >= 0) & (j < len(episodes)-1)
        i = pairs[valid]; j = j[valid]
        valid = (episodes[i] == episodes[j]) & (episodes[j] == episodes[j+1])
        i, j = i[valid], j[valid]
        x, y = action[i, :3], state[j+1, 6:9]-state[j, 6:9]
        correlations.append({"lag_frames": lag, "flattened_correlation": float(np.corrcoef(x.ravel(), y.ravel())[0,1]),
                             "least_squares_gain": float(np.sum(x*y)/np.sum(x*x)), "rmse_mm": float(1000*np.sqrt(np.mean(np.sum((x-y)**2,axis=1))))})
    report = {
        "dataset": str(root), "fps": info["fps"], "episodes": len(meta), "frames": len(index), "duration_minutes": len(index)/info["fps"]/60,
        "issues": issues, "task_names": task_names, "episodes_per_task": {k:len(v) for k,v in by_task.items()},
        "episode_lengths": quantiles([s["frames"] for s in episode_stats]),
        "suggested_split": {"seed":42,"train":train,"validation":validation,"train_frames":int(training.sum()),"validation_frames":int((~training).sum())},
        "state_ranges": {"min":state.min(0).tolist(),"max":state.max(0).tolist()},
        "action_ranges": {"min":action.min(0).tolist(),"max":action.max(0).tolist(),"std":action[training].std(0).tolist()},
        "action_translation_mm_per_cycle":quantiles(1000*np.linalg.norm(action[:,:3],axis=1)),
        "action_rotation_deg_per_cycle":quantiles(angles),
        "measured_translation_mm_per_cycle":quantiles(1000*np.linalg.norm(state_motion,axis=1)),
        "state_quaternion_norm":quantiles(qnorm),"action_quaternion_norm":quantiles(anorm),
        "rotvec_quaternion_consistency_deg":quantiles(consistency),
        "state_quaternion_reference":ref.tolist(),"min_abs_reference_dot":float(np.abs(dot).min()),
        "raw_state_quaternion_sign_flips":int(np.sum(np.sum(q[pairs]*q[pairs+1],axis=1)<0)),
        "canonical_state_quaternion_sign_flips":int(np.sum(np.sum(canonical[pairs]*canonical[pairs+1],axis=1)<0)),
        "translation_command_vs_measured_lags":correlations,
        "gripper_command_change_count":int(np.sum(np.abs(action[pairs+1,7]-action[pairs,7])>.01)),
        "episodes_with_gripper_command_range_below_01":[s["episode"] for s in episode_stats if np.ptp(s["gripper_command_range"])<.1],
        "per_episode":episode_stats,
        "semantic_limitations":"Column names and numerical consistency cannot prove command frame, clipping/scaling, control timing or demonstration success; these require recording/execution code and scene review.",
    }
    if args.compare and any((Path(args.compare) / "data").rglob("*.parquet")):
        old = read_data(Path(args.compare))
        oa, os = np.asarray(old["action"]), np.asarray(old["observation.state"])
        report["previous_dataset_comparison"] = {
            "dataset":args.compare,"frames":len(oa),"translation_mm_per_cycle":quantiles(1000*np.linalg.norm(oa[:,:3],axis=1)),
            "rotation_deg_per_cycle":quantiles(np.rad2deg(Rotation.from_quat(oa[:,3:7]).magnitude())),
            "initial_state_max_abs_difference":float(np.max(np.abs(os[0]-state[0]))),
        }
    elif args.compare:
        report["previous_dataset_comparison"] = {"dataset": args.compare, "available": False}
    output.parent.mkdir(parents=True,exist_ok=True)
    if args.videos:
        import torch
        from torchcodec.decoders import VideoDecoder
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        torch.set_num_threads(2)
        cameras=tuple(k for k,v in info["features"].items() if v["dtype"]=="video")
        contacts={task:[values[0],values[len(values)//2],values[-1]] for task,values in by_task.items()}
        contact_images={}
        video_reports=[]; decoded=0
        for camera in cameras:
            for path in sorted((root/"videos"/camera).rglob('*.mp4')):
                associated=[m for m in metadata if path==root/info['video_path'].format(video_key=camera,chunk_index=m[f'videos/{camera}/chunk_index'],file_index=m[f'videos/{camera}/file_index'])]
                decoder=VideoDecoder(str(path),device="cpu",num_ffmpeg_threads=2)
                indices=[]; requests=[]
                for m in associated:
                    start=m[f'videos/{camera}/from_timestamp']
                    duration=m[f'videos/{camera}/to_timestamp']-start
                    if not np.isclose(duration,m['length']/info['fps'],atol=1e-4):issues.append(f"Video duration mismatch episode {m['episode_index']} {camera}")
                    for offset in (0,m['length']//2,m['length']-1):
                        frame=int(round(start*info['fps']))+offset
                        if not 0<=frame<len(decoder):raise ValueError(f"Video index outside file: {path} {frame}")
                        indices.append(frame);requests.append((m['episode_index'],offset))
                batch=decoder.get_frames_at(indices=indices)
                expected_pts=np.asarray(indices)/info['fps']
                max_pts_error=float(np.max(np.abs(batch.pts_seconds.numpy()-expected_pts)))
                if max_pts_error>1/info['fps']+.0001:issues.append(f"PTS mismatch {path}")
                for frame,(episode,offset) in zip(batch.data,requests):
                    task=int(task_ids[meta[episode]['dataset_from_index']])
                    if episode in contacts[task]:contact_images[(episode,camera,offset)]=frame[:,::2,::2].permute(1,2,0).numpy()
                decoded+=len(indices)
                video_reports.append({'path':str(path.relative_to(root)),'codec':decoder.metadata.codec,'frames':len(decoder),'fps':decoder.metadata.average_fps,'samples':len(indices),'max_sample_pts_error_s':max_pts_error})
                print('VIDEO_OK',path.relative_to(root),len(indices),flush=True)
        contact_paths=[]
        for task,selected in contacts.items():
            fig,axes=plt.subplots(len(selected)*len(cameras),3,figsize=(12,3*len(selected)*len(cameras)))
            for e,episode in enumerate(selected):
                length=meta[episode]['length']
                for c,camera in enumerate(cameras):
                    for j,offset in enumerate((0,length//2,length-1)):
                        ax=axes[e*len(cameras)+c,j]
                        ax.imshow(contact_images[(episode,camera,offset)]);ax.axis('off')
                        ax.set_title(f'ep {episode} {camera.split(".")[-1]} t={offset/info["fps"]:.1f}s',fontsize=9)
            fig.suptitle(task_names[task]);fig.tight_layout()
            path=output.parent/f'ur3_20261016_contact_task{task}.png'
            fig.savefig(path,dpi=100);plt.close(fig);contact_paths.append(str(path))
        report['video_audit']={'files':len(video_reports),'sampled_frames':decoded,'reports':video_reports,'contact_sheets':contact_paths,'scope':'All video-file metadata and 3 sampled frames per episode per camera; not a full-frame corruption scan or automatic success classification.'}
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('REPORT',output,json.dumps({k:v for k,v in report.items() if k not in ('per_episode','video_audit','state_ranges','action_ranges','translation_command_vs_measured_lags')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
