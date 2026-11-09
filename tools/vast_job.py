"""Run the benchmark suite on rented vast.ai GPUs (one suite process per instance).

Uses the local `vastai` CLI (API key already configured) and ssh/scp with the
registered SSH key. Creating instances costs money: `launch` only runs with an
explicit list of offer ids, and every instance should be destroyed after `fetch`.
Code is uploaded from the working tree (tracked + untracked, not ignored); data
are downloaded on the instance from the official Hugging Face dataset.

    python tools/vast_job.py offers --gpu RTX_3090 -n 6
    python tools/vast_job.py launch OFFER_ID [OFFER_ID ...]
    python tools/vast_job.py setup                     # wait for ssh, upload code, install, download data
    python tools/vast_job.py run --datasets traffic electricity weather ETTm1 ETTm2 ETTh1 ETTh2 exchange_rate -- --amp bf16
    python tools/vast_job.py status
    python tools/vast_job.py fetch                     # results -> results/vast/<instance>/
    python tools/vast_job.py destroy
"""
import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results" / "vast_instances.json"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
REMOTE = "/workspace/GroupedRoPE"
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=20"]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_suite import SUITE  # noqa: E402

# Relative cost per dataset (training steps + test windows x channels), used to balance instances.
COST = {"ETTh1": 1, "ETTh2": 1, "exchange_rate": 1, "ETTm1": 2, "ETTm2": 2, "weather": 3, "electricity": 6, "traffic": 8}


def vast(*args):
    out = subprocess.run(["vastai", *args, "--raw"], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(out.stderr or out.stdout)
    return json.loads(out.stdout) if out.stdout.strip() else None


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else []


def save_state(ids):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(ids))


def ssh_target(instance):
    url = subprocess.run(["vastai", "ssh-url", str(instance)], capture_output=True, text=True).stdout.strip()
    if not url.startswith("ssh://"):
        raise RuntimeError(f"no ssh url yet for {instance}: {url}")
    host, port = url[len("ssh://"):].rsplit(":", 1)
    return host, port


def ssh(instance, command, check=True, capture=False, timeout=None):
    host, port = ssh_target(instance)
    return subprocess.run(["ssh", *SSH_OPTS, "-p", port, host, command], check=check,
                          capture_output=capture, text=True, timeout=timeout)


def scp(instance, src, dst, upload=True):
    host, port = ssh_target(instance)
    pair = [src, f"{host}:{dst}"] if upload else [f"{host}:{src}", dst]
    subprocess.run(["scp", *SSH_OPTS, "-r", "-P", port, *pair], check=True)


def code_tarball():
    listed = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files", "--cached", "--others",
                             "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in listed.splitlines():
            if (ROOT / name).is_file():
                tar.add(ROOT / name, arcname=name)
    path = ROOT / "results" / "code_upload.tar.gz"
    path.write_bytes(buffer.getvalue())
    return path


def cmd_offers(args):
    query = (f"gpu_name={args.gpu} num_gpus=1 reliability>0.98 inet_down>200 cuda_vers>=12.4 "
             f"disk_space>=40 rentable=true verified=true")
    offers = sorted(vast("search", "offers", query), key=lambda o: o["dph_total"])[:args.n]
    for o in offers:
        print(f"offer {o['id']:>9}  {o['gpu_name']:10} ${o['dph_total']:.3f}/h  reliability {o['reliability2']:.3f}  "
              f"down {o['inet_down']:.0f}Mbps  {o.get('geolocation', '')}")
    print(f"total for these {len(offers)}: ${sum(o['dph_total'] for o in offers):.2f}/h")


def cmd_launch(args):
    ids = load_state()
    for offer in args.offers:
        result = vast("create", "instance", str(offer), "--image", IMAGE, "--disk", "40", "--ssh", "--direct")
        ids.append(result["new_contract"])
        print(f"offer {offer} -> instance {result['new_contract']}")
    save_state(ids)


def cmd_setup(args):
    tarball = code_tarball()
    datasets = " ".join(args.datasets)
    for instance in load_state():
        for attempt in range(60):
            try:
                ssh(instance, "true", capture=True)
                break
            except (RuntimeError, subprocess.CalledProcessError):
                time.sleep(10)
        else:
            raise SystemExit(f"instance {instance} not reachable over ssh")
        ssh(instance, f"mkdir -p {REMOTE}")
        scp(instance, str(tarball), f"{REMOTE}/code.tar.gz")
        ssh(instance, f"cd {REMOTE} && tar xzf code.tar.gz && python -c 'import torch; print(torch.__version__, "
                      f"torch.cuda.get_device_name(0))' && python tools/download_data.py {datasets}")
        print(f"instance {instance}: ready")


def balance(datasets, n):
    buckets = [[] for _ in range(n)]
    for d in sorted(datasets, key=lambda d: -COST[d]):
        min(buckets, key=lambda b: sum(COST[x] for x in b)).append(d)
    return buckets


def cmd_run(args):
    instances = load_state()
    extra = " ".join(args.extra)
    for instance, bucket in zip(instances, balance(args.datasets, len(instances))):
        if not bucket:
            continue
        if ssh(instance, "pgrep -f benchmark_suite.py", check=False, capture=True).stdout.strip():
            print(f"instance {instance}: suite already running, skipped")
            continue
        # stdin from /dev/null and setsid so the ssh session returns immediately.
        command = (f"cd {REMOTE} && setsid nohup python tools/benchmark_suite.py --datasets {' '.join(bucket)} "
                   f"--horizons {' '.join(map(str, args.horizons))} {extra} > suite.log 2>&1 < /dev/null &")
        try:  # some hosts keep the session open after a detached start; the job keeps running
            ssh(instance, command, check=False, capture=True, timeout=30)
        except subprocess.TimeoutExpired:
            pass
        print(f"instance {instance}: {bucket}")


def cmd_status(args):
    for instance in load_state():
        out = ssh(instance, f"cd {REMOTE} && tail -n 4 suite.log; ls results/benchmark 2>/dev/null | grep -c json; "
                            "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
                  check=False, capture=True)
        print(f"== instance {instance}\n{out.stdout}{out.stderr}")


def cmd_fetch(args):
    for instance in load_state():
        target = ROOT / "results" / "vast" / str(instance)
        target.mkdir(parents=True, exist_ok=True)
        scp(instance, f"{REMOTE}/results/benchmark/.", str(target), upload=False)
        scp(instance, f"{REMOTE}/suite.log", str(target), upload=False)
        print(f"instance {instance} -> {target}")


def cmd_destroy(args):
    for instance in load_state():
        subprocess.run(["vastai", "destroy", "instance", str(instance)], input="y\n", text=True, check=False)
    save_state([])
    print("remaining instances:", [i["id"] for i in vast("show", "instances")])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("offers")
    p.add_argument("--gpu", default="RTX_3090")
    p.add_argument("-n", type=int, default=6)
    p = sub.add_parser("launch")
    p.add_argument("offers", nargs="+", type=int)
    p = sub.add_parser("setup")
    p.add_argument("--datasets", nargs="+", default=list(SUITE))
    p = sub.add_parser("run")
    p.add_argument("--datasets", nargs="+", choices=sorted(SUITE), default=list(SUITE))
    p.add_argument("--horizons", type=int, nargs="+", default=[96])
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments after -- go to benchmark_suite.py")
    for name in ("status", "fetch", "destroy"):
        sub.add_parser(name)
    args = parser.parse_args()
    if getattr(args, "extra", None) and args.extra[0] == "--":
        args.extra = args.extra[1:]
    globals()[f"cmd_{args.command}"](args)


if __name__ == "__main__":
    main()
