# G1 Cube-Sorting: Sim + WBC + GR00T VLA — Setup & Run

Verified working 2026-08-06 on Ubuntu 24.04, RTX 5090 (sm_120).

Layout:

```
/mnt/ssd2/Isaac-GR00T                 @ b995540   VLA code + checkpoint + dataset
/mnt/ssd2/vla_for_simulation_testing  @ f6b05b0   sim (gear_sonic) + WBC (deploy-wbc-on-robot)
/mnt/ssd2/dsrl                        DSRL project + venv + docker
```

---

## 0. System prerequisites

Docker must store data on ssd2 — the WBC image is 20 GB and will fill the root disk otherwise.
`/tmp` is on `/`, so a full root disk also breaks tooling.

```bash
sudo mkdir -p /mnt/ssd2/docker
sudo systemctl stop docker
echo '{ "data-root": "/mnt/ssd2/docker" }' | sudo tee /etc/docker/daemon.json
sudo systemctl start docker
docker info | grep "Docker Root Dir"      # -> /mnt/ssd2/docker
```

Keep all caches off the home disk. Put these in `~/.bashrc`:

```bash
export HF_HOME=/mnt/ssd2/huggingface_cache
export UV_CACHE_DIR=/mnt/ssd2/uv_cache
export TORCH_HOME=/mnt/ssd2/torch_home
export UV_HTTP_TIMEOUT=600
```

`UV_HTTP_TIMEOUT=600` is not optional: the default times out on the large NVIDIA CUDA
wheels and **uv still exits 0**, leaving a half-installed venv that fails later with a
confusing `ModuleNotFoundError`.

The NVIDIA Container Toolkit is **not** installed and is **not needed** — the WBC ONNX
policies run on CPU. Do not pass `--gpus all` / `--runtime=nvidia` to the WBC container;
it fails with `could not select device driver`.

---

## 1. Hugging Face access

The VLA backbone `nvidia/Cosmos-Reason2-2B` is a **gated** repo. Two separate things are
required, and having only one gives a confusing 401/403:

1. **Access granted** — accept the license at
   https://huggingface.co/nvidia/Cosmos-Reason2-2B ("Expand to review and access").
   Confirm at https://huggingface.co/settings/gated-repos → `ACCEPTED`.
2. **A token with gated-repo scope** — a fine-grained token with no scopes authenticates
   (`whoami` works) but cannot read repo contents, so gated fetches still 403 even when
   ACCEPTED. Either use a **classic** token with role `read`, or tick
   *"Read access to contents of all public gated repos you can access"*.

```bash
printf '%s' 'hf_YOUR_TOKEN' > /mnt/ssd2/huggingface_cache/token

# verify: must print 200
curl -s -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $(cat /mnt/ssd2/huggingface_cache/token)" \
  https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json
```

---

## 2. Repos, checkpoint, dataset

```bash
cd /mnt/ssd2
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git
git clone --depth 1 https://github.com/MethdAI-Modinoid/vla_for_simulation_testing.git

cd /mnt/ssd2/Isaac-GR00T && mkdir -p datasets checkpoints
git clone https://huggingface.co/datasets/rahul-ai-01/red_green_blue_cube_sorting_in_sim \
    datasets/red_green_blue_cube_sorting_in_sim
git clone https://huggingface.co/rahul-ai-01/red_green_blue_cube_sorting_in_sim \
    checkpoints/red_green_blue_cube_sorting_in_sim
```

**Verify all three shards materialised.** LFS can fetch the blob into `.git/lfs` and fail
to check it out, leaving a missing file that only surfaces much later:

```bash
cd /mnt/ssd2/Isaac-GR00T/checkpoints/red_green_blue_cube_sorting_in_sim
ls -la checkpoint-22000/*.safetensors     # expect 3 files, ~12.6 GB total
git lfs status | grep deleted             # if any, run:
git lfs checkout                          # recovers from the local .git/lfs cache
```

Optional, reclaims ~12 GB after a confirmed load: `git lfs prune`.

---

## 3. Host venv (VLA / policy server / DSRL)

```bash
mkdir -p /mnt/ssd2/{claude_tmp,uv_cache,torch_home,huggingface_cache}
cd /mnt/ssd2/dsrl
uv venv --python 3.10 .venv

uv pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

uv pip install --python .venv/bin/python \
    transformers==4.57.3 diffusers==0.35.1 einops==0.8.1 peft==0.17.1 \
    omegaconf==2.3.0 albumentations==1.4.18 termcolor dm-tree \
    opencv-python-headless pandas tyro jsonlines pyarrow \
    msgpack msgpack-numpy pyzmq lmdb gymnasium \
    numpy mujoco onnxruntime pyyaml safetensors pillow pytest
```

`transformers` must be **4.57.3**. On 5.x, `Gr00tN1d7Config` fails to build with
`non-default argument 'diffusion_model_cfg' follows default argument` — 5.x added
default-valued fields to `PretrainedConfig`, breaking the dataclass field ordering.

Verify:

```bash
cd /mnt/ssd2/dsrl
.venv/bin/python -c "
import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
a=torch.randn(1024,1024,device='cuda'); print('matmul ok:', bool((a@a).isfinite().all()))"
# -> 2.7.1+cu128 NVIDIA GeForce RTX 5090 (12, 0)
```

---

## 4. Sim venv (Part A) — their installer

```bash
cd /mnt/ssd2/vla_for_simulation_testing
bash install_scripts/install_mujoco_sim.sh      # creates .venv_sim
.venv_sim/bin/python -c "import mujoco, gear_sonic, unitree_sdk2py; print('ok')"
```

---

## 5. WBC image (Part B)

```bash
docker pull nvgear/gr00t_wbc:latest
cd /mnt/ssd2/dsrl/docker && docker build -t gr00t_wbc:dsrl -f Dockerfile .
```

`gr00t_wbc:dsrl` adds what the base image lacks: `transformers==4.57.3`, an entrypoint that
sources ROS, and `ROS_DOMAIN_ID=99` / `ROS_LOCALHOST_ONLY=1` / `PYTHONPATH=/opt/isaac-gr00t`.
Isaac-GR00T is **mounted, not baked**, so DSRL patches to it take effect without a rebuild.

---

## 6. Running it — four processes

### Terminal 1 — sim (host)

```bash
cd /mnt/ssd2/vla_for_simulation_testing
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
    --enable-offscreen --enable-image-publish --camera-port 5555
```

Expect `[DDS] domain_id=99 interface=lo` and `3 cameras on ZMQ port 5555`.
The `create domain error` line right after is a harmless pre-existing double-init.

### Terminal 2 — WBC (container)

```bash
docker run -it --rm --network=host --privileged \
  -v /mnt/ssd2/vla_for_simulation_testing/deploy-wbc-on-robot:/root/Projects/GR00T-WholeBodyControl \
  -v /mnt/ssd2/Isaac-GR00T:/opt/isaac-gr00t:ro \
  gr00t_wbc:dsrl bash
```

Inside:

```bash
cd /root/Projects/GR00T-WholeBodyControl
grep DOMAIN_ID gr00t_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml   # must be 99

/root/venv/bin/python -u gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
  --wbc_version gear_wbc \
  --wbc_model_path policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx \
  --wbc_policy_class GIDecoupledWholeBodyPolicy \
  --interface lo --simulator None --control_frequency 50 \
  --no-enable_waist --with_hands --no-high_elbow_pose --no-enable_gravity_compensation
```

The `-v` mount of the WBC repo is **load-bearing**: the image's baked-in copy still has
`DOMAIN_ID: 0`.

Then:
- **`]`** in this terminal → activates the policy (`use_policy_action = True`). Without it
  the WBC receives GR00T's targets and silently ignores them.
- **`9`** in the MuJoCo viewer → releases the elastic band.
- `o` deactivates, `w/s/a/d` walk, `z` zero velocity command.

Healthy output: `observe_avg ~1.8ms`, `total_loop_avg ~3.2ms`.
`Invalid simulator type: None` is expected — it means "use the external sim".

### Terminal 3 — GR00T policy server (host)

```bash
cd /mnt/ssd2/Isaac-GR00T
export PYTHONPATH=/mnt/ssd2/Isaac-GR00T HF_HOME=/mnt/ssd2/huggingface_cache NO_ALBUMENTATIONS_UPDATE=1
/mnt/ssd2/dsrl/.venv/bin/python -u -m gr00t.eval.run_gr00t_server \
  --model_path /mnt/ssd2/Isaac-GR00T/checkpoints/red_green_blue_cube_sorting_in_sim/checkpoint-22000 \
  --embodiment_tag unitree_g1_full_body_with_waist_height_nav_cmd \
  --device cuda --host 0.0.0.0 --port 5560
```

**Port 5560 is deliberate** — the server's default is 5555, which collides with the sim's
camera port.

First run downloads the Cosmos backbone (~4.6 GB) even though the checkpoint already
contains those weights: the code builds the model from the repo, then overwrites it with
yours. One-time, then cached. Wait for `✓ Server ready — listening on 0.0.0.0:5560`
(~2 min after download). Load takes ~20 s once cached; ~7.8 GB VRAM.

### Terminal 4 — inference client (second shell into the WBC container)

```bash
docker ps --format '{{.Names}}'          # find the WBC container's name
docker exec -it "$(docker ps --format '{{.Names}}' | head -1)" bash

docker exec -it wizardly_greider bash #here
cd /root/Projects/GR00T-WholeBodyControl/sim_inference
/root/venv/bin/python -u run_Inferene_without_client_for_test.py \
  --policy_host localhost --policy_port 5560 \
  --camera_host localhost --camera_port 5555 \
  --no-add_stereo_camera --frequency 20
```

- **`--frequency 20`** — the dataset is 20 fps. The script's default is `1`, which plays
  motion 20× too slow.
- **`--no-add_stereo_camera`** — the policy's modality config accepts only `ego_view`.
  The default `True` adds stereo keys it never trained on.
- **No timeout flag exists.** Ctrl+C to stop. Demos take **36–53 s** for all three cubes,
  so allow 90 s+ before judging.

Healthy output: `[Loop N] Action idx: 1/16 … 16/16` cycling.

---
