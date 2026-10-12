"""Start the model server declared in an experiment YAML; no experiment flags."""

import os
import csv
import subprocess
from pathlib import Path
import sys

import yaml


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "-c":
        raise SystemExit("usage: serve_baseline_config.py -c config.yml")
    spec = yaml.safe_load(Path(sys.argv[2]).read_text())["vllm"]
    if spec.get("cuda_visible_devices") is not None:
        devices = str(spec["cuda_visible_devices"]).split(",")
        if any(device.startswith("GPU-") for device in devices):
            inventory = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"],
                check=True, capture_output=True, text=True,
            )
            indices = {uuid.strip(): index.strip() for uuid, index in csv.reader(inventory.stdout.splitlines())}
            devices = [indices[device] if device.startswith("GPU-") else device for device in devices]
        # vLLM 0.11's NVML device lookup requires numeric PCI-ordered ordinals.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    command = ["vllm", "serve", spec["hf_model_name"]]
    for key in ("host", "port", "served_model_name", "dtype", "max_model_len", "gpu_memory_utilization"):
        command.extend(["--" + key.replace("_", "-"), str(spec[key])])
    command.extend(spec.get("extra_args", []))
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
