#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import subprocess
import time

# Ensure the project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.resolve()))

def main():
    # Directory containing .pth files
    checkpoint_dir = Path("/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/checkpoints")
    # GPUs to use and max concurrent jobs per GPU
    gpus = ["2"]
    max_per_gpu = 2

    # Collect all matching checkpoint files
    ckpts = sorted(checkpoint_dir.glob("final_checkpoint_*.pth"))
    if not ckpts:
        print(f"No checkpoints found in {checkpoint_dir} matching 'final_checkpoint_*.pth'.")
        return

    # check if the results for the ckpt already exists
    results_base_dir = Path("/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/results")
    results_dir_names = [d.name for d in results_base_dir.glob("*") if d.is_dir()]

    # Create a new list with checkpoints that need processing
    ckpts_to_process = []
    for ckpt in ckpts:
        if ckpt.stem in results_dir_names:
            print(f"Results for {ckpt.stem} already exist in {results_base_dir}")
        else:
            ckpts_to_process.append(ckpt)

    # Replace original list with filtered list
    only_process_new = False
    if only_process_new:
        ckpts = ckpts_to_process
    else:
        ckpts = ckpts

    # Directory for logs
    log_dir = Path("/radraid2/dongwoolee/VNet_LungNoduleSeg/richard/logs/parallel_tests")
    log_dir.mkdir(parents=True, exist_ok=True)

    # List of active subprocesses as (process, gpu_id)
    processes = []

    for ckpt in ckpts:
        # Wait until a GPU has a free slot
        while True:
            # Remove completed processes
            processes = [(p, gpu) for (p, gpu) in processes if p.poll() is None]
            # Count how many jobs are running per GPU
            counts = {gpu: 0 for gpu in gpus}
            for (p, gpu) in processes:
                counts[gpu] += 1
            # Find first GPU with available slot
            available = next((gpu for gpu in gpus if counts[gpu] < max_per_gpu), None)
            if available is not None:
                break
            time.sleep(1)

        # Launch test on the selected GPU
        gpu_id = available
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Starting test for {ckpt.name} on GPU {gpu_id}")

        log_file = log_dir / f"{ckpt.stem}.log"
        cmd = [
            sys.executable,
            "-m", "richard.src.test.test",
            "--cuda_visible", gpu_id,
            "--ckpt_name", ckpt.name
        ]

        with open(log_file, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

        processes.append((proc, gpu_id))

    # Wait for all spawned processes to finish
    for (proc, gpu_id) in processes:
        proc.wait()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All tests completed.")

if __name__ == "__main__":
    main() 