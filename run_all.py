import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(1)

def main():
    # run_cmd("python -u src/data_collection.py")
    run_cmd("python -u src/sequence_features.py")
    run_cmd("python -u src/structure_features.py")
    run_cmd("python -u src/preprocessing.py")
    run_cmd("python -u src/similarity_clustering.py")
    run_cmd("python -u -m experiments.run_experiments")
    run_cmd("python -u src/generate_report.py")

if __name__ == "__main__":
    main()
