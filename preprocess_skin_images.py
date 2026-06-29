import subprocess
import sys

if __name__ == '__main__':
    py_path = sys.executable
    subprocess.run([py_path, "-m", "src.preprocess"])
