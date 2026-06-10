import subprocess
import sys
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) < 2:
    print("Usage: python launcher.py [backend|frontend]")
    sys.exit(1)

if sys.argv[1] == "backend":
    subprocess.run([
        sys.executable, "-m", "uvicorn", "app:app",
        "--host", "0.0.0.0", "--port", "8000", "--workers", "1"
    ])
elif sys.argv[1] == "frontend":
    os.chdir("frontend")
    subprocess.run(["npx", "vite", "--host", "--port", "5173"])