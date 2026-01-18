#!/usr/bin/env python3
"""
DET Local Agency - Setup Script
================================

Automated setup for DET Local Agency including:
- System requirements check
- Virtual environment creation
- Python dependencies installation
- C kernel build
- Ollama model downloads

Usage:
    python setup_det.py [options]

Options:
    --models-only     Only download LLM models
    --build-only      Only build C kernel
    --check           Check requirements without installing
    --skip-models     Skip model downloads
    --minimal         Minimal setup (skip optional features)
"""

import os
import sys
import subprocess
import shutil
import argparse
import platform
from pathlib import Path
from typing import List, Tuple, Optional


# Configuration
REQUIRED_PYTHON_VERSION = (3, 10)
DEFAULT_MODELS = [
    "llama3.2:3b",       # Primary reasoning model
    "qwen2.5-coder:3b",  # Code specialist
]
OPTIONAL_MODELS = [
    "llama3.2:1b",       # Fast responses
    "deepseek-r1:1.5b",  # Math/reasoning specialist
    "phi4-mini:3.8b",    # Compact all-rounder
]


def print_header(text: str):
    """Print a formatted header."""
    width = 60
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}\n")


def print_step(text: str):
    """Print a step indicator."""
    print(f"[*] {text}")


def print_success(text: str):
    """Print a success message."""
    print(f"[+] {text}")


def print_error(text: str):
    """Print an error message."""
    print(f"[!] {text}", file=sys.stderr)


def print_warning(text: str):
    """Print a warning message."""
    print(f"[?] {text}")


def run_command(cmd: List[str], cwd: Optional[Path] = None,
                capture: bool = False) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    print_step(f"Checking Python version...")
    version = sys.version_info
    required = REQUIRED_PYTHON_VERSION

    if version >= required:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} (>= {required[0]}.{required[1]})")
        return True
    else:
        print_error(f"Python {required[0]}.{required[1]}+ required, found {version.major}.{version.minor}")
        return False


def check_ollama() -> bool:
    """Check if Ollama is installed and running."""
    print_step("Checking Ollama...")

    # Check if installed
    code, out, err = run_command(["ollama", "--version"], capture=True)
    if code != 0:
        print_error("Ollama not found. Install from: https://ollama.com")
        return False

    print_success(f"Ollama installed: {out.strip()}")

    # Check if running
    code, out, err = run_command(["ollama", "list"], capture=True)
    if code != 0:
        print_warning("Ollama may not be running. Start with: ollama serve")
        return False

    print_success("Ollama is running")
    return True


def check_cmake() -> bool:
    """Check if CMake is installed."""
    print_step("Checking CMake...")

    code, out, err = run_command(["cmake", "--version"], capture=True)
    if code != 0:
        print_error("CMake not found. Install via: brew install cmake (macOS) or apt install cmake (Linux)")
        return False

    version_line = out.split('\n')[0] if out else "unknown"
    print_success(f"CMake: {version_line}")
    return True


def check_compiler() -> bool:
    """Check if C compiler is available."""
    print_step("Checking C compiler...")

    # Try clang first (macOS), then gcc
    for compiler in ["clang", "gcc"]:
        code, out, err = run_command([compiler, "--version"], capture=True)
        if code == 0:
            version_line = out.split('\n')[0] if out else "unknown"
            print_success(f"Compiler: {version_line}")
            return True

    print_error("No C compiler found. Install Xcode CLT (macOS) or build-essential (Linux)")
    return False


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent


def create_venv(project_root: Path) -> bool:
    """Create virtual environment if it doesn't exist."""
    venv_path = project_root / ".venv"

    if venv_path.exists():
        print_success(f"Virtual environment exists: {venv_path}")
        return True

    print_step("Creating virtual environment...")
    code, out, err = run_command([sys.executable, "-m", "venv", str(venv_path)])

    if code != 0:
        print_error(f"Failed to create venv: {err}")
        return False

    print_success(f"Created: {venv_path}")
    return True


def get_venv_python(project_root: Path) -> Path:
    """Get path to venv Python executable."""
    venv_path = project_root / ".venv"
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def install_dependencies(project_root: Path) -> bool:
    """Install Python dependencies."""
    print_step("Installing Python dependencies...")

    venv_python = get_venv_python(project_root)
    requirements = project_root / "requirements.txt"

    if not requirements.exists():
        print_error(f"requirements.txt not found: {requirements}")
        return False

    code, out, err = run_command(
        [str(venv_python), "-m", "pip", "install", "-r", str(requirements)],
        capture=True,
    )

    if code != 0:
        print_error(f"Failed to install dependencies:\n{err}")
        return False

    print_success("Dependencies installed")
    return True


def build_c_kernel(project_root: Path) -> bool:
    """Build the C kernel."""
    print_step("Building C kernel...")

    det_core_dir = project_root / "src" / "det_core"
    build_dir = det_core_dir / "build"

    # Create build directory
    build_dir.mkdir(parents=True, exist_ok=True)

    # Run cmake
    print_step("Running CMake...")
    code, out, err = run_command(
        ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=build_dir,
        capture=True,
    )

    if code != 0:
        print_error(f"CMake failed:\n{err}")
        return False

    # Build
    print_step("Compiling...")
    code, out, err = run_command(
        ["make", "-j4"],
        cwd=build_dir,
        capture=True,
    )

    if code != 0:
        print_error(f"Build failed:\n{err}")
        return False

    # Check for library
    lib_name = "libdet_core.dylib" if platform.system() == "Darwin" else "libdet_core.so"
    lib_path = build_dir / lib_name

    if lib_path.exists():
        print_success(f"Built: {lib_path}")
        return True
    else:
        print_error(f"Library not found: {lib_path}")
        return False


def download_model(model_name: str) -> bool:
    """Download an Ollama model."""
    print_step(f"Downloading {model_name}...")

    # Check if already downloaded
    code, out, err = run_command(["ollama", "list"], capture=True)
    if code == 0 and model_name.split(':')[0] in out:
        print_success(f"Already downloaded: {model_name}")
        return True

    # Download
    code, out, err = run_command(["ollama", "pull", model_name])

    if code != 0:
        print_error(f"Failed to download {model_name}")
        return False

    print_success(f"Downloaded: {model_name}")
    return True


def download_models(models: List[str]) -> Tuple[int, int]:
    """Download multiple models. Returns (success_count, total_count)."""
    success = 0
    for model in models:
        if download_model(model):
            success += 1
    return success, len(models)


def run_tests(project_root: Path) -> bool:
    """Run the test suite to verify installation."""
    print_step("Running verification tests...")

    venv_python = get_venv_python(project_root)
    test_file = project_root / "src" / "python" / "test_det.py"

    if not test_file.exists():
        print_warning("Test file not found, skipping verification")
        return True

    code, out, err = run_command(
        [str(venv_python), str(test_file)],
        capture=True,
    )

    if code == 0:
        print_success("All tests passed")
        return True
    else:
        print_warning(f"Some tests failed:\n{out}\n{err}")
        return False


def print_next_steps(project_root: Path):
    """Print next steps for the user."""
    venv_activate = ".venv/bin/activate"
    if platform.system() == "Windows":
        venv_activate = ".venv\\Scripts\\activate"

    print_header("Setup Complete!")

    print(f"""
Next steps:

1. Activate the virtual environment:
   source {venv_activate}

2. Start Ollama (if not running):
   ollama serve

3. Run the CLI:
   cd src/python
   python det_cli.py

4. Or start the web visualization:
   python -c "from det import DETCore, create_harness, run_server; \\
              core = DETCore(); run_server(core=core)"
   Then open http://127.0.0.1:8420

5. Run tests:
   python src/python/test_det.py
   python src/python/test_phase6.py

For more information, see GETTING_STARTED.md
""")


def main():
    parser = argparse.ArgumentParser(
        description="DET Local Agency Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models-only", action="store_true",
                        help="Only download LLM models")
    parser.add_argument("--build-only", action="store_true",
                        help="Only build C kernel")
    parser.add_argument("--check", action="store_true",
                        help="Check requirements without installing")
    parser.add_argument("--skip-models", action="store_true",
                        help="Skip model downloads")
    parser.add_argument("--minimal", action="store_true",
                        help="Minimal setup (required models only)")
    parser.add_argument("--all-models", action="store_true",
                        help="Download all models (including optional)")

    args = parser.parse_args()

    print_header("DET Local Agency Setup")

    project_root = get_project_root()
    print(f"Project root: {project_root}\n")

    # Check requirements
    all_ok = True

    all_ok &= check_python_version()
    all_ok &= check_ollama()
    all_ok &= check_cmake()
    all_ok &= check_compiler()

    if args.check:
        if all_ok:
            print_success("\nAll requirements met!")
        else:
            print_error("\nSome requirements not met.")
        return 0 if all_ok else 1

    if not all_ok:
        print_error("\nCannot proceed - fix requirements above.")
        return 1

    # Models only mode
    if args.models_only:
        print_header("Downloading Models")
        models = DEFAULT_MODELS
        if args.all_models:
            models = DEFAULT_MODELS + OPTIONAL_MODELS
        success, total = download_models(models)
        print(f"\nDownloaded {success}/{total} models")
        return 0 if success == total else 1

    # Build only mode
    if args.build_only:
        print_header("Building C Kernel")
        return 0 if build_c_kernel(project_root) else 1

    # Full setup
    print_header("Setting Up Environment")

    if not create_venv(project_root):
        return 1

    if not install_dependencies(project_root):
        return 1

    print_header("Building C Kernel")

    if not build_c_kernel(project_root):
        return 1

    # Download models
    if not args.skip_models:
        print_header("Downloading LLM Models")

        models = DEFAULT_MODELS
        if args.all_models:
            models = DEFAULT_MODELS + OPTIONAL_MODELS

        success, total = download_models(models)
        print(f"\nDownloaded {success}/{total} models")

        if success == 0:
            print_warning("No models downloaded - you'll need to download them manually")

    # Run verification
    print_header("Verification")
    run_tests(project_root)

    # Print next steps
    print_next_steps(project_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
