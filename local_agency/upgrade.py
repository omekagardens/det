#!/usr/bin/env python3
"""
DET Local Agency - Upgrade Script
==================================

Pulls latest changes from repository and rebuilds everything.

Usage:
    python upgrade.py [options]

Options:
    --check         Check for updates without applying
    --force         Force rebuild even if no updates
    --skip-pull     Skip git pull (rebuild only)
    --skip-models   Skip model updates
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path
from typing import Tuple, Optional


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


def run_command(cmd: list, cwd: Optional[Path] = None,
                capture: bool = False) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture,
            text=True,
            timeout=600,
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent


def get_current_commit(project_root: Path) -> str:
    """Get current git commit hash."""
    code, out, _ = run_command(["git", "rev-parse", "HEAD"], cwd=project_root, capture=True)
    return out.strip() if code == 0 else ""


def get_remote_commit(project_root: Path) -> str:
    """Get remote HEAD commit hash."""
    # Fetch first
    run_command(["git", "fetch"], cwd=project_root, capture=True)

    # Get current branch
    code, branch, _ = run_command(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=project_root,
        capture=True
    )
    branch = branch.strip()

    # Get remote commit
    code, out, _ = run_command(
        ["git", "rev-parse", f"origin/{branch}"],
        cwd=project_root,
        capture=True
    )
    return out.strip() if code == 0 else ""


def check_for_updates(project_root: Path) -> Tuple[bool, str, str]:
    """
    Check if updates are available.

    Returns:
        (has_updates, current_commit, remote_commit)
    """
    print_step("Checking for updates...")

    current = get_current_commit(project_root)
    remote = get_remote_commit(project_root)

    if not current or not remote:
        print_warning("Could not determine commit status")
        return False, current, remote

    has_updates = current != remote

    if has_updates:
        print_success(f"Updates available: {current[:8]} → {remote[:8]}")
    else:
        print_success("Already up to date")

    return has_updates, current, remote


def git_pull(project_root: Path) -> bool:
    """Pull latest changes from remote."""
    print_step("Pulling latest changes...")

    # Check for local changes
    code, status, _ = run_command(["git", "status", "--porcelain"], cwd=project_root, capture=True)
    if status.strip():
        print_warning("You have local changes. Stashing them...")
        run_command(["git", "stash"], cwd=project_root)

    # Pull
    code, out, err = run_command(["git", "pull"], cwd=project_root)

    if code != 0:
        print_error(f"Git pull failed: {err}")
        return False

    print_success("Pull complete")
    return True


def get_venv_python(project_root: Path) -> Path:
    """Get path to venv Python executable."""
    venv_path = project_root / ".venv"
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def update_dependencies(project_root: Path) -> bool:
    """Update Python dependencies."""
    print_step("Updating Python dependencies...")

    venv_python = get_venv_python(project_root)
    requirements = project_root / "requirements.txt"

    if not venv_python.exists():
        print_warning("Virtual environment not found. Run setup_det.py first.")
        return False

    if not requirements.exists():
        print_warning("requirements.txt not found")
        return False

    code, out, err = run_command(
        [str(venv_python), "-m", "pip", "install", "-r", str(requirements), "--upgrade"],
        capture=True,
    )

    if code != 0:
        print_error(f"Dependency update failed: {err}")
        return False

    print_success("Dependencies updated")
    return True


def rebuild_c_kernel(project_root: Path) -> bool:
    """Rebuild the C kernel."""
    print_step("Rebuilding C kernel...")

    build_dir = project_root / "src" / "det_core" / "build"

    if not build_dir.exists():
        build_dir.mkdir(parents=True)

    # Clean old build
    print_step("Cleaning old build...")
    run_command(["make", "clean"], cwd=build_dir, capture=True)

    # Run cmake
    print_step("Running CMake...")
    code, out, err = run_command(
        ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=build_dir,
        capture=True,
    )

    if code != 0:
        print_error(f"CMake failed: {err}")
        return False

    # Build
    print_step("Compiling...")
    code, out, err = run_command(
        ["make", "-j4"],
        cwd=build_dir,
        capture=True,
    )

    if code != 0:
        print_error(f"Build failed: {err}")
        return False

    # Check for library
    lib_name = "libdet_core.dylib" if platform.system() == "Darwin" else "libdet_core.so"
    lib_path = build_dir / lib_name

    if lib_path.exists():
        print_success(f"Built: {lib_path.name}")
        return True
    else:
        # Try with version suffix
        for p in build_dir.glob("libdet_core*.dylib"):
            print_success(f"Built: {p.name}")
            return True
        for p in build_dir.glob("libdet_core*.so"):
            print_success(f"Built: {p.name}")
            return True

    print_error("Library not found after build")
    return False


def run_tests(project_root: Path) -> bool:
    """Run tests to verify the build."""
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
        # Extract test count from output
        for line in out.split('\n'):
            if 'tests passed' in line.lower() or 'passed' in line:
                print_success(line.strip())
                break
        else:
            print_success("Tests passed")
        return True
    else:
        print_warning(f"Some tests failed")
        return False


def check_models() -> bool:
    """Check if models are up to date."""
    print_step("Checking Ollama models...")

    code, out, err = run_command(["ollama", "list"], capture=True)

    if code != 0:
        print_warning("Ollama not running or not installed")
        return False

    # Check for required models
    required = ["llama3.2:3b"]
    installed = out.lower()

    missing = []
    for model in required:
        if model.split(':')[0] not in installed:
            missing.append(model)

    if missing:
        print_warning(f"Missing models: {', '.join(missing)}")
        print_step("Run 'python setup_det.py --models-only' to download")
        return False

    print_success("Required models available")
    return True


def show_changelog(project_root: Path, from_commit: str, to_commit: str):
    """Show changelog between commits."""
    if not from_commit or not to_commit or from_commit == to_commit:
        return

    print_header("Changelog")

    code, out, err = run_command(
        ["git", "log", "--oneline", f"{from_commit}..{to_commit}"],
        cwd=project_root,
        capture=True
    )

    if code == 0 and out.strip():
        for line in out.strip().split('\n')[:20]:  # Show last 20 commits
            print(f"  {line}")

        total = len(out.strip().split('\n'))
        if total > 20:
            print(f"  ... and {total - 20} more commits")


def main():
    parser = argparse.ArgumentParser(
        description="DET Local Agency Upgrade Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Check for updates without applying")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if no updates")
    parser.add_argument("--skip-pull", action="store_true",
                        help="Skip git pull (rebuild only)")
    parser.add_argument("--skip-models", action="store_true",
                        help="Skip model check")

    args = parser.parse_args()

    print_header("DET Local Agency Upgrade")

    project_root = get_project_root()
    print(f"Project root: {project_root}\n")

    # Check for updates
    has_updates, current_commit, remote_commit = check_for_updates(project_root)

    if args.check:
        if has_updates:
            show_changelog(project_root, current_commit, remote_commit)
            print("\nRun 'python upgrade.py' to apply updates")
        return 0

    # Skip if no updates and not forcing
    if not has_updates and not args.force and not args.skip_pull:
        print("\nNo updates available. Use --force to rebuild anyway.")
        return 0

    # Pull updates
    if not args.skip_pull and has_updates:
        if not git_pull(project_root):
            return 1
        show_changelog(project_root, current_commit, remote_commit)

    # Update dependencies
    print_header("Updating Dependencies")
    if not update_dependencies(project_root):
        print_warning("Dependency update had issues, continuing...")

    # Rebuild C kernel
    print_header("Rebuilding C Kernel")
    if not rebuild_c_kernel(project_root):
        print_error("C kernel build failed")
        return 1

    # Run tests
    print_header("Verification")
    run_tests(project_root)

    # Check models
    if not args.skip_models:
        check_models()

    # Done
    print_header("Upgrade Complete")

    new_commit = get_current_commit(project_root)
    print(f"Current version: {new_commit[:8]}")
    print("\nRestart any running DET processes to use the new version.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
