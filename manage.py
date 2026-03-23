#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path


def _reexec_with_project_venv() -> None:
    """Force manage.py commands to run with ./ .venv interpreter when available."""
    project_root = Path(__file__).resolve().parent
    venv_root = project_root / ".venv"
    venv_python = venv_root / "bin" / "python"
    if not venv_python.exists():
        return

    try:
        current_prefix = Path(sys.prefix).resolve()
        target_prefix = venv_root.resolve()
    except OSError:
        return

    if current_prefix == target_prefix:
        return

    os.environ.setdefault("VIRTUAL_ENV", str(venv_root))
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


def main():
    """Run administrative tasks."""
    _reexec_with_project_venv()
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "excel_agent.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
