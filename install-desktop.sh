#!/usr/bin/env bash
# Installation par utilisateur, sans sudo ni démarrage du moteur d'inférence.
set -Eeuo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v python3 >/dev/null 2>&1 || { echo "Python 3 est requis." >&2; exit 1; }
if [[ "${1:-}" != "" && "${1:-}" != --uninstall ]] || (( $# > 1 )); then
  echo "Usage : bash install-desktop.sh [--uninstall]" >&2
  exit 1
fi

python3 - "$PROJECT_DIR" "${1:-}" <<'PY'
import os
from pathlib import Path
import shutil
import subprocess
import sys

project = Path(sys.argv[1])
data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
applications = data_home / "applications"
entry = applications / "qwen-on-demand.desktop"
icon = data_home / "icons/hicolor/scalable/apps/qwen-on-demand.svg"

desktop_dir = None
if shutil.which("xdg-user-dir"):
    result = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        candidate = Path(result.stdout.strip())
        # XDG peut renvoyer HOME lorsque le bureau est désactivé.
        if candidate != Path.home() and candidate.is_dir():
            desktop_dir = candidate
shortcut = desktop_dir / entry.name if desktop_dir else None

if sys.argv[2] == "--uninstall":
    for path in (entry, icon, shortcut):
        if path is not None:
            path.unlink(missing_ok=True)
    print("Lanceur désinstallé. Projet, services, modèles et profil navigateur conservés.")
else:
    def desktop_string(value):
        return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    def exec_argument(value):
        # Deux niveaux : guillemets Exec, puis échappement de la valeur desktop.
        escaped = str(value).replace("%", "%%")
        for char in ("\\", '"', "`", "$"):
            escaped = escaped.replace(char, "\\" + char)
        return desktop_string('"' + escaped + '"')

    applications.mkdir(parents=True, exist_ok=True)
    icon.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(project / "assets/qwen-on-demand.svg", icon)
    contents = "\n".join([
        "[Desktop Entry]",
        "Version=1.0",
        "Type=Application",
        "Name=Qwen On Demand",
        "Comment=Démarrer et ouvrir le tableau de bord vLLM du DGX Spark",
        "Exec=/bin/bash " + exec_argument(project / "scripts/launch-desktop.sh"),
        "Icon=" + desktop_string(icon),
        "Terminal=false",
        "Categories=Development;",
        "Keywords=Qwen;vLLM;DGX;Spark;IA;",
        "StartupWMClass=QwenOnDemand",
        "StartupNotify=false",
        "",
    ])
    entry.write_text(contents, encoding="utf-8")
    entry.chmod(0o755)
    if shortcut:
        shutil.copyfile(entry, shortcut)
        shortcut.chmod(0o755)
        if shutil.which("gio"):
            subprocess.run(["gio", "set", str(shortcut), "metadata::trusted", "true"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Raccourci bureau : {shortcut}")
    print(f"Application installée : {entry}")
    print("Ouvre le menu des applications et cherche « Qwen On Demand ».")

if applications.is_dir() and shutil.which("update-desktop-database"):
    subprocess.run(["update-desktop-database", str(applications)], check=False)
PY
