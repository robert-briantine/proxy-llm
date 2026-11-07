#!/usr/bin/env bash
# Lanceur GNOME : réutilise le proxy ou le démarre, puis ouvre le dashboard.
set -Eeuo pipefail
umask 077

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/qwen-on-demand"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/qwen-on-demand"
URL="http://127.0.0.1:8000/"
mkdir -p "$STATE_DIR"

show_error() {
  local message="$1"
  printf '%s\n' "$message" >&2
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --no-markup --title="Qwen On Demand" --width=560 \
      --text="$message

Journal : $STATE_DIR/launcher.log" || true
  elif command -v notify-send >/dev/null 2>&1; then
    notify-send -u critical "Qwen On Demand" "$message — $STATE_DIR/launcher.log" || true
  fi
  exit 1
}

command -v flock >/dev/null 2>&1 || show_error "La commande flock (util-linux) est requise."
# Un double-clic ne doit pas lancer deux constructions Docker simultanées.
exec 9>"$STATE_DIR/launcher.lock"
flock -w 180 9 || show_error "Un démarrage est déjà en cours. Réessaie dans quelques instants."
if [[ -f "$STATE_DIR/launcher.log" ]]; then
  mv -f -- "$STATE_DIR/launcher.log" "$STATE_DIR/launcher.previous.log"
fi
exec >>"$STATE_DIR/launcher.log" 2>&1
trap 'show_error "Le lancement a échoué. Consulte le journal pour le détail."' ERR
printf 'Qwen On Demand — %s\n' "$(date --iso-8601=seconds)"
cd -- "$PROJECT_DIR"

for program in docker curl; do
  command -v "$program" >/dev/null 2>&1 || show_error "La commande $program est requise."
done
docker compose version >/dev/null 2>&1 || show_error "Docker Compose v2 est requis."
docker info >/dev/null 2>&1 || show_error "Docker est inaccessible. Vérifie que Docker est démarré et que ton utilisateur a accès à son socket."

dashboard_ready() {
  curl --silent --fail --output /dev/null --connect-timeout 1 --max-time 3 "$URL"
}

if [[ "$(docker inspect -f '{{.State.Running}}' qwen-on-demand-proxy 2>/dev/null || true)" != true ]] || ! dashboard_ready; then
  if command -v notify-send >/dev/null 2>&1; then
    notify-send "Qwen On Demand" "Démarrage du service… La fenêtre s’ouvrira dès que l’interface sera prête." || true
  fi
  bash "$PROJECT_DIR/start.sh"
fi

ready=false
for ((attempt = 0; attempt < 60; attempt++)); do
  if dashboard_ready; then
    ready=true
    break
  fi
  sleep 1
done
[[ "$ready" == true ]] || show_error "Le service ne répond pas sur $URL. Consulte le journal de lancement et les journaux Docker."

# Libère le verrou avant d'ouvrir la fenêtre, qui peut rester ouverte longtemps.
flock -u 9
exec 9>&-
for browser in google-chrome google-chrome-stable chromium chromium-browser; do
  if command -v "$browser" >/dev/null 2>&1; then
    mkdir -p "$DATA_DIR/browser"
    "$browser" --app="$URL" --class=QwenOnDemand \
      --user-data-dir="$DATA_DIR/browser" --no-first-run --no-default-browser-check \
      >>"$STATE_DIR/browser.log" 2>&1
    exit 0
  fi
done

# Firefox et les autres navigateurs restent utilisables via le navigateur par défaut.
command -v xdg-open >/dev/null 2>&1 || show_error "Installe un navigateur et xdg-utils pour ouvrir l’interface."
xdg-open "$URL"
