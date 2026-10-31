#!/usr/bin/env bash
# Arrête le service sans supprimer les caches ni les images Docker.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! docker info >/dev/null 2>&1; then echo "Docker inaccessible." >&2; exit 1; fi
# Arrêt via API : refus si requêtes en cours ; ne pas forcer la fin d'une requête.
if command -v curl >/dev/null 2>&1 && curl --silent --fail --max-time 130 \
   -X POST http://127.0.0.1:8000/api/stop >/dev/null; then
  echo "vLLM arrêté via le proxy."
else
  if [ "$(docker inspect -f '{{.State.Running}}' qwen-on-demand-vllm 2>/dev/null || true)" = "true" ]; then
    echo "ATTENTION : impossible de confirmer l'inactivité du modèle ; arrêt annulé." >&2
    echo "Vérifie /api/status et arrête d'abord le modèle depuis l'interface." >&2
    exit 1
  fi
fi
docker compose stop --timeout 20 proxy
printf 'Proxy arrêté ; caches et images conservés. Pour relancer : ./start.sh\n'
