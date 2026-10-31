#!/usr/bin/env bash
# Démarre le proxy. Le conteneur vLLM est ensuite créé à la demande par le proxy.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  echo "Docker et Docker Compose v2 sont requis." >&2; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker est inaccessible ; vérifie les permissions." >&2; exit 1
fi
mkdir -p data "$HOME/.cache/huggingface" "$HOME/.cache/vllm" \
  "$HOME/.cache/flashinfer" "$HOME/.triton" "$HOME/.cache/b12x"
# Conserver le conteneur vLLM V3 existant si un ancien service tourne encore.
# Lors du premier changement, le proxy le remplacera quand il sera inactif.
echo "Construction et démarrage du proxy (vLLM reste à la demande)..."
docker compose up -d --build proxy
if command -v curl >/dev/null 2>&1; then
  for i in {1..30}; do
    if curl --silent --fail --max-time 2 http://127.0.0.1:8000/api/status >/dev/null; then break; fi
    sleep 1
  done
fi
printf '\nInterface : http://127.0.0.1:8000/\nAPI OpenAI : http://127.0.0.1:8000/v1\n'
printf 'Modèles : Qwen3.8 27B et Qwen3.8 Flash-Next.\n'
printf 'Changement de modèle depuis le dashboard, sans interruption des requêtes actives.\n\n'
docker compose ps -a
