# Smart vLLM On Demand — Qwen3.8 27B ↔ Flash-Next — DGX Spark 128 Go

Version 4 du proxy fourni. **Même adresse OpenAI pour Qwen Code**, avec changement de modèle à la demande uniquement lorsque **toutes les requêtes déjà admises sont terminées**, y compris les réponses SSE en streaming.

- Interface : `http://127.0.0.1:8000/`
- API OpenAI : `http://127.0.0.1:8000/v1`
- Nom public inchangé : `qwen3.8:27b-mtp` (quelle que soit la version du modèle chargée)
- Conteneurs Docker : **un conteneur proxy permanent**, **un seul conteneur d'inférence vLLM**, remplacé lors d'un changement de modèle. Deux conteneurs au total en fonctionnement normal (pas deux conteneurs d'inférence).
- Sélection mémorisée dans `data/selection.json` ; caches de modèles préservés.
- Les traces d'appels d'outils, leur analyse et la détection de boucles **ne sont pas implémentées dans cette version**.

## Prérequis

DGX Spark 128 Go, Ubuntu ARM64, Docker Compose v2, runtime NVIDIA et une capacité disque suffisante. Le modèle Flash-Next pèse environ **96 Gio sur le disque** : le premier téléchargement et le premier chargement peuvent être longs. Pour Flash-Next, cette version reprend le checkpoint et **l'image vLLM spéciale B12X** recommandés dans la recette NVIDIA (2026-10-15) ; une image vLLM générique ne suffit pas.

**Important :** le code et le scénario de changement sont testés automatiquement avec des simulations Docker/HTTP, **pas exécutés sur un vrai DGX Spark** dans cet environnement. Les performances >20 tok/s, l'utilisation mémoire réelle à 256K et la compatibilité des outils Qwen Code doivent être vérifiées sur ta machine.

## Installation et migration de la v3

Sur le DGX Spark, arrête d'abord tes sessions Qwen Code / les appels en cours avant de remplacer les fichiers du projet. Décompresse cette nouvelle version dans le dossier de ton choix (ou sauvegarde la v3 puis remplace ses fichiers). Il n'est pas nécessaire de supprimer les modèles Hugging Face.

```bash
unzip Smart_vLLM_On_Demand_DGX_Spark_v4.zip
cd qwen-on-demand
chmod +x start.sh stop.sh
./start.sh
```

Le script lance **uniquement le proxy**. Le conteneur d'inférence `qwen-on-demand-vllm` est démarré/créé quand on clique sur « Démarrer », qu'un client envoie une requête, ou qu'un autre profil est choisi. S'il existe encore un conteneur v3 avec les mêmes paramètres 27B, il peut être réutilisé tant que le modèle 27B reste sélectionné. Au premier changement vers Flash-Next, il est arrêté puis remplacé. Ne lance pas simultanément un autre serveur vLLM utilisant les 128 Go.

Attention en cas de migration d'un ancien fichier Compose : `docker compose` peut indiquer que le conteneur vLLM v3 est un **orphelin**. C'est intentionnel : ce conteneur est désormais géré par le proxy via l'API Docker. **N'utilise pas `docker compose down --remove-orphans`** pendant une session, car cela peut supprimer le conteneur d'inférence.

## Changement de modèle

Depuis l'interface Web, sélectionne « Qwen3.8 Flash-Next » puis clique sur **Changer de modèle**. Le changement est planifié même si une requête est en cours :

1. Le proxy marque le changement comme « en attente ».
2. Les nouvelles requêtes `/v1/*` patientent (elles ne passent pas à l'ancien modèle).
3. Les requêtes déjà envoyées à vLLM, y compris les flux SSE, finissent normalement.
4. Lorsqu'il n'en reste plus, le proxy stoppe et supprime le conteneur vLLM précédent.
5. Il crée le conteneur avec **l'image, le checkpoint et les options du nouveau modèle**, puis attend `/health`.
6. Le trafic reprend automatiquement vers le nouveau modèle, sous le **même alias OpenAI**.
7. Si le chargement échoue, le proxy tente de restaurer le modèle précédent et affiche le diagnostic.

Le conteneur proxy n'est jamais redémarré lors du changement. Les nouvelles requêtes HTTP peuvent toutefois expirer **côté client** si le délai d'attente de Qwen Code est inférieur au temps nécessaire au chargement du modèle. Éviter d'envoyer de nouvelles requêtes pendant les longues phases de remplacement.

La bascule ne conserve **pas** le KV cache du modèle précédent : Qwen Code doit renvoyer le contexte à chaque requête, comme d'habitude.

## API

```bash
# Profils disponibles
curl http://127.0.0.1:8000/api/models

# État du moteur, profil actif, switch éventuel, nombre de requêtes
curl http://127.0.0.1:8000/api/status

# Passer à Flash-Next : renvoie HTTP 202 immédiatement, switch asynchrone
curl -X POST http://127.0.0.1:8000/api/switch \
  -H 'Content-Type: application/json' \
  -d '{"model":"flash-next"}'

# Revenir au 27B
curl -X POST http://127.0.0.1:8000/api/switch \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen27"}'

# Démarrer le modèle sélectionné à la demande
curl -X POST http://127.0.0.1:8000/api/start

# Arrêter vLLM (409 si requêtes en cours)
curl -X POST http://127.0.0.1:8000/api/stop

# Appel avec le même alias quel que soit le modèle actif
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8:27b-mtp","messages":[{"role":"user","content":"Bonjour"}],"max_tokens":64}'
```

Dans `/api/status`, `switch_phase` vaut `waiting`, `replacing`, `rollback` ou `idle`. `switch_error` indique la dernière erreur éventuelle. Le champ `engine_profile` permet de savoir quel profil tourne réellement. Pour une transition réussie, `pending_model` redevient `null` et `active_model` est mis à jour.

## Configuration des modèles

Édite **`proxy/models.json`** pour modifier les paramètres vLLM, les images ou les profils ; redémarre ensuite **uniquement le proxy** :

```bash
docker compose up -d --build proxy
```

Ce fichier contient notamment :

| Profil | Checkpoint | Image |
|---|---|---|
| `qwen27` | `PassingByPixels/Qwen3.8-27B-NVFP4` | `vllm/vllm-openai:qwen38` |
| `flash-next` | `nvidia/Qwen3.8-Flash-Next-FP4-Local` | `eugr/spark-vllm-b12x:nightly-20261015` |

Les arguments vLLM des deux profils sont indépendants. Le 27B conserve les options de **ton archive v3** (`gpu-memory-utilization=0.45`, MTP=2, contexte 262144). Flash-Next utilise les options FP4/B12X, MTP=4, 262144 tokens et les variables d'environnement NVIDIA. Le modèle plus gros doit être testé spécifiquement avec le harnais Qwen Code et son parseur `qwen3_xml`.

Les caches Hugging Face, FlashInfer et B12X sont conservés sous `$HOME/.cache` sur le DGX Spark ; les caches vLLM et Triton réutilisent les **volumes Docker nommés de ta v3** (`qwen-on-demand_vllm-cache`, `qwen-on-demand_triton-cache`). Si `HF_TOKEN` est nécessaire, définis-le sur l'hôte ou dans un fichier `.env` non partagé. La recette communautaire Flash-Next nécessite parfois des réglages ou mises à jour spécifiques selon le build.

## Mesure des performances

Le dashboard conserve les mesures de la v3 : débit global en tok/s via `/metrics` (différence de compteur entre deux rafraîchissements), KV, requêtes actives, mémoire conteneur. Le bouton « Tester le modèle » permet une réponse SSE et calcule un **débit approximatif de décodage** après le premier token.

Pour ton seuil **20 tokens/s**, fais des tests en conditions réelles, notamment avec un contexte 256K ; aucun seuil de performance n'est supposé garanti dans cette version.

## Arrêt, dépannage et sécurité

```bash
./stop.sh
./start.sh

docker compose logs -f proxy
docker logs -f qwen-on-demand-vllm
docker inspect qwen-on-demand-vllm --format '{{.Config.Image}}'
```

`stop.sh` privilégie l'API de l'application et refuse de forcer l'arrêt si le modèle tourne et qu'il est impossible de vérifier l'inactivité. Pour un arrêt administrateur forcé, `docker stop qwen-on-demand-vllm` est possible, mais interrompt les appels en cours.

**Sécurité :** accès au socket Docker = pouvoir administratif sur la machine. Le port du proxy est lié à `127.0.0.1` uniquement. Pour y accéder à distance, utilise un tunnel SSH (`ssh -L 8000:127.0.0.1:8000 utilisateur@dgx-spark`). Le modèle Flash-Next utilise certains paramètres conteneur non confinés issus de la recette NVIDIA : ne l'utilise que sur une machine de confiance.

## Documentation amont utilisée

- NVIDIA, DGX Spark : https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark/index.html
- NVIDIA, checkpoint et paramètres Flash-Next : https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/blob/main/README.md

