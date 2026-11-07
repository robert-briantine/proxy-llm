# Smart vLLM On Demand — Qwen3.8 27B ↔ Flash-Next — DGX Spark 128 Go

Version 4 du proxy fourni. **Même adresse OpenAI pour Qwen Code**, avec changement de modèle à la demande uniquement lorsque **toutes les requêtes déjà admises sont terminées**, y compris les réponses SSE en streaming.

- Interface : `http://127.0.0.1:8000/`
- API OpenAI : `http://127.0.0.1:8000/v1`
- Nom public inchangé : `qwen3.8:27b-mtp` (quelle que soit la version du modèle chargée)
- Conteneurs Docker : **un conteneur proxy permanent**, **un seul conteneur d'inférence vLLM**, remplacé lors d'un changement de modèle. Deux conteneurs au total en fonctionnement normal (pas deux conteneurs d'inférence).
- Sélection mémorisée dans `data/selection.json` ; caches de modèles préservés.
- Détection déterministe des boucles d’outils, traces SQLite et récupération temporaire par prompt et/ou thinking ; mode observation par défaut.

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

## Application de bureau sur le DGX Spark

Depuis le dossier du projet, installe le lanceur pour ton utilisateur (sans `sudo`) :

```bash
bash install-desktop.sh
```

Ouvre ensuite **Qwen On Demand** dans le menu des applications Ubuntu ou avec le raccourci ajouté au bureau s'il est activé. Si GNOME le demande, fais un clic droit sur le raccourci puis **Autoriser le lancement**. Tu peux aussi l'épingler aux favoris depuis le menu des applications.

Le lanceur réutilise le proxy s'il répond déjà, sinon il exécute `start.sh`, puis attend que l'interface soit disponible. Avec Chrome ou Chromium, le dashboard s'ouvre dans une fenêtre d'application dédiée, avec un profil navigateur séparé ; sinon, il s'ouvre dans le navigateur par défaut. Docker, Compose v2, `curl`, `flock` (util-linux) et un navigateur doivent être installés ; Python 3 sert à l'installation. `zenity` permet d'afficher les erreurs et `notify-send` les notifications s'ils sont disponibles.

Le modèle reste chargé **à la demande** depuis le dashboard ou l'API. Fermer la fenêtre ne coupe ni le proxy ni les requêtes en cours. Pour arrêter les services, utilise `./stop.sh` après tes sessions.

Le lanceur référence le dossier actuel du projet : conserve ce dossier à cet emplacement, ou relance l'installation après l'avoir déplacé. Les journaux se trouvent dans `~/.local/state/qwen-on-demand/launcher.log` et `browser.log` (ou sous `$XDG_STATE_HOME` si défini). Le précédent journal de lancement est conservé dans `launcher.previous.log`.

Pour supprimer les raccourcis, sans arrêter les services ni supprimer les modèles :

```bash
bash install-desktop.sh --uninstall
```

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

Le premier téléchargement Flash-Next entre dans le délai de démarrage. Son profil définit `startup_timeout_seconds=14400` (4 heures), alors que le 27B conserve le délai global `STARTUP_TIMEOUT_SECONDS` (20 minutes par défaut). Le dashboard affiche le délai applicable pendant le changement. Modifier ce délai ne recrée pas le moteur. Après redémarrage du proxy, sélectionner à nouveau Flash-Next reprend l’attente du conteneur Flash-Next existant sans interrompre son téléchargement. En cas d’échec, les 80 dernières lignes des journaux du moteur (32 Kio maximum) sont copiées dans les journaux du proxy avant la restauration du modèle précédent. Le délai d’arrêt automatique commence après que le modèle est prêt.

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


## Détection des boucles et récupération

La section **Détection des boucles** du dashboard permet de sauvegarder la configuration sans redémarrage. Les paramètres sont persistés dans `data/loop-settings.json`, séparément de `models.json` et de la sélection du modèle. Une détection active ne fixe aucune limite bloquante au nombre d’outils et ne change jamais automatiquement de modèle.

### Identification des conversations

Le mode **client unique** (`loopDetection.singleClient=true`, valeur par défaut) suit automatiquement les requêtes sans identifiant dans une session commune. Il convient à cette installation avec un seul client LLM : aucun en-tête supplémentaire n’est nécessaire pour compter les outils, détecter les boucles et appliquer la récupération.

Un identifiant explicite reste prioritaire pour séparer certaines conversations :

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Proxy-Session-ID: projet-a-conversation-1' \
  -d '{"model":"qwen3.8:27b-mtp","messages":[{"role":"user","content":"Bonjour"}]}'
```

`metadata.session_id` est aussi reconnu. Les identifiants explicites sont hachés avec l’en-tête Authorization pour isoler les conversations utilisant des identifiants d’accès différents. Sans identifiant, le mode client unique utilise une identité interne commune, indépendante de la clé API. L’identité de session et les caches de détection expirent toujours selon les paramètres de mémoire ci-dessous.

Avec `singleClient=false`, un identifiant stable devient nécessaire pour la continuité entre requêtes. Dans ce mode, sans identifiant, seuls les nouveaux outils de la réponse sont comptés ; l’historique entrant n’est pas recompté et aucune récupération n’est appliquée. Le proxy renvoie alors un `X-Proxy-Session-ID` aléatoire réutilisable par le client.

### Algorithme et interventions

Le proxy reconstitue les appels `assistant.tool_calls` et leurs réponses `role: tool` avec `tool_call_id`. Il observe aussi les outils produits dans les réponses JSON et SSE. Les identifiants stables évitent de compter une seconde fois les appels présents dans l’historique. Lorsqu’un historique long est rejoué, le dernier appel connu sert d’ancre pour ignorer son préfixe. Une troncature avec chevauchement conserve donc la continuité ; une modification des messages utilisateur ouvre une nouvelle fenêtre de détection.

Le détecteur cherche dans les **80 derniers appels** un suffixe périodique de longueur 1 à 4, répété au moins 3 fois par défaut. Les arguments JSON sont normalisés (ordre des clés et espaces sans effet). Les empreintes des résultats complets sont comparées, pas leurs extraits affichés. Un résultat différent, un autre outil entre deux compilations ou un résultat manquant empêchent de considérer ces opérations comme une répétition simple. Les mêmes échecs sont classés `repeated_failure`, les autres boucles `simple` ou `cycle`. Aucun LLM ne participe à cette analyse.

| Mode `recovery.mode` | Comportement |
|---|---|
| `observe` | Détecter et journaliser seulement (valeur par défaut). |
| `prompt` | Ajouter le prompt configurable et le diagnostic à la seule requête sortante. |
| `thinking` | Ajouter `chat_template_kwargs.enable_thinking=true` à cette requête. |
| `prompt_and_thinking` | Combiner les deux interventions. |

`recovery.temporaryThinking=false` désactive la partie thinking, même dans les deux derniers modes. Les autres paramètres de génération et les autres clés de `chat_template_kwargs` sont conservés. L’intervention survient à la requête suivante contenant les résultats nécessaires à la détection. Les messages d’origine, les outils et les identifiants transmis au harnais restent cohérents ; aucun outil n’est exécuté ou relancé par le proxy.

Les deux familles documentent le thinking par requête : [Qwen3.8-27B / quantification utilisée](https://huggingface.co/PassingByPixels/Qwen3.8-27B-NVFP4#upstream-model-description), [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage), [vLLM](https://docs.vllm.ai/en/latest/features/reasoning_outputs/). `supports_thinking` dans chaque profil déclare cette capacité ; mettre `false` pour un profil incompatible. Dans ce cas, l’option est ignorée et la limitation figure dans la trace (la partie prompt reste applicable). Flash-Next active normalement déjà le thinking : l’intervention peut donc être sans effet supplémentaire. La compatibilité effective des images installées doit être vérifiée sur le DGX ; il n’y a ni redémarrage vLLM, ni requête de sondage, ni tentative automatique de renvoyer une requête rejetée.

L’état d’une session passe de `NORMAL` à `RECOVERY` pendant la préparation puis revient à `NORMAL`. Une signature indépendante de la rotation du cycle partage un budget de **2 tentatives** par défaut. Un historique inchangé ne déclenche pas une nouvelle intervention. Lorsque le budget est épuisé, la trace indique `limit_reached` et le proxy continue en observation, sans bloquer le harnais. Sauvegarder les paramètres ne réinitialise pas ce budget.

### API de configuration et traces

| Méthode et URL | Réponse / action |
|---|---|
| `GET /api/loop-settings` | Configuration complète actuelle. |
| `PUT /api/loop-settings` | Remplacement validé et sauvegarde atomique ; HTTP 422 si invalide, 503 si écriture impossible. Les champs omis reprennent leurs valeurs par défaut. |
| `GET /api/traces?limit=30&kind=all` | Compteurs cumulés `tool_calls`, `loops`, `requests`, résumés des dernières traces, occupation disque (`disk_bytes`, `max_disk_bytes`) et état `trace_enabled`. Filtre `kind` : `all`, `requests`, `tools`, `loops`. Limite entre 1 et 100. |
| `GET /api/traces/{id}` | Détail d’une requête, d’un outil ou d’une boucle ; HTTP 404 si inconnu ou expiré. |

Exemple : activer prompt + thinking avec les autres valeurs par défaut :

```bash
curl -X PUT http://127.0.0.1:8000/api/loop-settings \
  -H 'Content-Type: application/json' \
  -d '{"loopDetection":{"enabled":true,"repetitionThreshold":3,"maxCycleLength":4,"storeToolHistory":true},"recovery":{"mode":"prompt_and_thinking","prompt":"Analyse les derniers échecs et change de stratégie.","temporaryThinking":true,"maxAttempts":2}}'
```

### Conservation, confidentialité et limites

- **Arrêt manuel immédiat** : décocher **Activer les traces** dans le dashboard sauvegarde immédiatement `loopDetection.traceEnabled=false`, sans devoir cliquer sur Sauvegarder. Aucun nouvel événement d’outil, de boucle ou de requête LLM n’est enregistré ; les captures LLM en cours et les événements encore en file sont écartés. La détection et la récupération continuent. Les traces déjà écrites restent consultables et soumises à la purge. Recocher reprend l’enregistrement pour les nouveaux appels. Ce réglage persiste après redémarrage.
- **Tous les appels LLM** : activer **Tracer toutes les requêtes et réponses LLM** (`traceAllRequests=true`, désactivé par défaut). Toutes les routes `/v1/*` sont tracées, y compris Responses, Embeddings, erreurs HTTP et flux SSE. Chaque trace contient route, méthode, modèle actif, statut, date, durée, volumes et indicateur de fin du flux. Les traces d’outils restent disponibles séparément via `storeToolHistory`. Ces options fonctionnent même avec la détection des boucles désactivée.
- SQLite : `data/tool-history.sqlite3`, modifiable via `TOOL_HISTORY_PATH`. Écriture dans un thread dédié, file de 2048 éléments maximum. Une panne ou saturation abandonne des traces et incrémente les compteurs d’erreur sans bloquer les requêtes. Les compteurs sont donc mis à jour avec un léger délai et deviennent incomplets en cas de perte.
- `storeToolHistory=true` conserve noms, modèle, identifiants hachés, dates, erreurs et empreintes des arguments/résultats. Avec `false`, seuls compteurs et événements de boucle sont persistés.
- `traceContent=false` par défaut : aucun contenu des requêtes, réponses, arguments ou résultats n’est enregistré. Avec `true`, le contenu est expurgé puis enregistré jusqu’à `maxTraceContentKB=1024` par requête/réponse (en Kio d’entrée ; pour les arguments/résultats d’outils, en milliers de caractères). Les champs `*_truncated` signalent une troncature. Les en-têtes HTTP ne sont pas enregistrés. L’expurgation des clés sensibles et de certains secrets textuels reste partielle, notamment avec du texte libre ou tronqué. Les comparaisons utilisent toujours les empreintes complètes ; le prompt correctif utilise des extraits de 512 caractères.
- **Quota disque** : `maxDiskMB=256` Mio par défaut, réglable de 1 à 102400 Mio. Le writer limite les pages SQLite, réserve environ la moitié du quota pour le journal transactionnel et supprime les événements les plus anciens, toutes catégories confondues, pour faire place aux nouveaux. `auto_vacuum=FULL` restitue réellement les pages libres au disque. Une réduction du quota déclenche la purge dans le writer, même sans nouvelle inférence. L’occupation affichée inclut la base et ses journaux. Une trace individuelle trop volumineuse est abandonnée et comptée, sans bloquer le LLM. Le quota concerne les traces du proxy ; les poids des modèles et les journaux Docker ont leur propre stockage. La migration d’une ancienne base et la réduction du quota peuvent nécessiter temporairement de l’espace de compactage.
- `retentionDays=7`, `maxTraceRows=10000` par table. Purge après chaque écriture et toutes les 30 secondes d’inactivité du writer ; les compteurs cumulés restent conservés. Au redémarrage, les paramètres et la purge sont appliqués dès le démarrage du proxy. La purge n’est pas un effacement sécurisé du support.
- Mémoire bornée : 256 sessions par défaut (`maxSessions`), expiration après 3600 secondes (`sessionTTLSeconds`), 256 appels connus, 80 appels analysés et 64 signatures de récupération par session. L’état de récupération n’est pas conservé au redémarrage. La déduplication SQLite survit aux redémarrages tant que les lignes correspondantes sont conservées et `storeToolHistory=true` ; les événements de détection peuvent être détectés à nouveau après redémarrage. Après expiration/éviction, sans ancre commune ou si les identifiants changent pendant la compression, la continuité et la déduplication ne sont pas garanties. Fournir des identifiants d’outils uniques et stables.
- Le périmètre de **détection des boucles d’outils** est Chat Completions, outils OpenAI `tool_calls`, choix de réponse d’index 0. Les autres routes OpenAI sont relayées et peuvent être tracées via `traceAllRequests`, mais leurs outils ne sont pas analysés : cela inclut les anciens `function_call`, les blocs propriétaires de Qwen et l’API Responses. Avec plusieurs choix, un seul est suivi pour ne pas confondre des branches alternatives.
- Le streaming est relayé octet pour octet. Seuls les fragments d’outils sont assemblés (événement SSE de 64 Kio, 128 outils et 256 Kcaractères de fragments maximum ; JSON de réponse limité à 1 Mio pour l’observation). Au-delà, l’observation de cette réponse est abandonnée et le flux continue ; les appels peuvent être reconstruits depuis l’historique de la prochaine requête identifiée.
- Une boucle observée reste une probabilité : un sondage externe volontaire peut donner un résultat identique plusieurs fois ; à l’inverse, des horodatages changeants dans les résultats empêchent une détection. La détection d’erreur textuelle est indicative ; elle ne remplace pas un protocole d’erreur structuré.

### Tests et redémarrage

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q

# Après installation de cette évolution : reconstruire uniquement le proxy.
# Attendre d’abord la fin des appels en cours : ce redémarrage du proxy coupe ses flux.
docker compose up -d --build proxy
```

Les tests simulent Docker, les réponses OpenAI JSON/SSE, les résultats d’outils et SQLite. Ils vérifient les quatre modes, les cycles de longueur 1 à 4, la progression, les doublons, l’isolation, les budgets, la persistance et la conservation des flux et du changement dynamique existant. Les paramètres sauvegardés via le dashboard/API s’appliquent immédiatement et ne nécessitent pas de redémarrage.

Modules ajoutés : `settings.py` (validation/persistance), `tool_tracker.py` (sessions et corrélation), `loop_detector.py` (cycles), `recovery.py` (intervention temporaire), `request_trace.py` (capture HTTP optionnelle) et `trace_store.py` (SQLite et quota disque). `app.py` conserve la gestion Docker, l’admission et le drainage des requêtes.
