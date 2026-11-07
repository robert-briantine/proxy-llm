# Évolution du Proxy-LLM : détection des boucles et récupération automatique

## 1. Objectif

Faire évoluer le projet Proxy-LLM existant pour ajouter un mécanisme intelligent de détection des boucles d'utilisation d'outils.

Le proxy est utilisé entre le harnais Qwen Code et un serveur vLLM sur un DGX Spark 128 Go.

Architecture actuelle :

Qwen Code → Proxy-LLM (FastAPI) → vLLM → Modèle actif

Deux modèles sont actuellement configurés :
- Qwen3.8-27B NVFP4
- Qwen3.8-Flash-Next FP4

Le proxy sait déjà changer dynamiquement de modèle lorsque toutes les requêtes en cours sont terminées.

**Cette fonctionnalité doit impérativement être conservée.**

L'objectif de cette évolution est de déplacer la détection des boucles et la gestion des stratégies de récupération dans le proxy, sans modifier Qwen Code.

Le proxy doit rester totalement transparent pour le harnais.

## 2. Analyser le projet existant

Avant toute modification :

- Examiner `proxy/app.py`.
- Examiner `proxy/models.json`.
- Examiner `proxy/dashboard.html`.
- Examiner les tests existants.
- Identifier le fonctionnement du proxy OpenAI et des réponses streaming.
- Identifier les mécanismes de gestion des requêtes et de changement dynamique des modèles.

Conserver les fonctionnalités existantes.

Privilégier une architecture modulaire : ne pas concentrer toute la nouvelle logique dans `app.py`.

## 3. Journalisation des appels d'outils

Le proxy doit observer les appels d'outils produits par les modèles.

Enregistrer notamment :

- L'identifiant de session lorsqu'il est disponible.
- Le modèle utilisé.
- Le nom de l'outil appelé.
- Les arguments transmis.
- Le résultat de l'exécution lorsqu'il est disponible.
- La date et l'heure.
- Le nombre d'appels d'outils.
- Les éventuelles erreurs.

Les informations doivent pouvoir être enregistrées dans une base SQLite.

Attention : les outils sont exécutés par Qwen Code, pas par le proxy.

Le résultat d'un appel d'outil est généralement transmis dans la requête suivante, avec l'historique de conversation.

Le proxy doit donc être capable de reconstituer la séquence des appels et de leurs résultats.

Comme Qwen Code retransmet l'historique à chaque requête, il faut éviter de comptabiliser plusieurs fois les mêmes appels.

Prévoir également le cas où l'historique est compressé ou tronqué.

La journalisation ne doit pas mélanger les conversations de clients distincts. Si aucun identifiant fiable de session n'est disponible, prévoir un mécanisme explicite d'identification et documenter les limites de l'identification automatique.

## 4. Détection des boucles

Développer un module indépendant, par exemple :

`proxy/loop_detector.py`

Ce module doit détecter plusieurs types de répétitions.

### Cas A : répétition simple

Exemple :

```text
Tool A
Tool A
Tool A
```

Si les appels et leurs résultats sont identiques, considérer qu'une boucle est probable.

### Cas B : répétition cyclique

Exemple :

```text
Tool A
Tool B
Tool A
Tool B
Tool A
Tool B
```

Détecter les cycles de plusieurs appels d'outils.

Prévoir initialement des cycles de longueur 1 à 4, avec un seuil de répétition configurable.

### Cas C : répétition d'échec

Exemple :

```text
npm run build → erreur TS2304
npm run build → erreur TS2304
npm run build → erreur TS2304
```

Détecter la répétition d'une même erreur lorsque les appels ne montrent aucune progression.

### Précautions

Ne pas considérer automatiquement chaque répétition comme une boucle.

Par exemple, deux compilations identiques peuvent être parfaitement légitimes si le code a été modifié entre les deux.

Utiliser les informations disponibles dans l'historique pour limiter les faux positifs.

Normaliser les arguments JSON avant comparaison.

Ne pas masquer systématiquement les différences entre résultats : certaines différences constituent une preuve de progression.

L'algorithme doit être déterministe, configurable et testable.

Ne pas utiliser un LLM pour détecter les boucles dans cette première version.

## 5. Stratégies de récupération

Lorsque le proxy détecte une boucle, il doit pouvoir intervenir automatiquement.

Prévoir quatre modes :

**Mode 0 : Observation**

Le proxy détecte et journalise les boucles, mais ne modifie pas les requêtes.

**Mode 1 : Prompt correctif**

Le proxy ajoute une instruction temporaire au contexte envoyé au modèle.

Exemple de prompt configurable :

« Une boucle d'utilisation des outils a été détectée. Tu sembles répéter les mêmes opérations sans progresser. Analyse les dernières tentatives et leurs résultats. Ne reproduis pas les actions ayant échoué sans apporter de modification pertinente. Essaie une stratégie différente pour atteindre l'objectif demandé. »

Le prompt doit pouvoir inclure des informations générées par le détecteur :

- Type de boucle.
- Nombre de répétitions.
- Outils concernés.
- Derniers résultats ou erreurs.

**Mode 2 : Thinking**

Le proxy active temporairement le mode thinking pour le modèle Qwen compatible.

Utiliser si le modèle le prend en charge :

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

L'activation doit être effectuée par requête, sans redémarrer vLLM.

Vérifier la compatibilité avec les deux modèles et conserver les autres paramètres de génération.

Si un modèle ne prend pas en charge cette option, ne pas provoquer d'erreur : signaler cette limitation.

**Mode 3 : Prompt + Thinking**

Combiner les deux stratégies précédentes.

### Comportement important

Le prompt correctif doit être ajouté uniquement à la requête transmise au modèle.

Il ne doit pas modifier durablement le contexte de Qwen Code.

Conserver la structure OpenAI des messages ainsi que la cohérence des appels d'outils.

La récupération ne doit jamais exécuter un outil directement ni réexécuter automatiquement une action ayant des effets de bord.

Pour cette version, appliquer l'intervention à la prochaine requête concernée, sans nécessiter de mettre en mémoire tampon l'intégralité du streaming de la réponse en cours.

## 6. Configuration

Ajouter une configuration persistante, par exemple :

```json
{
  "loopDetection": {
    "enabled": true,
    "repetitionThreshold": 3,
    "maxCycleLength": 4,
    "storeToolHistory": true
  },
  "recovery": {
    "mode": "prompt_and_thinking",
    "prompt": "Une boucle a été détectée. Analyse les derniers échecs et change de stratégie.",
    "temporaryThinking": true
  }
}
```

Les modes autorisés sont :

- `observe`
- `prompt`
- `thinking`
- `prompt_and_thinking`

Le texte correctif doit être modifiable librement.

Les paramètres doivent être sauvegardés et restaurés au redémarrage du proxy.

Ne pas supprimer ou casser les configurations des modèles déjà présentes dans `models.json`.

Prévoir une API REST permettant de consulter et modifier les paramètres.

## 7. Interface Web

Faire évoluer `proxy/dashboard.html`.

Ajouter une section « Détection des boucles ».

Elle doit afficher :

- Activation/désactivation de la détection.
- Nombre de répétitions avant détection.
- Longueur maximale des cycles.
- Mode de récupération.
- Zone de texte pour configurer le prompt correctif.
- Bouton de sauvegarde.

Ajouter une section « Traces d'exécution ».

Afficher :

- Le nombre total d'appels d'outils.
- Le nombre de boucles détectées.
- Les dernières boucles.
- Le modèle concerné.
- Les outils impliqués.
- Le type de boucle.
- L'intervention effectuée.
- La date de détection.

Permettre de consulter les détails d'une boucle détectée.

Conserver l'interface actuelle de sélection des modèles, les informations de performances et les commandes de démarrage/arrêt.

L'interface doit continuer à fonctionner sur le port actuel du proxy.

## 8. État des interventions

Maintenir un état simple par session :

```text
NORMAL
   |
   | Boucle détectée
   v
RECOVERY
   |
   | Prompt et/ou thinking
   v
NORMAL
```

Le proxy doit conserver la signature de la boucle détectée.

Éviter d'injecter indéfiniment le même prompt si le problème persiste.

Prévoir une limitation configurable des tentatives de récupération.

Si la limite est atteinte, enregistrer l'échec et revenir à un comportement clairement défini.

Ne pas construire une machine à états inutilement complexe.

## 9. Contraintes impératives

- Ne pas modifier Qwen Code.
- Ne pas modifier les outils du harnais.
- Ne pas inventer de nouveaux outils.
- Ne pas modifier les appels d'outils valides transmis au harnais.
- Conserver l'API OpenAI compatible.
- Conserver les réponses streaming.
- Conserver la gestion dynamique des deux modèles.
- Conserver la protection des requêtes en cours lors d'un changement de modèle.
- Ne pas introduire de changement automatique de modèle dans cette version.
- Ne pas ajouter d'appel à Codex ou à un service externe.
- Ne pas bloquer les requêtes normales à cause de la journalisation.
- Limiter la consommation de mémoire de l'historique des outils.
- Prévoir une politique de conservation des traces et éviter de journaliser les secrets inutilement.

Le comptage des outils doit être réalisé dans le proxy, indépendamment du compteur éventuellement présent dans Qwen Code.

Le proxy ne doit pas imposer une limite bloquante d'appels d'outils dans cette première version.

## 10. Tests automatisés

Ajouter des tests couvrant au minimum :

1. Détection de trois appels identiques.
2. Détection d'un cycle A-B-A-B.
3. Détection d'un cycle de trois ou quatre outils.
4. Absence de faux positif lorsque les résultats montrent une progression.
5. Absence de double comptage de l'historique.
6. Injection correcte du prompt.
7. Activation temporaire du thinking.
8. Combinaison prompt + thinking.
9. Désactivation de la détection.
10. Persistance des paramètres.
11. Isolation entre sessions.
12. Conservation du streaming et du format des appels d'outils.
13. Compatibilité avec le changement dynamique des modèles existant.

Utiliser des réponses OpenAI simulées pour tester les mécanismes sans nécessiter le chargement réel des modèles.

Exécuter également tous les tests déjà présents dans le projet.

## 11. Organisation du code

Proposer une séparation claire des responsabilités.

Par exemple :

```text
proxy/
  app.py
  loop_detector.py
  tool_tracker.py
  recovery.py
  settings.py
  dashboard.html
  models.json
```

Cette organisation est indicative : adapter les fichiers à l'architecture réelle du projet.

Éviter les dépendances inutiles.

## 12. Livraison attendue

Modifier directement le projet existant.

Ne pas recréer le proxy depuis zéro.

À la fin :

- Présenter les fichiers modifiés.
- Expliquer succinctement l'algorithme de détection.
- Indiquer comment configurer les quatre modes.
- Indiquer les nouvelles API.
- Fournir les résultats des tests.
- Expliquer comment redémarrer le proxy.
- Signaler les limitations éventuelles.

**Ne pas se limiter à proposer une architecture : réaliser l'implémentation, écrire les tests et les exécuter.**

### Objectif final

Le Proxy-LLM doit devenir capable d'observer les appels d'outils, de détecter des boucles et d'essayer de corriger automatiquement le comportement du modèle en enrichissant temporairement son contexte ou en activant le thinking.

Le tout doit rester transparent pour Qwen Code.

La sélection automatique d'un modèle de secours sera développée dans une évolution ultérieure.