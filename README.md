# Simulateur d'Atelier avec Apprentissage par Renforcement

## Présentation

Ce projet est un simulateur interactif d'atelier industriel qui utilise l'apprentissage par renforcement (RL) pour optimiser l'attribution des tâches aux opérateurs. L'application est développée avec Streamlit et permet de visualiser en temps réel la simulation d'un environnement de production avec des machines et des opérateurs.

![Capture d'écran du simulateur](https://via.placeholder.com/800x400)

## Fonctionnalités

- **Simulation d'atelier en temps réel** : Visualisation d'un environnement de production avec machines et opérateurs
- **Gestion automatique des pannes** : Les machines peuvent tomber en panne aléatoirement et nécessiter des réparations
- **Système de tâches prioritaires** : Différents types de tâches (réparation, maintenance, configuration, nettoyage) avec des niveaux de priorité distincts
- **Apprentissage par renforcement SARSA** : Optimisation de l'attribution des tâches aux opérateurs
- **Interface utilisateur interactive** : Contrôle de la simulation et visualisation des données en temps réel
- **Métriques de performance** : Suivi des récompenses d'apprentissage et des statistiques de production

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/votre-utilisateur/simulateur-atelier-rl.git
cd simulateur-atelier-rl
```

2. Créez et activez un environnement virtuel (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Lancez l'application Streamlit :
```bash
streamlit run factory_app.py
```

2. Accédez à l'application dans votre navigateur (généralement à l'adresse http://localhost:8501)

3. Utilisez les contrôles dans la barre latérale pour :
   - Démarrer/Pause la simulation
   - Réinitialiser la simulation
   - Ajuster la vitesse de simulation
   - Configurer les paramètres de l'algorithme SARSA
   - Activer/désactiver l'apprentissage par renforcement

## Structure du code

Le simulateur est organisé autour de plusieurs classes principales :

- `Position` : Représente les coordonnées dans la grille
- `Machine` : Gère l'état des machines (fonctionnement, panne, maintenance)
- `Task` : Représente les différentes tâches à effectuer
- `Operator` : Gère les opérateurs qui se déplacent et effectuent les tâches
- `FactoryEnvironment` : Environnement principal qui coordonne la simulation
- `SarsaAgent` : Implémentation de l'algorithme d'apprentissage par renforcement SARSA

## Algorithme SARSA

Le simulateur utilise l'algorithme SARSA (State-Action-Reward-State-Action) pour optimiser l'attribution des tâches. Cet algorithme d'apprentissage par renforcement permet à l'agent d'apprendre une politique optimale par exploration et exploitation.

Paramètres ajustables :
- `alpha` : Taux d'apprentissage
- `gamma` : Facteur d'actualisation
- `epsilon` : Paramètre d'exploration

## Stratégies d'attribution des tâches

L'agent peut choisir parmi plusieurs stratégies d'attribution :
1. Ne rien faire
2. Priorité standard (basée sur la valeur de priorité des tâches)
3. Priorité aux réparations
4. Priorité à la proximité (distance minimale entre opérateur et machine)
5. Priorité au temps d'attente des tâches

## Métriques et visualisation

L'interface affiche plusieurs visualisations et métriques :
- Grille de l'atelier avec machines et opérateurs
- État des machines (fonctionnement, panne, maintenance)
- Liste des tâches en attente
- État des opérateurs
- Journal d'événements
- Graphique d'évolution des récompenses
- Valeurs Q pour l'état actuel

## Dépendances

- Python 3.7+
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Altair

## Personnalisation

Vous pouvez modifier les constantes au début du code pour personnaliser la simulation :
- `GRID_SIZE` : Taille de la grille de l'atelier
- `NUM_MACHINES` : Nombre de machines
- `NUM_OPERATORS` : Nombre d'opérateurs
- `TASK_TYPES` : Types de tâches disponibles

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Auteur

[Votre nom] - [votre.email@exemple.com]

---

Développé avec ❤️ et Python