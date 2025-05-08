import streamlit as st
import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import altair as alt

# Configuration de la page Streamlit
st.set_page_config(layout="wide", page_title="Simulateur d'Atelier avec RL Realiser par Ourti Abdelilah")

# Constantes
GRID_SIZE = 10
NUM_MACHINES = 6
NUM_OPERATORS = 3
TASK_TYPES = ['réparation', 'maintenance', 'configuration', 'nettoyage']
MACHINE_STATES = {
    'WORKING': 'en fonctionnement',
    'BROKEN': 'en panne',
    'MAINTENANCE': 'en maintenance'
}

# Classes pour la simulation
class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def distance_to(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)

class Machine:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.state = MACHINE_STATES['WORKING']
        self.time_since_last_maintenance = 0
        self.time_to_break = random.randint(30, 90)
        self.operator = None
    
    def update(self):
        if self.state == MACHINE_STATES['WORKING']:
            self.time_since_last_maintenance += 1
            self.time_to_break -= 1
            
            # Machine breaks down
            if self.time_to_break <= 0:
                self.state = MACHINE_STATES['BROKEN']
                return {'event': 'breakdown', 'machine_id': self.id}
            
            # Machine needs maintenance
            if self.time_since_last_maintenance >= 100:
                return {'event': 'maintenance_needed', 'machine_id': self.id}
        
        return None
    
    def to_dict(self):
        return {
            'id': self.id,
            'position_x': self.position.x,
            'position_y': self.position.y,
            'state': self.state,
            'time_since_maintenance': self.time_since_last_maintenance,
            'time_to_break': self.time_to_break if self.state == MACHINE_STATES['WORKING'] else 0,
            'operator': self.operator
        }

class Task:
    def __init__(self, id, machine_id, task_type, priority, created_at):
        self.id = id
        self.machine_id = machine_id
        self.type = task_type
        self.priority = priority
        self.duration = self._get_duration(task_type)
        self.created_at = created_at
        self.waiting_time = 0
    
    def _get_duration(self, task_type):
        if task_type == 'réparation':
            return 10
        elif task_type == 'maintenance':
            return 5
        else:
            return 3
    
    def update_waiting_time(self, current_time):
        self.waiting_time = current_time - self.created_at
        # Dynamic priority adjustment
        if self.type == 'réparation':
            self.priority += 0.1
        return self
    
    def to_dict(self):
        return {
            'id': self.id,
            'machine_id': self.machine_id,
            'type': self.type,
            'priority': self.priority,
            'duration': self.duration,
            'waiting_time': self.waiting_time
        }

class Operator:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.busy = False
        self.current_task = None
        self.path = []
        self.current_task_progress = 0
    
    def move_toward(self, target_pos):
        if self.position.x < target_pos.x:
            self.position.x += 1
        elif self.position.x > target_pos.x:
            self.position.x -= 1
        elif self.position.y < target_pos.y:
            self.position.y += 1
        elif self.position.y > target_pos.y:
            self.position.y -= 1
        
        return self.position
    
    def assign_task(self, task):
        self.busy = True
        self.current_task = task
        self.current_task_progress = 0
        return {'event': 'task_assigned', 'operator_id': self.id, 'task_id': task.id}
    
    def complete_task(self):
        task = self.current_task
        self.busy = False
        self.current_task = None
        self.current_task_progress = 0
        return {'event': 'task_completed', 'operator_id': self.id, 'task_type': task.type, 'machine_id': task.machine_id}
    
    def work_on_task(self):
        if self.busy and self.current_task:
            self.current_task_progress += 1
            if self.current_task_progress >= self.current_task.duration:
                return self.complete_task()
        return None
    
    def to_dict(self):
        return {
            'id': self.id,
            'position_x': self.position.x,
            'position_y': self.position.y,
            'busy': self.busy,
            'task': self.current_task.id if self.current_task else None,
            'progress': f"{int((self.current_task_progress / self.current_task.duration) * 100)}%" if self.current_task else "N/A"
        }

# Environnement RL pour SARSA
class FactoryEnvironment:
    def __init__(self):
        self.grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.machines = []
        self.operators = []
        self.tasks = []
        self.pending_tasks = []
        self.time = 0
        self.logs = []
        
        # Initialisation
        self.initialize()
    
    def initialize(self):
        self.grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.machines = []
        self.operators = []
        self.tasks = []
        self.pending_tasks = []
        self.time = 0
        self.logs = []
        
        # Créer les machines
        for i in range(NUM_MACHINES):
            pos = self._get_random_empty_position()
            machine = Machine(f"m{i}", pos)
            self.machines.append(machine)
            self.grid[pos.y][pos.x] = {'type': 'machine', 'id': machine.id}
        
        # Créer les opérateurs
        for i in range(NUM_OPERATORS):
            pos = self._get_random_empty_position()
            operator = Operator(f"o{i}", pos)
            self.operators.append(operator)
            self.grid[pos.y][pos.x] = {'type': 'operator', 'id': operator.id}
        
        self.add_log("Simulation initialisée")
    
    def _get_random_empty_position(self):
        while True:
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            if self.grid[y][x] is None:
                return Position(x, y)
    
    def add_task(self, machine_id, task_type, priority):
        task = Task(f"t{len(self.tasks) + 1}", machine_id, task_type, priority, self.time)
        self.tasks.append(task)
        self.pending_tasks.append(task)
        self.add_log(f"Nouvelle tâche: {task_type} pour machine {machine_id} (priorité: {priority:.1f})")
    
    def add_log(self, message):
        self.logs.insert(0, f"T{self.time}: {message}")
        if len(self.logs) > 20:
            self.logs.pop()
    
    def get_state(self):
        """
        État pour l'apprentissage par renforcement
        """
        # Nombre de machines dans chaque état
        machines_working = sum(1 for m in self.machines if m.state == MACHINE_STATES['WORKING'])
        machines_broken = sum(1 for m in self.machines if m.state == MACHINE_STATES['BROKEN'])
        machines_maintenance = sum(1 for m in self.machines if m.state == MACHINE_STATES['MAINTENANCE'])
        
        # Nombre de tâches de chaque type
        tasks_repair = sum(1 for t in self.pending_tasks if t.type == 'réparation')
        tasks_maintenance = sum(1 for t in self.pending_tasks if t.type == 'maintenance')
        tasks_other = len(self.pending_tasks) - tasks_repair - tasks_maintenance
        
        # Opérateurs disponibles
        available_operators = sum(1 for o in self.operators if not o.busy)
        
        # État = tuple pour pouvoir être utilisé comme clé de dictionnaire
        return (machines_working, machines_broken, machines_maintenance, 
                tasks_repair, tasks_maintenance, tasks_other, available_operators)
    
    def get_actions(self):
        """
        Actions possibles pour l'agent RL : différentes façons d'attribuer les tâches
        """
        if not self.pending_tasks or not any(not o.busy for o in self.operators):
            return [0]  # Action "ne rien faire"
        
        actions = []
        # Action 0: Ne rien faire
        actions.append(0)
        # Action 1: Attribuer selon la priorité standard
        actions.append(1)
        # Action 2: Priorité aux réparations
        actions.append(2)
        # Action 3: Priorité à la proximité
        actions.append(3)
        # Action 4: Priorité au temps d'attente
        actions.append(4)
        
        return actions
    
    def take_action(self, action):
        """
        Exécuter une action et retourner la récompense
        """
        reward = 0
        if action == 0 or not self.pending_tasks or not any(not o.busy for o in self.operators):
            # Ne rien faire
            return reward
        
        # Opérateurs disponibles
        free_operators = [op for op in self.operators if not op.busy]
        if not free_operators:
            return reward
        
        # Trier les tâches selon la stratégie choisie
        if action == 1:  # Priorité standard
            sorted_tasks = sorted(self.pending_tasks, key=lambda t: (-t.priority))
        elif action == 2:  # Priorité aux réparations
            sorted_tasks = sorted(self.pending_tasks, 
                                key=lambda t: (0 if t.type == 'réparation' else 1, -t.priority))
        elif action == 3:  # Priorité à la proximité
            # Pour chaque opérateur, trouver la tâche la plus proche
            assigned_tasks = []
            for op in free_operators:
                if not self.pending_tasks:
                    break
                # Calcul des distances
                distances = []
                for task in self.pending_tasks:
                    machine = next((m for m in self.machines if m.id == task.machine_id), None)
                    if machine:
                        dist = op.position.distance_to(machine.position)
                        distances.append((task, dist))
                
                if distances:
                    # Trier par distance puis par priorité
                    closest_task = min(distances, key=lambda x: (x[1], -x[0].priority))[0]
                    self._assign_task_to_operator(closest_task, op)
                    assigned_tasks.append(closest_task)
                    reward += 0.5  # Récompense pour optimisation de distance
            
            # Retirer les tâches déjà assignées
            for task in assigned_tasks:
                if task in self.pending_tasks:
                    self.pending_tasks.remove(task)
            
            return reward
        elif action == 4:  # Priorité au temps d'attente
            sorted_tasks = sorted(self.pending_tasks, 
                                key=lambda t: (-t.waiting_time, -t.priority))
        else:
            # Action inconnue
            return -1.0  # Pénalité
            
        # Si action 1, 2 ou 4, assigner les tâches dans l'ordre de tri
        if action in [1, 2, 4]:
            assigned_count = 0
            for op in free_operators:
                if not sorted_tasks:
                    break
                task = sorted_tasks.pop(0)
                self._assign_task_to_operator(task, op)
                self.pending_tasks.remove(task)
                assigned_count += 1
                
                # Récompenses selon le type de tâche
                if task.type == 'réparation':
                    reward += 1.0  # Haute récompense pour réparation
                elif task.type == 'maintenance':
                    reward += 0.7  # Récompense moyenne pour maintenance
                else:
                    reward += 0.3  # Récompense faible pour tâches secondaires
            
            # Bonus pour avoir assigné plusieurs tâches
            if assigned_count > 1:
                reward += 0.2 * assigned_count
        
        return reward
    
    def _assign_task_to_operator(self, task, operator):
        result = operator.assign_task(task)
        machine = next((m for m in self.machines if m.id == task.machine_id), None)
        if machine:
            if task.type in ['réparation', 'maintenance']:
                machine.state = MACHINE_STATES['MAINTENANCE']
                machine.operator = operator.id
        self.add_log(f"Opérateur {operator.id} assigné à {task.type} sur machine {task.machine_id}")
        return result
    
    def step(self):
        """
        Avance la simulation d'un pas de temps
        """
        self.time += 1
        events = []
        
        # Mettre à jour l'état des machines
        for machine in self.machines:
            event = machine.update()
            if event:
                if event['event'] == 'breakdown':
                    self.add_task(machine.id, 'réparation', 5.0)
                    self.add_log(f"Machine {machine.id} est tombée en panne")
                elif event['event'] == 'maintenance_needed':
                    self.add_task(machine.id, 'maintenance', 3.0)
        
        # Mettre à jour les tâches en attente
        for task in self.pending_tasks:
            task.update_waiting_time(self.time)
        
        # Mettre à jour les opérateurs
        updated_grid = [row[:] for row in self.grid]
        for op in self.operators:
            if op.busy and op.current_task:
                # Trouver la machine cible
                target_machine = next((m for m in self.machines if m.id == op.current_task.machine_id), None)
                
                if target_machine and op.position != target_machine.position:
                    # Déplacer vers la machine
                    old_pos = Position(op.position.x, op.position.y)
                    new_pos = op.move_toward(target_machine.position)
                    
                    # Mettre à jour la grille
                    updated_grid[old_pos.y][old_pos.x] = None
                    updated_grid[new_pos.y][new_pos.x] = {'type': 'operator', 'id': op.id}
                elif target_machine and op.position == target_machine.position:
                    # Travailler sur la tâche
                    event = op.work_on_task()
                    if event:  # Tâche terminée
                        if target_machine:
                            if op.current_task and op.current_task.type == 'réparation':
                                target_machine.state = MACHINE_STATES['WORKING']
                                target_machine.time_to_break = random.randint(30, 90)
                                self.add_log(f"Machine {target_machine.id} réparée par {op.id}")
                            elif op.current_task and op.current_task.type == 'maintenance':
                                target_machine.state = MACHINE_STATES['WORKING']
                                target_machine.time_since_last_maintenance = 0
                                self.add_log(f"Maintenance de {target_machine.id} terminée par {op.id}")
                            target_machine.operator = None
        
        self.grid = updated_grid
        
        # Génération aléatoire de nouvelles tâches
        if random.random() < 0.05:
            random_machine = random.choice(self.machines)
            task_type = random.choice(['configuration', 'nettoyage'])
            priority = 2.0 if task_type == 'configuration' else 1.0
            self.add_task(random_machine.id, task_type, priority)
        
        # Calcul des statistiques pour la récompense
        num_broken = sum(1 for m in self.machines if m.state == MACHINE_STATES['BROKEN'])
        avg_waiting_time = np.mean([t.waiting_time for t in self.pending_tasks]) if self.pending_tasks else 0
        
        # Récompense négative pour les machines en panne et les longs temps d'attente
        penalty = -0.5 * num_broken - 0.1 * avg_waiting_time
        
        return penalty
    
    def render(self):
        """
        Retourne les données pour le rendu dans Streamlit
        """
        # Données pour grille
        grid_data = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                cell = self.grid[y][x]
                if cell:
                    cell_type = cell['type']
                    cell_id = cell['id']
                    if cell_type == 'machine':
                        machine = next((m for m in self.machines if m.id == cell_id), None)
                        state = machine.state if machine else ""
                        grid_data.append({
                            'x': x, 'y': y, 'type': cell_type, 'id': cell_id, 'state': state
                        })
                    else:
                        grid_data.append({
                            'x': x, 'y': y, 'type': cell_type, 'id': cell_id, 'state': ""
                        })
        
        # Données des machines
        machines_data = [m.to_dict() for m in self.machines]
        
        # Données des opérateurs
        operators_data = [o.to_dict() for o in self.operators]
        
        # Données des tâches
        tasks_data = [t.to_dict() for t in self.pending_tasks]
        
        return {
            'grid': grid_data,
            'machines': machines_data,
            'operators': operators_data,
            'tasks': tasks_data,
            'logs': self.logs,
            'time': self.time
        }

# Algorithme SARSA
class SarsaAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha      # Taux d'apprentissage
        self.gamma = gamma      # Facteur d'actualisation
        self.epsilon = epsilon  # Paramètre d'exploration
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.last_state = None
        self.last_action = None
        self.episode_rewards = []
        self.cumulative_reward = 0
    
    def get_action(self, state, available_actions):
        """
        Sélectionne une action en utilisant la politique epsilon-greedy
        """
        if random.random() < self.epsilon:
            # Exploration: choisir une action aléatoire
            return random.choice(available_actions)
        else:
            # Exploitation: choisir l'action avec la plus grande valeur Q
            q_values = {a: self.q_table[state][a] for a in available_actions}
            max_q = max(q_values.values()) if q_values else 0
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, next_action):
        """
        Met à jour la table Q en utilisant l'algorithme SARSA
        """
        # Q(s,a) = Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        
        # Mettre à jour la récompense cumulative
        self.cumulative_reward += reward
    
    def start_episode(self):
        """
        Réinitialise l'état pour un nouvel épisode
        """
        self.last_state = None
        self.last_action = None
        self.cumulative_reward = 0
    
    def end_episode(self):
        """
        Finalise l'épisode et enregistre la récompense
        """
        self.episode_rewards.append(self.cumulative_reward)
    
    def get_rewards_history(self):
        """
        Retourne l'historique des récompenses
        """
        return self.episode_rewards
    
    def decrease_epsilon(self, factor=0.995):
        """
        Diminue progressivement l'exploration
        """
        self.epsilon *= factor
        return self.epsilon

# Interface Streamlit
def main():
    st.title("Simulateur d'Atelier avec Apprentissage par Renforcement realiser par Ourti Abdelilah")
    
    # Sidebar pour les contrôles
    with st.sidebar:
        st.header("Contrôles")
        if 'running' not in st.session_state:
            st.session_state.running = False
        
        if st.session_state.running:
            if st.button("Pause"):
                st.session_state.running = False
        else:
            if st.button("Démarrer"):
                st.session_state.running = True
        
        if st.button("Réinitialiser"):
            st.session_state.env = FactoryEnvironment()
            st.session_state.agent = SarsaAgent()
            st.session_state.running = False
            st.session_state.episode = 0
            st.session_state.step_count = 0
        
        speed = st.slider("Vitesse", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        
        st.header("Paramètres SARSA")
        alpha = st.slider("Taux d'apprentissage (alpha)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        gamma = st.slider("Facteur d'actualisation (gamma)", min_value=0.5, max_value=1.0, value=0.9, step=0.05)
        epsilon = st.slider("Exploration (epsilon)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        
        st.session_state.use_rl = st.checkbox("Utiliser l'apprentissage par renforcement", value=True)
    
    # Initialiser l'environnement et l'agent si nécessaire
    if 'env' not in st.session_state:
        st.session_state.env = FactoryEnvironment()
    if 'agent' not in st.session_state:
        st.session_state.agent = SarsaAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
    if 'episode' not in st.session_state:
        st.session_state.episode = 0
    if 'step_count' not in st.session_state:
        st.session_state.step_count = 0
    
    # Mise à jour des paramètres de l'agent
    st.session_state.agent.alpha = alpha
    st.session_state.agent.gamma = gamma
    st.session_state.agent.epsilon = epsilon
    
    # Mettre à jour les statistiques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Temps", st.session_state.env.time)
    with col2:
        st.metric("Épisode", st.session_state.episode)
    with col3:
        st.metric("Epsilon", f"{st.session_state.agent.epsilon:.3f}")
    with col4:
        machines_broken = sum(1 for m in st.session_state.env.machines if m.state == MACHINE_STATES['BROKEN'])
        st.metric("Machines en panne", machines_broken)
    
    # Grille et visualisation
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Grille
        render_data = st.session_state.env.render()
        st.subheader("Atelier")
        
        # Créer la grille avec Altair
        grid_df = pd.DataFrame(render_data['grid'])
        if not grid_df.empty:
            # Définir les couleurs selon le type et l'état
            def get_color(row):
                if row['type'] == 'machine':
                    if row['state'] == MACHINE_STATES['WORKING']:
                        return '#4CAF50'  # Vert
                    elif row['state'] == MACHINE_STATES['BROKEN']:
                        return '#F44336'  # Rouge
                    else:
                        return '#FFC107'  # Jaune
                else:
                    return '#2196F3'  # Bleu pour les opérateurs
            
            if 'type' in grid_df.columns:
                grid_df['color'] = grid_df.apply(get_color, axis=1)
                
                # Créer le graphique Altair
                chart = alt.Chart(grid_df).mark_rect().encode(
                    x=alt.X('x:O', axis=alt.Axis(title='', labels=False, ticks=False), scale=alt.Scale(domain=list(range(GRID_SIZE)))),
                    y=alt.Y('y:O', axis=alt.Axis(title='', labels=False, ticks=False), scale=alt.Scale(domain=list(range(GRID_SIZE)))),
                    color=alt.Color('color:N', scale=None),
                    tooltip=['id', 'type', 'state']
                ).properties(
                    width=500,
                    height=500
                )
                
                # Ajouter les identifiants
                text = alt.Chart(grid_df).mark_text(fontSize=10).encode(
                    x=alt.X('x:O'),
                    y=alt.Y('y:O'),
                    text='id:N',
                    color=alt.value('white')
                )
                
                # Combiner les graphiques
                st.altair_chart(chart + text, use_container_width=True)
        else:
            st.write("Grille vide")
    
    with col2:
        # Légende
        st.subheader("Légende")
        legend_data = pd.DataFrame([
            {"État": "En fonctionnement", "Couleur": "#4CAF50"},
            {"État": "En panne", "Couleur": "#F44336"},
            {"État": "En maintenance", "Couleur": "#FFC107"},
            {"État": "Opérateur", "Couleur": "#2196F3"}
        ])
        
        # Créer la légende avec Altair
        legend = alt.Chart(legend_data).mark_rect().encode(
            y=alt.Y('État:N', axis=alt.Axis(title=None)),
            color=alt.Color('Couleur:N', scale=None)
        ).properties(width=20, height=100)
        
        text = alt.Chart(legend_data).mark_text(align='left', dx=30).encode(
            y=alt.Y('État:N'),
            text='État:N'
        )
        
        st.altair_chart(legend + text, use_container_width=True)
        
        # Graphique des récompenses
        st.subheader("Récompenses par épisode")
        rewards = st.session_state.agent.get_rewards_history()
        if rewards:
            rewards_df = pd.DataFrame({
                'Épisode': range(1, len(rewards) + 1),
                'Récompense': rewards
            })
            
            chart = alt.Chart(rewards_df).mark_line(point=True).encode(
                x='Épisode:Q',
                y='Récompense:Q',
                tooltip=['Épisode', 'Récompense']
            ).properties(height=200)
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Pas encore de données de récompense")
    
    # Informations détaillées
    col1, col2 = st.columns(2)
    
    with col1:
        # Machines
        st.subheader("Machines")
        machines_df = pd.DataFrame(render_data['machines'])
        if not machines_df.empty:
            # Convertir l'état en couleur pour l'affichage
            def highlight_state(val):
                if val == MACHINE_STATES['WORKING']:
                    return 'background-color: #4CAF5066'
                elif val == MACHINE_STATES['BROKEN']:
                    return 'background-color: #F4433666'
                else:
                    return 'background-color: #FFC10766'
            
            # Afficher le tableau avec stylage
            st.dataframe(machines_df.style.applymap(highlight_state, subset=['state']))
        else:
            st.write("Pas de données sur les machines")
        
        # Tâches
        st.subheader("Tâches en attente")
        tasks_df = pd.DataFrame(render_data['tasks'])
        if not tasks_df.empty:
            # Trier par priorité
            tasks_df = tasks_df.sort_values('priority', ascending=False)
            
            # Stylage en fonction du type de tâche
            def highlight_task_type(val):
                if val == 'réparation':
                    return 'background-color: #F4433666'
                elif val == 'maintenance':
                    return 'background-color: #FFC10766'
                else:
                    return 'background-color: #2196F366'
            
            st.dataframe(tasks_df.style.applymap(highlight_task_type, subset=['type']))
        else:
            st.write("Pas de tâches en attente")
    
    with col2:
        # Opérateurs
        st.subheader("Opérateurs")
        operators_df = pd.DataFrame(render_data['operators'])
        if not operators_df.empty:
            st.dataframe(operators_df)
        else:
            st.write("Pas de données sur les opérateurs")
        
        # Journal
        st.subheader("Journal d'événements")
        st.write("\n".join(render_data['logs']))
    
    # Section SARSA
    st.subheader("Détails de l'algorithme SARSA")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.use_rl:
            state = st.session_state.env.get_state()
            actions = st.session_state.env.get_actions()
            
            # Afficher l'état actuel et les valeurs Q
            st.write(f"État actuel: {state}")
            st.write(f"Actions disponibles: {actions}")
            
            # Afficher les valeurs Q pour l'état actuel
            q_values = {a: st.session_state.agent.q_table[state][a] for a in actions}
            q_df = pd.DataFrame({
                'Action': list(q_values.keys()),
                'Valeur Q': list(q_values.values())
            })
            st.write("Valeurs Q pour l'état actuel:")
            st.dataframe(q_df)
    
    with col2:
        st.write("Signification des actions:")
        action_meanings = {
            0: "Ne rien faire",
            1: "Priorité standard basée sur la valeur de priorité",
            2: "Priorité aux réparations",
            3: "Priorité à la proximité (distance minimale)",
            4: "Priorité au temps d'attente"
        }
        
        meanings_df = pd.DataFrame({
            'Action': list(action_meanings.keys()),
            'Signification': list(action_meanings.values())
        })
        st.dataframe(meanings_df)
    
    # Boucle principale de simulation
    if st.session_state.running:
        # Pas de simulation
        step_reward = st.session_state.env.step()
        st.session_state.step_count += 1
        
        if st.session_state.use_rl:
            # Obtenir l'état et les actions disponibles
            state = st.session_state.env.get_state()
            actions = st.session_state.env.get_actions()
            
            # Si c'est le début d'un épisode ou le premier pas
            if st.session_state.agent.last_state is None:
                action = st.session_state.agent.get_action(state, actions)
                st.session_state.agent.last_state = state
                st.session_state.agent.last_action = action
            else:
                # Choisir la nouvelle action avec SARSA
                next_action = st.session_state.agent.get_action(state, actions)
                
                # Apprendre de la transition
                st.session_state.agent.learn(
                    st.session_state.agent.last_state,
                    st.session_state.agent.last_action,
                    step_reward,
                    state,
                    next_action
                )
                
                # Mettre à jour l'état et l'action
                st.session_state.agent.last_state = state
                st.session_state.agent.last_action = next_action
                action = next_action
            
            # Exécuter l'action choisie
            action_reward = st.session_state.env.take_action(action)
        else:
            # Sans RL, utiliser la priorité standard
            action_reward = st.session_state.env.take_action(1)
        
        # Vérifier si l'épisode doit se terminer
        if st.session_state.step_count >= 100:
            st.session_state.agent.end_episode()
            st.session_state.episode += 1
            st.session_state.step_count = 0
            st.session_state.agent.start_episode()
            st.session_state.agent.decrease_epsilon()
            st.session_state.env.initialize()
        
        # Pause entre les itérations pour le visuel
        time.sleep(0.5 / speed)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
     