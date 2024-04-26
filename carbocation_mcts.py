import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs 


class CarbocationState:
    def __init__(self, smiles, other_attributes):
        self.smiles = smiles
        self.other_attributes = other_attributes

    def __repr__(self):
        return f"CarbocationState(smiles={self.smiles}, other_attributes={self.other_attributes})"

     def to_molecule(self):
        return Chem.MolFromSmiles(self.smiles)

 class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.visits = 0
        self.reward = 0
        self.children = []

 
    def is_fully_expanded(self, charged_positions):
        return len(self.children) == len(self.get_possible_actions(charged_positions))

    def get_possible_actions(self, charged_positions):
        # Use charged positions as possible actions for expansion
        return charged_positions

     def expand(self, charged_positions, similarity_to_goal):
        if not self.is_fully_expanded(charged_positions):
            possible_actions = self.get_possible_actions(charged_positions)
            action = random.choice(possible_actions)
            new_state = CarbocationState(smiles=action, other_attributes={})
            new_similarity_to_goal = self.calculate_similarity(new_state, self.state)  # Similarity to parent
            if new_similarity_to_goal is not None and new_similarity_to_goal >= similarity_to_goal:
                new_child = Node(new_state, parent=self, action=action)
                self.children.append(new_child)
                return new_child, new_similarity_to_goal
        return None, None


    def select_child(self):
        return random.choice(self.children)


    def calculate_morgan_fingerprint(self, mol, radius=2, nbits=2048):
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        return fp

 
    def calculate_similarity(self, state1, state2):
        mol1 = state1.to_molecule()
        mol2 = state2.to_molecule()
        fp1 = self.calculate_morgan_fingerprint(mol1)
        fp2 = self.calculate_morgan_fingerprint(mol2)
        if fp1 is None or fp2 is None:
            return None
        # Using Braun-Blanquet coefficient for similarity
        similarity = DataStructs.BraunBlanquetSimilarity(fp1, fp2)
        return similarity

 
class CarbocationMCTSExample:
    def __init__(self, start_smiles, goal_smiles, smiles_file, charged_positions_file, max_iterations, max_steps, similarity_threshold):
        self.start_state = self.parse_smiles(start_smiles)
        self.goal_state = self.parse_smiles(goal_smiles)
        self.charged_positions = self.parse_charged_positions(charged_positions_file)
        self.root = Node(self.start_state)
        self.max_iterations = max_iterations
        self.max_steps = max_steps
        self.steps_taken = 0
        self.similarity_threshold = similarity_threshold


    def parse_charged_positions(self, charged_positions_file):
        with open(charged_positions_file, 'r') as f:
            lines = f.readlines()

        charged_positions = []
        for line in lines:
            smiles = line.strip().split()[0]  # Extract the SMILES string
            charged_positions.append(smiles)
        return charged_positions

    def parse_smiles(self, smiles):
        return CarbocationState(smiles=smiles, other_attributes={})

    def run(self):
        similarity_to_goal = self.calculate_similarity(self.start_state, self.goal_state)
        for _ in range(self.max_iterations):
            node = self.select_node()
            new_child, new_similarity_to_goal = node.expand(self.charged_positions, similarity_to_goal)  # Pass charged_positions here
            if new_child is not None:
                reward = self.simulate(new_child)
                self.backpropagate(new_child, reward)
                self.steps_taken += 1
                if self.steps_taken >= self.max_steps:
                    break
                if new_similarity_to_goal is not None:
                    similarity_to_goal = max(similarity_to_goal, new_similarity_to_goal)
                    print(f"Selected child: {new_child.state.smiles}, Similarity with goal: {new_similarity_to_goal}")
                    if similarity_to_goal >= self.similarity_threshold:
                        break

    def select_node(self):
        node = self.root
        while not node.is_fully_expanded(self.charged_positions) and node.children:
            node = node.select_child()
        return node
 

    def simulate(self, node):
        # Simulate the reward (e.g., by using a simple scoring function)
        return -np.random.uniform(0, 1)

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def calculate_similarity(self, state1, state2):
        node = Node(state1)
        return node.calculate_similarity(state1, state2)
 

    def get_best_child_sequence(self):
        best_sequence = [self.root]
        current_node = self.root
        while current_node.children:
            best_child = max(current_node.children, key=lambda child: child.visits)
            best_sequence.append(best_child)
            current_node = best_child
        return best_sequence

    def run(self, visualize=False):
        all_sequences = []
        similarity_to_goal = self.calculate_similarity(self.start_state, self.goal_state)
        for _ in range(self.max_iterations):
            node = self.select_node()
            new_child, new_similarity_to_goal = node.expand(self.charged_positions, similarity_to_goal)  # Pass charged_positions here
            if new_child is not None:
                reward = self.simulate(new_child)
                self.backpropagate(new_child, reward)
                self.steps_taken += 1
                if self.steps_taken >= self.max_steps:
                    break
                if new_similarity_to_goal is not None:
                    similarity_to_goal = max(similarity_to_goal, new_similarity_to_goal)
                    if similarity_to_goal >= self.similarity_threshold:
                        break
                    if visualize:
                        best_sequence = self.get_best_child_sequence()
                        all_sequences.append((best_sequence, similarity_to_goal))
  

        if visualize:
            self.visualize_routes(all_sequences)

    def visualize_routes(self, all_sequences):
        print("All possible routes and their similarity scores:")
        for i, (sequence, similarity_score) in enumerate(all_sequences, start=1):
            print(f"Route {i}: Similarity Score: {similarity_score}")
            for node in sequence:
                print(f"   {node.state.smiles}")
            print()

    def run_multiple_times(self, num_runs):
        all_routes = []

        for _ in range(num_runs):
            self.root = Node(self.start_state)
            self.steps_taken = 0
            self.run()
            best_sequence = self.get_best_child_sequence()
            similarity_score = self.calculate_sequence_similarity(best_sequence)
            all_routes.append((best_sequence, similarity_score))
        return all_routes

    def save_routes_to_file(self, all_routes, file_name):
        with open(file_name, 'w') as f:
            for i, (sequence, similarity_score) in enumerate(all_routes, start=1):
                f.write(f"Route {i}: Similarity Score: {similarity_score}\n")
                for node in sequence:
                    f.write(f"   {node.state.smiles}\n")
                f.write('\n')
    def calculate_sequence_similarity(self, sequence):
        total_similarity = 0
        sequence_length = len(sequence)
        if sequence_length > 1:
            for i in range(sequence_length - 1):
                total_similarity += self.calculate_similarity(sequence[i].state, sequence[i+1].state)
            return total_similarity / (sequence_length - 1)
        else:
            return 0  # If sequence has only one node, return 0 similarity

# Example usage:
start_smiles = "C[CH+]C(C)CCC=C(C)C"  # SMILES representation of the starting carbocation
goal_smiles = "CC(C)C12CC(C)C(C2)[C+]1"      # SMILES representation of the goal carbocation
smiles_file = "chargedresults.smi"  # Path to the file containing SMILES representations of carbocations
charged_positions_file = "ChargedLocationVector.smi"  # Path to the file containing charged positions
max_iterations = 10
max_steps = 110
similarity_threshold = 0.9
num_runs = 2000  # Number of MCTS runs

mcts_example = CarbocationMCTSExample(start_smiles, goal_smiles, smiles_file, charged_positions_file, max_iterations, max_steps, similarity_threshold)
all_routes = mcts_example.run_multiple_times(num_runs)
mcts_example.save_routes_to_file(all_routes, 'allroutes.smi')

print("done")

