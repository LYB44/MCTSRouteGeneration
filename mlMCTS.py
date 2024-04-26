
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np


def read_smiles_from_file(file_path, num_smiles=2000):
    smiles_list = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= num_smiles:
                break
            smiles = line.strip()
            smiles_list.append(smiles)
    return smiles_list
 
def extract_features_from_smiles(smiles_list):
    features_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            features_list.append(fp)
        else:
            features_list.append(None)
    return features_list

def calculate_similarity_scores(smiles_list, target_smiles):
    similarity_scores_list = []
    for smiles in smiles_list:
        mol1 = Chem.MolFromSmiles(smiles)
        mol2 = Chem.MolFromSmiles(target_smiles)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        similarity_scores_list.append(similarity)
    return similarity_scores_list


# Read the first 2000 SMILES strings from the file
smiles_list = read_smiles_from_file('chargedresults.smi', num_smiles=2000)
# Extract features for the SMILES strings
features_list = extract_features_from_smiles(smiles_list)
# Calculate similarity scores for the first 2000 SMILES strings
target_smiles = 'CC(C)C12CC(C)C(C2)[C+]1'
similarity_scores_list = calculate_similarity_scores(smiles_list, target_smiles)
# Save features and similarity scores to files
np.save('features.npy', np.array(features_list))
np.save('similarity_scores.npy', np.array(similarity_scores_list))
 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# Load features and similarity scores from files
X = np.load('features.npy')
y = np.load('similarity_scores.npy')
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)
# Predict similarity scores for the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
# training data X_train (features) and y_train (similarity scores)
X_train = np.load('features.npy')
y_train = np.load('similarity_scores.npy')
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor()
# Fit the model
model.fit(X_train, y_train)
# Pass the trained model instance to the CarbocationMCTSExample class
routes = []
# Run the MCTS algorithm 2000 times
for _ in range(2000):
    mcts_example = CarbocationMCTSExample(start_smiles, goal_smiles, smiles_file, charged_positions_file, max_iterations, max_steps, similarity_threshold, model)
    mcts_example.run()
    best_sequence = mcts_example.get_best_child_sequence()
    route_smiles = [node.state.smiles for node in best_sequence]
    routes.append(route_smiles)

# Save the routes to a file
with open('mlallroutes.smi', 'w') as f:
    for route in routes:
        f.write('\n'.join(route) + '\n\n')
print("Routes saved to mlallroutes.smi")

 
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

def calculate_braun_blanquet_similarity(route):
    # Load the route as molecules
    mols = [Chem.MolFromSmiles(smiles) for smiles in route]
    # Calculate fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]
    # Calculate similarity scores
    similarity_scores = []
    for i in range(len(fps) - 1):
        for j in range(i + 1, len(fps)):
            similarity = DataStructs.BraunBlanquetSimilarity(fps[i], fps[j])
            similarity_scores.append(similarity)

    # Calculate Braun-Blanquet similarity score
    bb_score = np.mean(similarity_scores)
    return bb_score

# Load routes from file
routes = []
with open('mlallroutes.smi', 'r') as f:
    current_route = []
    for line in f:
        line = line.strip()
        if line:
            current_route.append(line)
        else:
            routes.append(current_route)
            current_route = []


# Calculate Braun-Blanquet similarity scores for each route
bb_similarity_scores = []
for idx, route in enumerate(routes, start=1):
    bb_score = calculate_braun_blanquet_similarity(route)
    num_steps = len(route)
    bb_similarity_scores.append((num_steps, bb_score))
    print(f"Route {idx}: Similarity Score: {bb_score}, Number of Steps: {num_steps}")

# Save Braun-Blanquet similarity scores to a new file
with open('ssmlallrouteswithnumsteps.smi', 'w') as f:
    for idx, (num_steps, bb_score) in enumerate(bb_similarity_scores, start=1):
        f.write(f"Route {idx}: Similarity Score: {bb_score}, Number of Steps: {num_steps}\n")

