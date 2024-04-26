# MCTSRouteGeneration
# Carbocation MCTS Algorithm for Route Generation

 

This repository contains an implementation of the Monte Carlo Tree Search (MCTS) algorithm for generating routes to monoterpene carbocations.

 

## Overview

 

Carbocations are important intermediates in monoterpene synthesis chemistry, and predicting efficient routes for their formation is essential for future synthesis of monoterpenes. This project explores the use of MCTS, a tree search algorithm commonly used in game AI, to generate optimal routes for transforming one carbocation into another based on charged positions and molecular similarity. It makes use of the gneraion of molecular fingerprints and Braun-Blanquet Similarity scores to predict proximity of carbocations within a sequence.

 

## Files

 

- `carbocation_mcts.py`: Main script containing the MCTS algorithm implementation.

- `chargedresults.smi`: File containing SMILES representations of carbocations generated using Surge, an open-source chemical graph generator.

- `ChargedLocationVector.smi`: File containing charged positions data for carbocations, also generated using Surge.

 

## Usage

 

1. **Clone the Repository**:

 

   ```bash

   git clone https://github.com/<your_username>/Carbocation-MCTS-Route-Generation.git

 
## Install Dependencies
Ensure to install the required dependencies before running the MCTS algorithm. You can install the dependencies using pip:



```bash 
pip install rdkit
```


## Run the MCTS Algorithm

To run the MCTS algorithm for route generation, execute the following command


```bash
python carbocation_mcts.py
```


## Requirements
- Python 3.x
- RDKit (Cheminformatics library for Python)

## Acknowledgements
Spacial thanks to the developers of RDKit and Surge for their contributions to cheminformaics and chemical structure generation.
