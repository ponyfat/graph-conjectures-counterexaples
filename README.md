# Automatic search for graph conjectures counterexamples

## Description and motivation
The code is based on the https://arxiv.org/abs/2104.14516 article. The author exploits the usage of RL deep cross-entropy algorithm in order to find counterexaples to the known conjectures in combinatorics.

## How to run
First install all the needed dependencies
```pip install -r requirements.txt```
Then adjust the model and learning parameters in model.py and constants.py files. Also set the required scorer in scores.py. Then you can simply run the main.py and enjoy the tensorboard graphs. 

# Experiments
The logs directory already contains two of my successful experiments for conjectures 2.1 and 2.3 from the above mentioned paper.
