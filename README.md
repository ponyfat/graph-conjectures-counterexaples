# Automatic search for graph conjectures counterexamples

## Description and motivation
The code is based on the https://arxiv.org/abs/2104.14516 article. The author exploits the usage of RL deep cross-entropy algorithm in order to find counterexaples to the known conjectures in combinatorics.

## How to run
First install all the needed dependencies

```
pip install -r requirements.txt
```

Then adjust the model and learning parameters in model.py and constants.py files (please find the detailed parameter descriptions there). Also set the required scorer in scores.py.

If you want yo speed up calculations or get rid of uneccessary parallelization set the number of threads suitable for you in main.py. Then you can simply run the main.py.

```
python main.py
```

And enjoy the tensorboard by running.
```
tensorboard --logdir logs
```

## Experiments
The logs directory already contains two of my successful experiments for conjectures 2.1 and 2.3 from the above mentioned paper. However, it makes the download process much slower, so if not needed - omit the logs dir.

Playground notebook shows an example of how to load and nicely display a networkx graph from .pt format.
The tensorboard can be set up using playground notebook. Just type: ```%tensorboard --logdir logs```.

## Run example
This is the example of tb-logs for Conjecture 2.1 from original paper.

Model loss over iterations (can be seen in tensorboard):

![conj21_loss](https://user-images.githubusercontent.com/26412001/138604873-dda77a97-75b9-4c5d-8308-d0cf5926720d.png)

Mean reward of top6% generated examples in each iteration (can be seen in tensorboard):

![conj21_reward](https://user-images.githubusercontent.com/26412001/138604912-a9ee9f12-cc94-44e7-904b-93e87efe49bc.png)

Top 9 graphs on iteration 100 (can be seen in tensorboard, as well as top 9 graphs on other iterations):

![conj21_100](https://user-images.githubusercontent.com/26412001/138605027-d9065829-84d7-47bb-946a-1d7bec896150.png)

The counterexample"graph:

![conj21_contr](https://user-images.githubusercontent.com/26412001/138605037-94959d26-c59c-46bb-9b12-8365dcdbb959.png)
