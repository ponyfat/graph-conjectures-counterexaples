"""Experiments constants"""

# number of vertices in the graph. Only used in the reward function,
# not directly relevant to the algorithm
N = 30

# The length of the word we are generating. Here we are generating a graph,
# so we create a 0-1 word of length (N choose 2)
WORD_LEN = int(N * (N - 1) / 2)

# Leave this at 2*WORD_LEN. The input vector will have size 2*WORD_LEN,
# where the first WORD_LEN letters encode our partial word
# (with zeros on the positions we haven't considered yet),
# and the next WORD_LEN bits one-hot encode which letter we are considering now.
# So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are
# considering the third letter now.
OBSERVATION_LEN = 2 * WORD_LEN

# number of new sessions per iteration
N_CANDIDATES = 1000

# top 1-X percentile we are learning from
BEST_PERCENTILE = 0.93

# top 1-X percentile that survives to next iteration
CARRYOVER_PERCENTILE = 0.94
CARRYOVER_NUM = int(N_CANDIDATES * (1 - CARRYOVER_PERCENTILE))

# The size of the alphabet. In this file we will assume this is 2.
# There are a few things we need to change when the alphabet size is larger,
# such as one-hot encoding the input, and using categorical_crossentropy
# as a loss function.
n_actions = 2

state_dim = (OBSERVATION_LEN,)
INF = 1000000