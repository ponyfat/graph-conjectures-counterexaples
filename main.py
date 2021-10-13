"""Graph conjecture counterexamples finder."""
import os
from multiprocessing import Pool
from typing import Tuple, Callable

import torch
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
from joblib import Parallel, delayed

from constants import WORD_LEN, OBSERVATION_LEN, N_CANDIDATES, \
    BEST_PERCENTILE, CARRYOVER_NUM
from model import LinearModel
from torchvision.utils import make_grid

from scores import conj21_score, conj23_score
from utils import get_graph_image, seed_everything, CounterexampleFoundException


def generate_candidates(
        model: Callable,
        n_candidates: int,
        scoring_function: Callable,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Generate candidate sequences.

    Args:
        model (Callable): instance, having predict method to return logits
        n_candidates (int): number of candidate sequences to generate
        scoring_function (Callable): function to compute rewards
            of the final states

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor]:
            states, actions and resulting_rewards
    """
    all_states = torch.zeros(n_candidates, WORD_LEN, OBSERVATION_LEN)
    actions = torch.zeros(n_candidates, WORD_LEN)
    resulting_rewards = torch.zeros(n_candidates)

    # set 'letter being generated' indexes for all generation steps
    all_states[:, :, WORD_LEN:] = torch.eye(WORD_LEN)

    # iterate over generations steps i.e. second dim
    for step_num, curr_states in enumerate(all_states.transpose(0, 1)):
        pos_probs = torch.sigmoid(model.predict(curr_states))
        probs = torch.cat([1 - pos_probs, pos_probs], dim=1)
        actions[:, step_num] = torch.multinomial(probs, num_samples=1).squeeze(1)

        # update next status with inferred action
        if step_num < WORD_LEN - 1:
            next_state = curr_states.clone()
            next_state[:, step_num] = actions[:, step_num]
            all_states[:, step_num + 1, :WORD_LEN] = next_state[:, :WORD_LEN]
        else:
            final_states = curr_states.clone()
            final_states[:, step_num] = actions[:, step_num]

            # # Multiprocessing
            # with Pool(4) as p:
            #     resulting_rewards = p.map(
            #         scoring_function,
            #         final_states.reshape(10, 100, -1)
            #     )
            # resulting_rewards = torch.cat(resulting_rewards, dim=0)

            # # Single thread - used for conj21
            # resulting_rewards = scoring_function(final_states)

            # Joblib - use for conj23
            resulting_rewards = torch.cat(
                Parallel(n_jobs=4)(
                    delayed(scoring_function)(state)
                    for state in final_states.reshape(10, 100, -1)
                ),
                dim=0,
            )

    return all_states, actions, resulting_rewards


def main():
    """Main routine."""
    print(f'saving logs to logs/run_{os.getpid()}_conj23')
    logger = SummaryWriter(logdir=f'logs/run_{os.getpid()}_conj23')

    carryover_states = torch.Tensor([])
    carryover_actions = torch.Tensor([])
    carryover_rewards = torch.Tensor([])
    for iter_num in tqdm(range(NUM_ITERS)):
        try:
            states, actions, rewards = generate_candidates(
                model,
                N_CANDIDATES,
                conj23_score,
            )
        except CounterexampleFoundException as e:
            torch.save(e.state, f'run_{os.getpid()}_counterexample.pt')
            counterexample_img = get_graph_image(e.state[:WORD_LEN])
            counterexample_grid = make_grid(
                counterexample_img,
                normalize=False,
                scale_each=True,
                nrow=1,
            )
            logger.add_image('counterexample', counterexample_grid, iter_num)

            logger.add_scalar(
                'mean_reward_best',
                e.score,
                iter_num,
            )

            logger.add_scalar(
                'mean_reward_10%',
                e.score,
                iter_num,
            )
            break

        states = torch.cat([states, carryover_states], dim=0)
        actions = torch.cat([actions, carryover_actions], dim=0)
        rewards = torch.cat([rewards, carryover_rewards], dim=0)

        best_threshold = torch.quantile(rewards, BEST_PERCENTILE)
        ten_procent_threshold = torch.quantile(rewards, 0.9)
        ten_procent_mask = (rewards >= ten_procent_threshold)
        best_candidates_mask = (rewards >= best_threshold)

        model.fit(
            states[best_candidates_mask],
            actions[best_candidates_mask],
            logger_info=(logger, iter_num),
        )

        # this selection leads to memory explosion when many similar good graphs
        # are generated. For example, having 900 copies of one good graph and
        # 100 bad graphs will lead to the selection of at least 900 graphs to
        # carry over.
        # carryover_threshold = torch.quantile(rewards, CARRYOVER_PERCENTILE)
        # carryover_candidates_mask = (rewards >= carryover_threshold)

        _, carryover_indices = rewards.topk(CARRYOVER_NUM)
        carryover_states = states[carryover_indices]
        carryover_actions = actions[carryover_indices]
        carryover_rewards = rewards[carryover_indices]

        logger.add_scalar(
            'mean_reward_best',
            rewards[best_candidates_mask].mean(),
            iter_num,
        )
        logger.add_scalar(
            'mean_reward_all',
            rewards.mean(),
            iter_num,
        )
        logger.add_scalar(
            'mean_reward_10%',
            rewards[ten_procent_mask].mean(),
            iter_num,
        )

        if (iter_num) % 10 == 0:
            _, best_graphs_idxs = rewards.topk(9)
            top_imgs = get_graph_image(states[best_graphs_idxs][:, -1, :WORD_LEN])
            top_grid = make_grid(top_imgs, normalize=False, scale_each=True, nrow=3)
            logger.add_image('top graphs', top_grid, iter_num)
    logger.close()


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("lr", help="learning_rate",
    #                     const=1e-4,
    #                     type=float)
    # args = parser.parse_args()

    seed_everything(os.getpid())
    lr = 1e-1
    model = LinearModel(OBSERVATION_LEN, 1, lr=lr)
    print(f'model with lr={lr}')
    print('SGD optimizer')
    NUM_ITERS = 20000
    torch.set_num_threads(2)
    main()
