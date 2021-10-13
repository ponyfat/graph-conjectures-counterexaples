from typing import Tuple, Optional

import numpy as np
import tensorboardX
import torch
from torch import nn

FIRST_LAYER_NEURONS = 128  # Number of neurons in the hidden layers.
SECOND_LAYER_NEURONS = 64
THIRD_LAYER_NEURONS = 4


class LinearModel(nn.Module):
    """Linear sequence generator"""

    def __init__(self, in_ch: int, out_ch: int, lr: float = 1e-3):
        """Initialize LinearModel.

        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            lr (float): learning rate
        """
        super().__init__()
        self.in_ch = in_ch
        self.net = nn.Sequential(
            nn.Linear(in_ch, FIRST_LAYER_NEURONS),
            nn.ReLU(),
            nn.Linear(FIRST_LAYER_NEURONS, SECOND_LAYER_NEURONS),
            nn.ReLU(),
            nn.Linear(SECOND_LAYER_NEURONS, THIRD_LAYER_NEURONS),
            nn.ReLU(),
            nn.Linear(THIRD_LAYER_NEURONS, out_ch),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

    def fit(self,
            xs: torch.tensor,
            ys: torch.tensor,
            logger_info: Optional[Tuple[tensorboardX.SummaryWriter, int]] = None,
        ):
        """Fit model.

        Args:
            xs (torch.tensor): training data
            ys (torch.tensor): training targets
        """
        self.net.train()
        running_loss = []
        for epoch_num in range(1):
            epoch_loss = []
            for x, y in zip(xs, ys):
                self.optimizer.zero_grad()
                out = self.net(x)
                loss = self.criterion(out.squeeze(1), y)
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

                # # this logging crashes tensorboard :(
                # if logger_info is not None:
                #     logger, iter = logger_info
                #     logger.add_scalar(
                #         'epoch_train_crossentropy',
                #         loss.item(),
                #         iter * 1 + epoch_num
                #     )
            running_loss.append(np.mean(epoch_loss))
        # # alternative all-at-once step
        # self.optimizer.zero_grad()
        # out = self.net(xs.reshape(-1, self.in_ch))
        # loss = self.criterion(out.squeeze(1), ys.reshape(-1))
        # loss.backward()
        self.optimizer.step()

        if logger_info is not None:
            logger, iter = logger_info

            logger.add_scalar(
                'mean_train_crossentropy',
                np.mean(running_loss),
                iter
            )

    def predict(self, x: torch.tensor) -> torch.tensor:
        """Predict with model.

        Args:
            x (torch.tensor): data

        Returns:
            torch.tensor: predictions
        """
        with torch.no_grad():
            self.net.eval()
            return self.net(x)


class ModelPlaceHolder(torch.nn.Module):
    """Dumb placeholder for testing"""
    def predict(self, x):
        return torch.randn(x.shape[0], 2)