from torch import nn
import torch
from torch import optim
import mbrl
from mbrl.infrastructure.utils import normalize, unnormalize
from mbrl.infrastructure import pytorch_util as ptu
from mbrl.models.base_model import BaseModel

class ModelWithReward(nn.Module):
    def __init__(self, input_size, output_size, n_layers, size):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.SELU())
            input_size = size

        self.features = nn.Sequential(*layers)
        self.ob_head = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.SELU(),
            nn.Linear(input_size, output_size))
        self.reward_head = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.SELU(),
            nn.Linear(input_size, 1))
        
    
    def forward(self, x):
        x = self.features(x)
        reward_pred = self.reward_head(x)
        ob_pred = self.ob_head(x)
        return ob_pred, reward_pred.view(-1)


class FFRewardModel(nn.Module, BaseModel):
    def __init__(self, ac_dim, ob_dim, n_layers, size, learning_rate=0.001):
        super().__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.delta_network = ModelWithReward(
            input_size=self.ob_dim + self.ac_dim,
            output_size=self.ob_dim,
            n_layers=self.n_layers,
            size=self.size)
        
        self.delta_network.to(ptu.device)
        self.optimizer = optim.Adam(
            self.delta_network.parameters(),
            self.learning_rate,
        )
        self.loss = nn.MSELoss()
        self.obs_mean = None
        self.obs_std = None
        self.acs_mean = None
        self.acs_std = None
        self.delta_mean = None
        self.delta_std = None

    def update_statistics(
            self,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        self.obs_mean = ptu.from_numpy(obs_mean)
        self.obs_std = ptu.from_numpy(obs_std)
        self.acs_mean = ptu.from_numpy(acs_mean)
        self.acs_std = ptu.from_numpy(acs_std)
        self.delta_mean = ptu.from_numpy(delta_mean)
        self.delta_std = ptu.from_numpy(delta_std)

    def forward(
            self,
            obs_unnormalized,
            acs_unnormalized,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        """
        :param obs_unnormalized: Unnormalized observations
        :param acs_unnormalized: Unnormalized actions
        :param obs_mean: Mean of observations
        :param obs_std: Standard deviation of observations
        :param acs_mean: Mean of actions
        :param acs_std: Standard deviation of actions
        :param delta_mean: Mean of state difference `s_t+1 - s_t`.
        :param delta_std: Standard deviation of state difference `s_t+1 - s_t`.
        :return: tuple `(next_obs_pred, delta_pred_normalized)`
        This forward function should return a tuple of two items
            1. `next_obs_pred` which is the predicted `s_t+1`
            2. `delta_pred_normalized` which is the normalized (i.e. not
                unnormalized) output of the delta network. This is needed
        """
        # normalize input data to mean 0, std 1
        obs_normalized = normalize(obs_unnormalized, obs_mean, obs_std)
        
        acs_normalized = normalize(acs_unnormalized, acs_mean, acs_std)
        # predicted change in obs
        concatenated_input = torch.cat([obs_normalized, acs_normalized], dim=1)

        # TODO(Q1) compute delta_pred_normalized and next_obs_pred
        # Hint: as described in the PDF, the output of the network is the
        # *normalized change* in state, i.e. normalized(s_t+1 - s_t).
        delta_pred_normalized, reward_pred = self.delta_network(concatenated_input)
        next_obs_pred = obs_unnormalized + unnormalize(delta_pred_normalized, delta_mean, delta_std) 

        return next_obs_pred, delta_pred_normalized, reward_pred


    def get_prediction_with_reward(self, obs, acs, data_statistics):
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: a numpy array of the predicted next-states (s_t+1)
        """
        data_statistics = { k: ptu.from_numpy(v) for k, v in data_statistics.items() }

        pred, _, reward_pred = self.forward(
            ptu.from_numpy(obs),
            ptu.from_numpy(acs),
            data_statistics["obs_mean"],
            data_statistics["obs_std"],
            data_statistics["acs_mean"],
            data_statistics["acs_std"],
            data_statistics["delta_mean"],
            data_statistics["delta_std"])

        return ptu.to_numpy(pred), ptu.to_numpy(reward_pred)

    def get_prediction(self, obs, acs, data_statistics):
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: a numpy array of the predicted next-states (s_t+1)
        """
        ob_pred, _ = self.get_prediction_with_reward(obs, acs, data_statistics)

        return ob_pred

    def update(self, obs, acs, next_obs, rewards, data_statistics):
        """
        :param obs: numpy array of observations
        :param acs: numpy array of actions
        :param next_obs: numpy array of next observations
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return:
        """ 
        data_statistics = {k: ptu.from_numpy(v) for k, v in data_statistics.items()}
        obs = ptu.from_numpy(obs)
        next_obs = ptu.from_numpy(next_obs)
        acs = ptu.from_numpy(acs)
        rewards = ptu.from_numpy(rewards)

        # compute the normalized target for the model
        target = normalize(
            next_obs - obs,
            data_statistics["delta_mean"],
            data_statistics["delta_std"])

        _, pred_delta, reward_pred = self.forward(
            obs,
            acs,
            data_statistics["obs_mean"],
            data_statistics["obs_std"],
            data_statistics["acs_mean"],
            data_statistics["acs_std"],
            data_statistics["delta_mean"],
            data_statistics["delta_std"])

        loss = self.loss(pred_delta, target)
        loss = loss + self.loss(reward_pred, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }