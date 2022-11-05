from mbrl.agents.mb_agent import MBAgent
from mbrl.policies.MPC_discrete_policy import MPCDiscretePolicy 
from  mbrl.models.ff_reward_model import FFRewardModel
import numpy as np

class MBDiscreteAgent(MBAgent):
    def __init__(self, env, agent_params):
        super().__init__(env, agent_params)
    
    def _create_policy_and_model(self):
        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFRewardModel(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['learning_rate'],
            )
            self.dyn_models.append(model)
        
        self.actor = MPCDiscretePolicy(
            self.env,
            ac_dim=self.agent_params['ac_dim'],
            dyn_models=self.dyn_models,
            horizon=self.agent_params['mpc_horizon'],
            N=self.agent_params['mpc_num_action_sequences'],
            sample_strategy=self.agent_params['mpc_action_sampling_strategy'])

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data / self.ensemble_size)

        for i in range(self.ensemble_size):

            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_ens variable defined above useful

            observations = ob_no[i*num_data_per_ens:(i+1)*num_data_per_ens]
            actions = ac_na[i*num_data_per_ens:(i+1)*num_data_per_ens]
            next_observations = next_ob_no[i*num_data_per_ens:(i+1)*num_data_per_ens]
            rewards = re_n[i*num_data_per_ens:(i+1)*num_data_per_ens]


            # use datapoints to update one of the dyn_models
            model =  self.dyn_models[i]
            log = model.update(observations, actions, next_observations, rewards,
                                self.data_statistics)
            loss = log['Training Loss']
            losses.append(loss)

        avg_loss = np.mean(losses)
        return {
            'Training Loss': avg_loss,
        }