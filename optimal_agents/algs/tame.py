from dataclasses import dataclass
import os
import numpy as np
import itertools
import torch
import torch_geometric

from optimal_agents.morphology import Morphology
from optimal_agents.utils.loader import get_env, get_morphology
from optimal_agents.utils.tester import eval_policy
from optimal_agents.policies import random_policies
from optimal_agents.policies import predictive_models

import random

@dataclass
class Individual:
    morphology: Morphology
    fitness: float
    start_index: int
    end_index: int
    index: int


class TAME(object):
    
    def __init__(self, params, eval_ep=8, save_freq=10, 
                       vf_lr=0.005, vf_batch_size=128, vf_n_epochs=10,
                       global_state=False, num_freqs=2, num_phases=2, sample_freq=50,
                       state_noise=0.0, keep_percent=0.0, reset_freq=-1,
                       matching_noise=False, random_policy="CosinePolicy", vf_arch=[192, 192, 192], classifier=None,
                       include_segments=False, include_end=False, num_joint_regularizer=0.0):

        # Save the model parameters and mutation parameters
        self.params = params
        self.mutation_kwargs = params['mutation_kwargs']
        self.keep_percent = keep_percent
        self.num_joint_regularizer = num_joint_regularizer

        # Save information for data collection
        self.eval_ep = eval_ep
        self.save_freq = save_freq
        self.matching_noise = matching_noise
        self.state_noise = state_noise
        self.global_state = global_state
        self.include_segments = include_segments
        self.include_end = include_end

        # Save info for model training
        self.lr = vf_lr
        self.batch_size = vf_batch_size
        self.n_epochs = vf_n_epochs
        self.reset_freq = reset_freq

        # Save the classifer type
        self.classifier_cls = vars(predictive_models)[classifier]
        self.vf_arch = vf_arch

        # Construct the Random Policy
        self.max_nodes = self.mutation_kwargs['max_nodes'] if 'max_nodes' in self.mutation_kwargs else 12
        random_policy_kwargs = {'sample_freq': sample_freq, 'num_freqs' : num_freqs, 'num_phases' : num_phases}
        random_policy_cls = vars(random_policies)[random_policy]
        self.policy = random_policy_cls(self.max_nodes, **random_policy_kwargs)

        self.buffer = []
        self.population = []

        test_morph = get_morphology(self.params)
        test_env = get_env(params, morphology=test_morph)
        self.state_dim = test_env.observation_space['x'].shape[1] # Get this by creating an enviornment and getting obs.shape[1]
        self.morphology_dim = test_env.get_morphology_obs(test_morph, include_segments=self.include_segments)['x'].shape[1]
        test_env.close()
        del test_env

        self._build_model()

    def _build_model(self):
        ''' Re-create model and optimizer for resetting '''
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "optim"):
            del self.optim
        if hasattr(self, "criterion"):
            del self.criterion

        self.model = self.classifier_cls(self.state_dim + self.morphology_dim, self.policy.num_actions, net_arch=self.vf_arch) 
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.policy.num_actions)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def preprocess_batch(self, data, noise=True):
        if noise and self.state_noise > 0:
            if self.matching_noise:
                # Need to apply the same noise to every data point.
                assert self.state_dim % 3 == 0, "Matching noise requires state dimension to be a multiple of 3."
                noisy_data = [] 
                for data_pt in data:
                    x = data_pt.x.clone()
                    noise = self.state_noise * torch.randn(1, 3) # All noise is on 3 position vectors x, y z
                    noise = noise.repeat(1, self.state_dim  // 3) 
                    x[:, :self.state_dim] += noise # Add noise to each limb differently
                    noisy_data.append(torch_geometric.data.Data(x=x, edge_index=data_pt.edge_index, edge_attr=data_pt.edge_attr, y=data_pt.y))
                return torch_geometric.data.Batch.from_data_list(noisy_data)
            else:
                # Apply different noise to every single node.
                batch = torch_geometric.data.Batch.from_data_list(data)
                x = batch.x.clone()
                noise = self.state_noise * torch.randn(x.shape[0], self.state_dim)
                x[:, :self.state_dim] += noise # Add noise to each limb differently
                batch.x = x
                return batch
        else:
            return torch_geometric.data.Batch.from_data_list(data)

    def update_model(self):
        # Train the model
        for epoch in range(self.n_epochs):
            num_data_pts, num_correct = 0, 0
            perm = np.random.permutation(len(self.buffer))
            num_full_batches = len(perm) // self.batch_size
            for i in range(num_full_batches + 1):
                if i != num_full_batches:
                    inds = perm[i*self.batch_size:(i+1)*self.batch_size]
                else:
                    inds = perm[i*self.batch_size:]
                if len(inds) == 0:
                    continue
                batch = self.preprocess_batch([self.buffer[ind] for ind in inds])
                self.optim.zero_grad()
                pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = self.criterion(pred, batch.y)
                loss.backward()
                self.optim.step()
                with torch.no_grad():
                    pred_labels = torch.argmax(pred, dim=1)
                    num_correct += torch.sum(pred_labels == batch.y).item()
                    num_data_pts += pred.shape[0]
            print("Epoch", epoch, "Loss", loss.item(), "Acc", num_correct / num_data_pts)
    
    def update_population(self):
        # Iterate through every individual in the population and determine the fitness.
        for individual in self.population:
            # Update the fitness by constructing data
            data = self.buffer[individual.start_index:individual.end_index]
            batch = self.preprocess_batch(data)
            with torch.no_grad():
                pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) # No noise added during eval
                fitness = -1*self.criterion(pred, batch.y).item()
                fitness = np.power(np.log(individual.morphology.num_joints), self.num_joint_regularizer) * (np.log(self.policy.num_actions) + fitness)
            individual.fitness = fitness

    def add_morphology(self, morphology):
        try:
            env = get_env(self.params, morphology=morphology)
            morphology_obs = env.get_morphology_obs(morphology, include_segments=self.include_segments)

            states = []
            labels = []
            action_dim = env.action_space.shape[0] # Get the size of the morphology action space.
            for i in range(self.eval_ep):
                actions, label = self.policy.step(1000) # generate action sequences and corresponding labels.
                done = False
                action_idx = 0
                obs = env.reset()
                while not done:
                    obs, _, done, _ = env.step(actions[action_idx, :action_dim])
                    action_idx += 1
                states.append(obs['x'])
                labels.append(label[:action_dim])

            data = []
            for state, label in zip(states, labels):
                edge_index = np.concatenate((morphology_obs['edge_index'], np.roll(morphology_obs['edge_index'] , 1, axis=1)), axis=0)
                y = torch.from_numpy(label)
                # TODO: Handle ignore index in the Node Case.
                y[0] = self.policy.num_actions # Set the value here so that we ignore predicting the root node with cross entropy loss
                x = torch.from_numpy(np.concatenate((state, morphology_obs['x']), axis=1).astype(np.float32))
                edge_index = torch.from_numpy(edge_index).t().contiguous()
                data.append(torch_geometric.data.Data(x=x, edge_index=edge_index, y=y))

            data_start_idx = len(self.buffer)
            self.buffer.extend(data)
            data_end_idx = len(self.buffer)

            individual = Individual(morphology, -np.inf, data_start_idx, data_end_idx, len(self.population))
            self.population.append(individual)
        except:
            print("Encountered exception adding morphology. Skipping for now. Investigate cause later.")
            return

    def learn(self, path, population_size, num_generations):
        # Generate the initial population
        os.makedirs(path, exist_ok=True)

        for _ in range(population_size*2):
            self.add_morphology(get_morphology(self.params))

        for gen_idx in range(num_generations):
            self.update_model()
            self.update_population()
            self.population.sort(key=lambda individual: individual.fitness, reverse=True)

            with open(os.path.join(path, "gen" + str(gen_idx) + ".txt"), "w+") as f:
                for individual in self.population:
                    f.write(str(individual.fitness) + " " + str(individual.index) + "\n")

            if (gen_idx + 1) % self.save_freq == 0:
                gen_path = os.path.join(path, 'gen_' + str(gen_idx))
                os.makedirs(gen_path, exist_ok=True)
                self.params.save(gen_path)
                for i, individual in enumerate(self.population):
                    individual.morphology.save(os.path.join(gen_path, str(i) + '.morphology.pkl'))

            # If were at the end, exit so we don't add extra morphologies.
            if gen_idx + 1 == num_generations:
                break

            # Take population size samples from the population to construct new morphologies.
            for i in range(population_size):
                if i >= 3 and self.keep_percent > 0 and int(self.keep_percent*len(self.population)) > 5:
                    morphology = random.choice(self.population[:max(int(len(self.population)*self.keep_percent),1)]).morphology
                else:
                    morphology = self.population[i].morphology # sample a new morphology according to fitness level
                new_morphology = morphology.mutate(**self.mutation_kwargs)
                self.add_morphology(new_morphology)

            # If reset freq, reset the model
            if self.reset_freq > 0 and gen_idx % self.reset_freq == 0 and gen_idx > 1:
                self._build_model()

            print("Finished Gen", gen_idx)

        assert self.population[0].fitness == max([individual.fitness for individual in self.population]), "Error, didn't get max fitness individual"

        morphology = self.population[0].morphology
        morphology.save(os.path.join(path, "best_morphology.pkl"))
