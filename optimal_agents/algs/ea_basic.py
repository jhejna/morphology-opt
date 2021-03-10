import os
import shutil
import subprocess
import copy
import tempfile
import numpy as np
import torch
import torch_geometric
from optimal_agents.morphology import Morphology
from optimal_agents.algs.ea_base import EvoAlg, Individual
from optimal_agents.utils.loader import get_env
from optimal_agents.policies.pruning_models import NodeMorphologyVF

import random

class BasicEA(EvoAlg):

    def __init__(self, *args, num_cores=4, cpus_per_ind=1, save_freq=1, eval_ep=10, **kwargs):
        super(BasicEA, self).__init__(*args, **kwargs)
        # Save the extra args for basic EA
        self.num_cores = num_cores
        self.cpus_per_ind = cpus_per_ind
        self.save_freq = save_freq # TODO: Currently doing nothing with save_freq
        self.eval_ep = eval_ep

    def _mutate(self, individual):
        return Individual(individual.morphology.mutate(**self.mutation_kwargs))
    
    def _train_policies(self, gen_idx):
        batch_size = self.num_cores // self.cpus_per_ind
        cpu_assignment_starts = [self.cpus_per_ind * k for k in range(batch_size)]
        run_path = os.path.join(os.path.dirname(__file__), 'ea_subproc.py')
        # Loop population size // batch_size times.
        for i in range(0, len(self.population), batch_size):
            processes = []
            for j, individual in enumerate(self.population[i:min(i+batch_size, len(self.population))]):
                _, individual_params_path = tempfile.mkstemp(text=True, prefix='params', suffix='.yaml')
                individual_params = copy.deepcopy(self.params) 
                if gen_idx == 0 and not self.retrain:
                    individual_params['learn_kwargs']['total_timesteps'] *= 2 # Train double length on gen 0
                individual_params['name'] = os.path.join("gen_" + str(gen_idx), "ind_" + str(individual.index))
                individual_params.save(individual_params_path)

                cmd_args = []
                cpus = ",".join([str(cpu) for cpu in range(cpu_assignment_starts[j], cpu_assignment_starts[j] + self.cpus_per_ind)])
                cmd_args.extend(["taskset", "-c", cpus])
                cmd_args.extend(["python", run_path, "--params", individual_params_path, "--base-path", self.path, '--eval-ep', str(self.eval_ep)])
                
                _, morphology_path = tempfile.mkstemp(text=False, prefix='morphology', suffix='.pkl')
                
                individual.morphology.save(morphology_path)
                cmd_args.extend(["--morphology", morphology_path])

                if not individual.model is None and not self.retrain:
                    cmd_args.extend(["--model-path", individual.model])
                
                cmd_args.extend(self._get_pruning_cmd_args(individual))

                proc = subprocess.Popen(cmd_args)
                processes.append(proc)
            
            print("Waiting for completion of ", len(processes), "training jobs.")
            for p in processes:
                if p is not None:
                    p.wait()

        # Now pull the results and update the population by population.update
        generation_folder = os.path.join(self.path, 'gen_' + str(gen_idx))
        num_updates = 0
        for output in os.listdir(generation_folder):
            # Get the correct individual from the population and update it.
            idx = int(output.split('_')[1]) # The second item in the name gives the index.
            model_path = os.path.join(generation_folder, output)
            with open(os.path.join(model_path, 'fitness.tmp'), 'r') as f:
                fitness = float(f.read())
            self.population[idx].update(fitness, model_path)
            num_updates += 1
        assert num_updates == len(self.population), "Did not update once per individual in population"


    def _get_pruning_cmd_args(self, individual):
        return []

    def _clean(self, gen_idx):
        # Remove files that are not in correspondence with gen_idx
        # Note that we will use current gen as next gen references its policies
        gen_idx_to_remove = gen_idx - 1
        if (gen_idx_to_remove + 1) % self.save_freq == 0 or gen_idx_to_remove < 1:
            return # return if that was generation we were going to keep
        remove_gen_folder = os.path.join(self.path, 'gen_' + str(gen_idx_to_remove))
        shutil.rmtree(remove_gen_folder) # Delete the entire folder, cleaning up gb of data

class BasicEA_VFPrune(BasicEA):

    def __init__(self, *args, vf_lr=0.001, vf_batch_size=64, 
                       vf_n_epochs=5, vf_buffer_size=1024, 
                       include_segments=False, thompson=False, **kwargs):
        super(BasicEA_VFPrune, self).__init__(*args, **kwargs) # Pass through the remaining args
        self.pruning_model = NodeMorphologyVF(self.params, lr=vf_lr, batch_size=vf_batch_size, buffer_size=vf_buffer_size,
                                             include_segments=include_segments, 
                                             thompson=thompson)
        self.vf_n_epochs = vf_n_epochs
        self.include_segments = include_segments

    def _get_pruning_cmd_args(self, individual):
        cmd_args = ["--eval-data", str(1)]
        if self.include_segments:
            cmd_args.append("--segment-embeddings")
        return  cmd_args # Determine extra commands

    def _pruning_update(self, gen_idx):
        if gen_idx == 0:
            with open(os.path.join(self.path, 'value_loss.txt'), 'w+') as f:
                pass
        value_data = []
        for individual in self.population:
            if not individual.model is None:
                # Get the data. 
                data_path = os.path.join(individual.model, 'pruning_data.pkl')
                if os.path.exists(data_path):
                    value_data.extend(torch.load(data_path))
                else:
                    print("For individual with trained model value data did not exist.")
            else:
                print("For individual model was none.")
        vf_loss = self.pruning_model.update(value_data, n_epochs=self.vf_n_epochs)
        with open(os.path.join(self.path, 'value_loss.txt'), 'a') as f:
            f.write(str(vf_loss) + "\n")
