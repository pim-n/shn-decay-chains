import numpy as np
import pandas as pd
from tqdm import tqdm

from _utils.termcolors import termcolors as tc

from _simulations.state import State
from _simulations.distributions import Distribution
from _simulations.event import Event

from _stats.schmidt_test import schmidt_test, generalised_schmidt_test as gst

class Chain:
    """
    Chain class generates a 1D array of State objects, with randomly chosen paths based on the probabilities given by available branches.

    Arguments:
        initial_state   The initial state from which to start the chain

    Attributes:
        chain           The simulated chain of decays
    """


    def __init__(self, initial_state):
        chain_data, decay_energies = self.generate_random_chain(initial_state)
        self.chain = [x[0] for x in chain_data]
        self.decay_energies = decay_energies

        state_id = lambda x: x.id if isinstance(x, State) else ''
        chain_string = [state_id(x[0]) + f' ==({x[1]})==> ' for x in chain_data]
        self.id = "".join(chain_string)

    def generate_random_chain(self, initial_state):
        """Generate a random decay path according to branching probabilities."""
        chain = [initial_state]
        decay_energies = []
        state = initial_state

        while not state.half_life == None: # run until a "stable" (half_life == None) state is found or SF is encountered
            # Randomly sample a decay branch based on the relative probabilities of each decay
            b = np.random.choice(state.branches.index.values, p=state.branches['probability'])

            chain[-1] = [chain[-1], b]

            if 'sf' in str(b):
                decay_energies.append(None)
                break

            else:
                excitation_energy = int(state.branches.loc[b]['excitation energy [keV]']) # fixing weird thingy where pandas turns int into float
                decay_energy = int(state.branches.loc[b]['energy [keV]']) # fixing weird thingy where pandas turns int into float

                decay_energies.append(decay_energy)

                if 'alpha' in str(b): # if alpha decay, find state A-4, Z-2 with corresponding excitation energy
                    state = State(state.A-4, state.Z-2, excitation_energy)

                elif 'gamma' in str(b):
                    state = State(state.A, state.Z, excitation_energy)

            chain.append(state)

        return chain, decay_energies

class ChainSimulation:
    """
        ChainSimulation simulates N number of decays from a given initial state. Simulations are done using Monte Carlo techniques.
        For each iteration, a new random path is determined based on branching ratios defined by the Chain object. From this,
        a cumulative distribution function (CDF) for each step is generated, and a random event time is generated, simulating
        a random radioactive decay that follows the original exponential distribution of any given state. The result is saved
        into a pandas DataFrame for further use.

        Arguments:
            initial_state               The initial state from which to simulate decay chains

        Attributes:
            run_simulation()            Function to start the simulation.
                                            int N (default 1000) - The number of decay chains to simulate
            results                     2D array of the results
            results_df                  The same results but formatted to a DataFrame
    """
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.all_states = initial_state.get_all_states()
        self.true_half_lives = initial_state.get_true_half_lives()
        self.result = None
        self.result_dfs = None

    def run_simulation(self, N=10_000, dist_time_range_factor=5):
        """
            Starts the Monte Carlo simulations and updates result attributes.

            Keyword arguments:
                int N (default 10_000)    The number of decay chains to simulate
        """
        chain_simulations = {}
        temp_dist_dict = {} # temporary dictionary to store CDF and half life, in order to avoid recalculations in the simulation for loop.

        for n in tqdm(range(N)):         
            chain_obj = Chain(self.initial_state)
            chain = chain_obj.chain
            chain_id = chain_obj.id

            event_times = [] # initialise empty list for generated event times

            for i in range(len(chain[:-1])):
                step = chain[i]
                last_step = chain[i-1]

                if step.half_life in temp_dist_dict.keys():
                    dist = temp_dist_dict[step.half_life] # If dist was already generated, load it from temporary dict
                else:
                    print(f'CDF for t₁/₂ = {step.half_life}s not found in temporary dictionary. Generating a new one...')
                    dist = Distribution(step.half_life, dist_time_range_factor*step.half_life) # generate new distribution for given half-life
                    temp_dist_dict[step.half_life] = dist # add newly generated distribution to dict

                event = Event(dist, parent=last_step, daughter=step)
                event_times.append(event.event_time)

            if chain_id in chain_simulations.keys():
                chain_simulations[chain_id].append(event_times)
            else:
                chain_simulations[chain_id] = [event_times]

            event_times.append('SF')
        result_dfs = {}

        for chain_sim in chain_simulations:
            col_names = chain_sim.split()
            col_names = [x for x in col_names if not '=' in x] # get column names for all decays (not including the final "stable" state)
            df = pd.DataFrame(chain_simulations[chain_sim], columns=col_names)
            result_dfs[chain_sim] = df

        self.result = chain_simulations
        self.result_dfs = result_dfs

        lifetimes = {}

        for (chain_id, df) in self.result_dfs.items():
            for column in df.columns[:-1]:
                try:
                    lifetimes[column] += df[column].to_numpy()
                except:
                    lifetimes[column] = df[column].to_numpy()

        mean_lifetimes = {k:np.mean(v) for (k, v) in lifetimes.items()}

        self.mean_lifetimes = pd.DataFrame(data={
                                          'Mean Lifetime [s]': mean_lifetimes.values(),
                                          '"True" Half-life [s]': self.true_half_lives[:-1]}, index=mean_lifetimes.keys())

    # getters

    def get_mean_lifetime(self, A, Z, E=0):
        """Returns a specific mean lifetime"""
        try:
            ret = self.mean_lifetimes.loc[f'{A}.{Z}.{E}']['Mean Lifetime [s]']
            return ret
        except:
            raise KeyError("State not found!")

    # printing functions

    def print_results(self):
        for i, k in enumerate(self.result_dfs.keys()):
            print(tc.BOLD+ f"Branch {i+1}: " + '\n' + tc.OKBLUE + k + tc.ENDC, '\n')
            print(self.result_dfs[k], '\n')

    def print_mean_lifetimes(self):
        print(tc.OKBLUE + tc.BOLD + "Mean Lifetime of states" + tc.ENDC)
        print(self.mean_lifetimes, '\n')

    def print_schmidt_test(self):
        for state in self.all_states[:-1]:
            print(tc.BOLD + tc.OKBLUE + f"Schmidt Test for {state}" + tc.ENDC)

            ls = []
            for df in self.result_dfs.values():
                try: lifetimes = df[state].to_list()
                except: lifetimes = []

                if lifetimes.__contains__('SF'):
                    pass
                else:
                    ls.append(lifetimes)

            arr = np.concatenate(ls)
            sigma_theta_exp, conf_int = schmidt_test(arr)
            lo = conf_int[0]
            hi = conf_int[1]

            if lo <= sigma_theta_exp <= hi:
                color = tc.OKGREEN
            else:
                color = tc.FAIL

            print('σ_θ: ' + color + str(round(sigma_theta_exp, 3)) + tc.ENDC,
                    f'[{round(lo, 3)}, {round(hi, 3)}]',
                    f'({arr.shape[0]} lifetimes)')
            print()
        
    def generalised_schmidt_test(self):
        print(tc.BOLD + tc.OKBLUE + "Generalised Schmidt Test" + tc.ENDC)
        for key in self.result_dfs.keys():
            df = self.result_dfs[key]
            print(key)
            sigma_theta_exp, conf_int = gst(df)
            lo = conf_int[0]
            hi = conf_int[1]

            if lo <= sigma_theta_exp <= hi:
                color = tc.OKGREEN
            else:
                color = tc.FAIL

            print('σ_θ: ' + color + str(round(sigma_theta_exp, 3)) + tc.ENDC,
                  f'[{round(lo, 3)}, {round(hi, 3)}]',
                  f'({df.shape[0]} chains)')
            print()