from _simulations.chain_simulation import ChainSimulation
from _simulations.state import State

from _stats.schmidt_test import generalised_schmidt_test as gst

from _utils.termcolors import termcolors as tc

NUMBER_OF_SIMS = 1

# define a state
initial_state = State(A=288, Z=115)

# initialise the decay chain simulation
sim = ChainSimulation(initial_state=initial_state)

# run the simulation
sim.run_simulation(NUMBER_OF_SIMS)

# print all dataframes
sim.print_results()

# print mean lifetimes
sim.print_mean_lifetimes()        

# print statistics of individual steps
sim.print_schmidt_test()

# print statistics of all the chains (INACCURATE CONFIDENCE INTERVALS!!!)
sim.generalised_schmidt_test()

# get a specific lifetime
x = sim.get_mean_lifetime(A=288, Z=115)
print(x)