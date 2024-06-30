import pandas as pd
import yaml

class State:
    """
        Object to describe a unique quantum state using the nucleon count A, proton number Z and the excitation energy E.
        
        Arguments:
            A                   Nucleon count of the isotope
            Z                   Proton number of the isotope
            excitation_energy   The energy level occupied, relative to the ground state E=0

        Attributes:
            id                  The unique identifier of the state
            A                   Nucleon count of the isotope
            Z                   Proton number of the isotope
            E                   Excitation energy relative to ground state
            half_life           The half-life of the isotope
            branches            DataFrame object containing the available decay branches for this state
    """

    # Initialize state database as class-level attribute
    with open('state_db.yml', 'r') as file:
        DATABASE = yaml.safe_load(file)

    def __init__(self, A, Z, excitation_energy=0):
        state_id, db_state = self.find_state(A, Z, excitation_energy) # Find the state in the existing database

        if db_state == None:
            raise KeyError("State not found in database.")
        
        else:
            self.id = state_id
            self.A = A
            self.Z = Z
            self.E = excitation_energy
            self.half_life = db_state['half_life']
            self.name = db_state['name']

            if self.half_life == None: # if there is no half life, we have a "stable" state and don't need branches
                self.branches = None
            
            else:
                self.branches = self.unpack_branches(db_state)

    def get_all_states(self):
        return list(self.DATABASE.keys())

    def get_true_half_lives(self):
        return [self.DATABASE[x]['half_life'] for x in self.DATABASE.keys()]

    def find_state(self, A, Z, excitation_energy):
        try:
            state_id = f'{A}.{Z}.{excitation_energy}'
            return state_id, self.DATABASE[f'{A}.{Z}.{excitation_energy}']
        except KeyError:
            return None, None 

    def unpack_branches(self, db_state):
        """Takes a state from YML database and unpacks into a Pandas DataFrame"""

        cols = ['probability', 'energy [keV]', 'excitation energy [keV]']
        df = pd.DataFrame.from_dict(db_state['branches'], orient='index', columns=cols)
        return df