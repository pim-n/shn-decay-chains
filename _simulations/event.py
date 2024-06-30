import bisect
import random

class Event:
    def __init__(self, dist, parent=None, daughter=None):
        self.parent = parent
        self.daughter = daughter

        if not parent == daughter == None:
            self.name = f'{parent.name} => {daughter.name}'
        
        self.event_time = self.generate_event_time(dist.cdf, dist.time_range)
    
    def generate_event_time(self, cdf, time_range):
        d_bin = time_range[1] - time_range[0] # define discrete bin
        r = random.random()
        i = bisect.bisect_left(cdf, r) # from the left, find the nearest value to r in the CDF
            
        if i:
            pass
        else:
            i = 0

        t = time_range[i]
        t += random.random() * d_bin # this avoids issues with the discretization selection (for continuum quantities)

        return t