from automan.api import Problem, Automator, Simulation
import matplotlib.pyplot as plt 
import numpy as np 
from automan.api import mdict

class Wedge(Problem):
    def get_name(self):
        return 'wedge_numba_automat'

    def setup(self):
        opts = mdict(nx=[20, 30, 40], L1=[1, 2, 3])
        base_cmd = 'python wedge_numba_automat.py --output-dir $output_dir'
        self.cases = [
            Simulation(
                root=self.input_path(str(i)),
                base_command=base_cmd,
                nx = opts[i]['nx'], 
                L1= opts[i]['L1'] 
            )
            for i in range(9)
        ]

    def run(self):
        self.make_output_dir()


if __name__ == '__main__':
    automator = Automator(
        simulation_dir='outputs',
        output_dir='manuscript/figures',
        all_problems=[Wedge]
    )

    automator.run() 