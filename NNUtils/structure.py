
import typing
from time import time

import ase
from ase import io
from ase import Atoms
from ase.optimize import FIRE
from ase.constraints import ExpCellFilter
 
from matdeeplearn.common.ase_utils import MDLCalculator

# logging.basicConfig(level=logging.INFO)


class StructureOptimizer:
    """
    This class provides functionality to optimize the structure of an Atoms object using a specified calculator.
    """

    def __init__(self,
                 calculator: MDLCalculator,
                 relax_cell: bool = False,
                 ):
        """
        Initialize the StructureOptimizer.

        Parameters:
        - calculator (Calculator): A calculator object for performing energy and force calculations.
        - relax_cell (bool): If True, the cell will be relaxed in addition to positions during the optimization process.
        """
        self.calculator = calculator
        self.relax_cell = relax_cell

    def optimize(self, atoms: Atoms, logfile=None, write_traj_name=None) -> typing.Tuple[Atoms, float]:
        """
        This method optimizes the structure of the given Atoms object using the specified calculator.
        If `relax_cell` is True, the cell will be relaxed. Trajectory information can be written to a file.

        Parameters:
        - atoms: An Atoms object representing the structure to be optimized.
        - logfile: File to write optimization log information.
        - write_traj_name: File to write trajectory of the optimization.

        Returns:
        - atoms: The optimized Atoms object.
        - time_per_step: The average time taken per optimization step.
        """
        atoms.calc = self.calculator
        if self.relax_cell:
            atoms = ExpCellFilter(atoms)

        optimizer = FIRE(atoms, logfile=logfile)

        if write_traj_name is not None:
            traj = io.trajectory.Trajectory(write_traj_name + '.traj', 'w', atoms)
            optimizer.attach(traj.write, interval=1)

        start_time = time()
        optimizer.run(fmax=0.001, steps=500)
        end_time = time()
        # num_steps = optimizer.get_number_of_steps()

        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
        time_used = end_time - start_time
        return atoms, time_used




class Structure:
    def __init__(self):
        self.path = None
        self.structure: typing.Union[ase.Atoms, None] = None

    def read_poscar(self, path: str):
        self.path = path
        self.structure = io.read(path, format='vasp')
        return self

    def read_cif(self, path: str):
        self.path = path
        self.structure = io.read(path, format='cif')
        return self

    def write_poscar(self, poscar_path: str):
        io.write(poscar_path, self.structure, format='vasp')

    def write_cif(self, cif_path: str):
        io.write(cif_path, self.structure, format='cif')

    def to_text(self):
        if self.structure is None:
            raise RuntimeError('Structure cannot be None')
        lines = []
        for atom in self.structure:
            coords = ','.join([f'{i:.3f}' for i in atom.position])
            lines.append(f'{atom.symbol} | {coords}')
        return '\n'.join(lines)

    def gen_element_list(self) -> typing.List[str]:
        elements = self.structure.get_chemical_symbols()
        elements_set = []
        elements_num = {}
        unique_elements_list = []
        for element in elements:
            if element not in elements_set:
                elements_set.append(element)
                elements_num[element] = 1
            else:
                elements_num[element] += 1
            unique_elements_list.append(f'{element}.{elements_num[element]}')
        return unique_elements_list

    def exchange(self, a: int, b: int) -> Atoms:
        tmp = self.structure[a].symbol
        self.structure[b].symbol = tmp
        self.structure[a].symbol = tmp
        return self.structure


    def substitute(self, index: int, element: str) -> Atoms:
        self.structure[index].symbol = element
        return self.structure

    # def assign_element(self, elements: typing.List[str]) -> Atoms:
    #     assert len(elements) == len(self.structure), "Number of elements does not match number of atoms"
    #     for i in range(len(elements)):
    #         self.structure[i].symbol = elements[i]
    #     return self.structure


