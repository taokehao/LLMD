import os
import subprocess
import re
import copy
import typing
import warnings
import json

import ast

import torch

from GPTClient.client import Client as GPTClient

from Designer import llm

# from NNUtils import migrate
from matdeeplearn.common.ase_utils import MDLCalculator
from NNUtils import structure as nn_struct

# mattersim
import os
from mattersim.forcefield.potential import MatterSimCalculator
from mattersim.applications.relax import Relaxer
from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential
from ase.io import read, write

# external model
from external_models.Predict import predict



class LLMDAgent:
    def __init__(
        self,
        llm_client: GPTClient,
        force_field_config_path: str=None,
        # band_gap_config_path: str=None,
        # formation_energy_config_path: str=None,
        device: str=None,
        external_model_path: str=None,
        workdir: str=None, save_all_cifs: bool=False,
    ):
        self.llm_client = llm_client
        # self.structure = nn_struct.Structure()



        # self.property_keys = ("force_field", "band_gap", "formation_energy")
        self.property_keys = ("force_field",)

        self.configs = {
            "force_field": force_field_config_path,
            # "band_gap": band_gap_config_path,
            # "formation_energy": formation_energy_config_path
        }

        self.calculators = {
            "force_field": None,
            # "band_gap": None,
            # "formation_energy": None
        }
        self.structure_optimizer = None

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        for k in self.property_keys:
            if self.configs[k] is not None:
                self.calculators[k] = MDLCalculator(self.configs[k], rank=self.device)

        if self.calculators["force_field"] is not None:
            self.structure_optimizer = nn_struct.StructureOptimizer(
                self.calculators["force_field"]
            )

        if external_model_path is None:
            raise ValueError("external_model_path is needed.")
        else:
            if not os.path.exists(external_model_path):
                raise ValueError("external_model_path does not exist.")

        self.external_model_path = os.path.abspath(external_model_path)
        if workdir is None:
            if os.path.exists('/tmp'):
                self.workdir = '/tmp'
            else:
                self.workdir = './tmp'
        else:
            self.workdir = workdir

        self.save_all_cifs = save_all_cifs

        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)


    def ask(self, prompt: str) -> str:
        answer = self.llm_client.ask(prompt)
        # print(answer)
        pattern = r'```python\n(.*?)\n```'
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return answer

    def optimize(self, structure: nn_struct.Structure, verbose=True) -> nn_struct.Structure:
        assert self.calculators["force_field"] is not None, \
            "force_field calculator has not been set."
        new_structure = copy.deepcopy(structure)
        atoms = new_structure.structure
        if len(atoms) == 1:
            optimized_atoms = atoms
        else:
            # atoms.set_calculator(self.calculators['force_field'])
            # optimized_atoms, optimized_time = self.structure_optimizer.optimize(atoms)
            # if verbose:
            #     print(f"Optimization structures took {optimized_time:.2f} seconds.")

            atoms.rattle(stdev=0.1)
            # 初始化MatterSim势函数
            potential = Potential.from_checkpoint()
            # 创建批处理松弛器配置（保持与原示例一致）
            relaxer = BatchRelaxer(potential,
                                   fmax=0.01,  # 收敛判据（力最大值）
                                   filter="EXPCELLFILTER",  # 晶胞过滤器
                                   optimizer="BFGS")  # 优化算法
            initial_structures = [atoms]
            optimized_atoms = relaxer.relax(initial_structures)[0][-1]

            # attach the calculator to the atoms object
            # atoms.calc = MatterSimCalculator()
            # # initialize the relaxation object
            # relaxer = Relaxer(
            #     optimizer="FIRE",  # the optimization method
            #     filter="ExpCellFilter",  # filter to apply to the cell
            #     constrain_symmetry=True,  # whether to constrain the symmetry
            # )
            # optimized_atoms = relaxer.relax(atoms, steps=500, fmax=0.01)

        new_structure.structure = optimized_atoms
        return new_structure

    def calculate(self, structure: nn_struct.Structure, save_cif_name: str=None) -> float:
        if save_cif_name is None:
            cif_path = os.path.abspath(os.path.join(self.workdir, "__tmp__.cif"))
        else:
            cif_path = os.path.abspath(
                os.path.join(
                    self.workdir,
                    f'{save_cif_name}.cif' if not save_cif_name.endswith('.cif') else save_cif_name
                )
            )
        structure.write_cif(cif_path)

        # run_script = os.path.join(
        #     self.external_model_path, 'run.sh'
        # )
        # process = subprocess.run(
        #     ["bash", run_script, cif_path], capture_output=True, text=True
        # )
        # if save_cif_name is None:
        #     os.remove(cif_path)
        # if process.returncode != 0:
        #     print(process.stdout)
        #     raise RuntimeError("Running External Model using container returns an non-zero value.")
        # ret = process.stdout.strip()
        # ret = float(ret)
        ret = predict(cif_path)
        return ret


    def perform_modification(
        self, structure: nn_struct.Structure,
        modifications: typing.List[dict],
        verbose=True
    ):
        new_structure = copy.deepcopy(structure)
        for m in modifications:
            if m['type'] not in ['exchange', 'substitute']:
                raise NotImplementedError
            else:
                new_structure.__getattribute__(m['type'])(*m['action'])
        new_structure = self.optimize(new_structure, verbose=verbose)
        return new_structure



class Workflow:
    def __init__(
        self, agent: LLMDAgent,
        input_structure: nn_struct.Structure,
        target_value: float,
        maximum_iterations: int=50,
        maximum_error: float=0.1,
        log_dir: str=None, log_file_name: str=None
    ):
        self.agent = agent
        if self.agent.llm_client.history.capacity != 0:
            warnings.warn("History recording enabled for llm client.")
        self.target_value = target_value
        self.maximum_iterations = maximum_iterations
        self.maximum_error = maximum_error

        self.structure = input_structure
        self.value = self.agent.calculate(self.structure, save_cif_name=None)

        self.properties_list: typing.List[tuple] = []
        self.answers_list: typing.List[dict] = []
        self.reflections_list: typing.List[str] = []

        self.log_dir = log_dir
        self.log_file_name = log_file_name
        if self.log_file_name is None and self.log_dir is not None:
            self.log_file_name = 'output.log'

        if self.log_dir is not None:
            if not os.path.exists(self.log_dir):
                raise ValueError("log_dir does not exist.")
            with open(os.path.join(self.log_dir, self.log_file_name), 'w') as f:
                pass


    def __write_log__(self, content: str, tee: bool=True):
        if tee:
            print(content)
        if self.log_dir is not None:
            with open(os.path.join(self.log_dir, self.log_file_name), "a", encoding="utf-8") as f:
                if content.endswith("\n"):
                    f.write(content)
                else:
                    f.write(f"{content}\n")
            return True
        else:
            return False


    def one_step(
        self, step: int=0, tee=True
    ) -> typing.Tuple[nn_struct.Structure, float, dict, str]:
        request_text = llm.utils.PromptGenerator.get_request(
            structure_text=self.structure.to_text(),
            current_value=self.value,
            target_value=self.target_value,
            max_modifications=15
        )

        histories_text = llm.utils.PromptGenerator.get_histories(
            answers_list=self.answers_list,
            properties_list=self.properties_list,
            reflections_list=self.reflections_list
        )
        prompt = f'{request_text}\n{histories_text}'
        answer = self.agent.ask(prompt=prompt)

        self.__write_log__("PROMPT:", tee=tee)
        self.__write_log__(prompt, tee=tee)
        self.__write_log__("\n", tee=tee)
        self.__write_log__("ANSWER:", tee=tee)
        self.__write_log__(answer, tee=tee)
        self.__write_log__("\n", tee=tee)

        answer_dict = ast.literal_eval(answer)
        new_structure = self.agent.perform_modification(
            structure=self.structure,
            modifications=answer_dict['Modification'],
            verbose=True
        )
        # if self.agent.save_all_cifs:
        #     save_cif_name = f'step-{step}.cif'
        # else:
        #     save_cif_name = None
        # new_value = self.agent.calculate(new_structure, save_cif_name=save_cif_name)
        new_value = self.agent.calculate(new_structure)

        reflection = self.agent.llm_client.ask(
            llm.utils.PromptGenerator.get_reflection(
                previous_structure=self.structure,
                new_structure=new_structure,
                previous_value=self.value,
                new_value=new_value,
                target_value=self.target_value,
                answer=answer_dict,
            )
        )
        return new_structure, new_value, answer_dict, reflection


    def run(
        self, max_iterations: typing.Union[int, None]=None, max_err=None,
        tee=True
    ):
        if max_iterations is None:
            max_iterations = self.maximum_iterations
        if max_err is None:
            max_err = self.maximum_error

        for i in range(max_iterations):
            self.__write_log__(f"-----> Iteration {i:02d} <-----", tee=tee)
            self.__write_log__(f"\n", tee=tee)
            new_structure, new_value, answer_dict, reflection = self.one_step(step=i)
            self.__write_log__("Modifications:", tee=tee)
            self.__write_log__("\n", tee=tee)
            for m in answer_dict["Modification"]:
                self.__write_log__(json.dumps(m), tee=tee)
            self.__write_log__("RESULT:\n\n", tee=tee)
            formula = new_structure.structure.get_chemical_formula(mode='hill')
            new_structure.write_cif(os.path.join(self.log_dir, f'{i}-{formula}.cif'))
            self.__write_log__(f"new formula = {formula}\n", tee=tee)
            self.__write_log__(f"current value = {new_value:.2f}, target value = {self.target_value:.2f}\n\n",
                               tee=tee)

            self.answers_list.append(answer_dict)
            self.properties_list.append((self.value, new_value))
            self.reflections_list.append(reflection)

            self.value = new_value
            self.structure = new_structure

            if abs(self.value - self.target_value) <= max_err:
                self.__write_log__(f"Target achieved at current_value={self.value:.2f} target={self.target_value:.2f}.",
                                   tee=tee)
                break
