import numpy as np
import yaml
import logging
import typing
from ase.atoms import Atoms
from ase.geometry import Cell
from ase.calculators.calculator import Calculator
import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.utils import add_self_loops, degree, dense_to_sparse
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader



import importlib


def _get_absolute_mapping(name: str):
    # in this case, the `name` should be the fully qualified name of the class
    # e.g., `matdeeplearn.tasks.base_task.BaseTask`
    # we can use importlib to get the module (e.g., `matdeeplearn.tasks.base_task`)
    # and then import the class (e.g., `BaseTask`)

    module_name = ".".join(name.split(".")[:-1])
    class_name = name.split(".")[-1]

    try:
        module = importlib.import_module(module_name)
    except (ModuleNotFoundError, ValueError) as e:
        raise RuntimeError(
            f"Could not import module {module_name=} for import {name=}"
        ) from e

    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise RuntimeError(
            f"Could not import class {class_name=} from module {module_name=}"
        ) from e


class Registry:
    r"""Class for registry object which acts as central source of truth."""
    mapping = {
        # Mappings to respective classes.
        "task_name_mapping": {},
        "dataset_name_mapping": {},
        "model_name_mapping": {},
        "logger_name_mapping": {},
        "trainer_name_mapping": {},
        "loss_name_mapping": {},
        "state": {},
        "transforms": {},
    }

    @classmethod
    def register_task(cls, name):
        r"""Register a new task to registry with key 'name'
        Args:
            name: Key with which the task will be registered.
        Usage::
            from matdeeplearn.common.registry import registry
            from matdeeplearn.tasks import BaseTask
            @registry.register_task("train")
            class TrainTask(BaseTask):
                ...
        """

        def wrap(func):
            cls.mapping["task_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_dataset(cls, name):
        r"""Register a dataset to registry with key 'name'

        Args:
            name: Key with which the dataset will be registered.

        Usage::

            from matdeeplearn.common.registry import registry
            from matdeeplearn.datasets import BaseDataset

            @registry.register_dataset("qm9")
            class QM9(BaseDataset):
                ...
        """

        def wrap(func):
            cls.mapping["dataset_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls, name):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the model will be registered.

        Usage::

            from matdeeplearn.common.registry import registry
            from matdeeplearn.modules.layers import CGCNNConv

            @registry.register_model("cgcnn")
            class CGCNN():
                ...
        """

        def wrap(func):
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    # @classmethod
    # def register_logger(cls, name):
    #     r"""Register a logger to registry with key 'name'
    #
    #     Args:
    #         name: Key with which the logger will be registered.
    #
    #     Usage::
    #
    #         from matdeeplearn.common.registry import registry
    #
    #         @registry.register_logger("tensorboard")
    #         class WandB():
    #             ...
    #     """
    #
    #     def wrap(func):
    #         from matdeeplearn.common.logger import Logger
    #
    #         assert issubclass(func, Logger), "All loggers must inherit Logger class"
    #         cls.mapping["logger_name_mapping"][name] = func
    #         return func
    #
    #     return wrap

    # @classmethod
    # def register_trainer(cls, name):
    #     r"""Register a trainer to registry with key 'name'
    #
    #     Args:
    #         name: Key with which the trainer will be registered.
    #
    #     Usage::
    #
    #         from matdeeplearn.common.registry import registry
    #
    #         @registry.register_trainer("active_discovery")
    #         class ActiveDiscoveryTrainer():
    #             ...
    #     """
    #
    #     def wrap(func):
    #         cls.mapping["trainer_name_mapping"][name] = func
    #         return func
    #
    #     return wrap

    # @classmethod
    # def register_loss(cls, name):
    #     r"""Register a loss class to registry with key 'name'
    #
    #     Args:
    #         name: Key with which the trainer will be registered.
    #
    #     Usage::
    #
    #         from matdeeplearn.common.registry import registry
    #
    #         @registry.register_loss("dos_loss")
    #         class DOSLoss():
    #             ...
    #     """
    #
    #     def wrap(func):
    #         cls.mapping["loss_name_mapping"][name] = func
    #         return func
    #
    #     return wrap

    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from matdeeplearn.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    # @classmethod
    # def register_transform(cls, transform_name: str):
    #     """Registers a transform function for bookkeeping."""
    #
    #     def wrap_func(transform: Callable):
    #         cls.mapping["transforms"][transform_name] = transform
    #         return transform
    #
    #     return wrap_func
    #
    @classmethod
    def __import_error(cls, name: str, mapping_name: str):
        kind = mapping_name[: -len("_name_mapping")]
        mapping = cls.mapping.get(mapping_name, {})
        existing_keys = list(mapping.keys())

        existing_cls_path = (
            mapping.get(existing_keys[-1], None) if existing_keys else None
        )
        if existing_cls_path is not None:
            existing_cls_path = (
                f"{existing_cls_path.__module__}.{existing_cls_path.__qualname__}"
            )
        else:
            existing_cls_path = "matdeeplearn.trainers.PropertyTrainer"

        existing_keys = [f"'{name}'" for name in existing_keys]
        existing_keys = ", ".join(existing_keys[:-1]) + " or " + existing_keys[-1]
        existing_keys_str = f" (one of {existing_keys})" if existing_keys else ""
        return RuntimeError(
            f"Failed to find the {kind} '{name}'. "
            f"You may either use a {kind} from the registry{existing_keys_str} "
            f"or provide the full import path to the {kind} (e.g., '{existing_cls_path}')."
        )

    @classmethod
    def get_class(cls, name: str, mapping_name: str):
        existing_mapping = cls.mapping[mapping_name].get(name, None)
        if existing_mapping is not None:
            return existing_mapping

        # mapping be class path of type `{module_name}.{class_name}` (e.g., `matdeeplearn.trainers.PropertyTrainer`)
        if name.count(".") < 1:
            raise cls.__import_error(name, mapping_name)

        try:
            return _get_absolute_mapping(name)
        except RuntimeError as e:
            raise cls.__import_error(name, mapping_name) from e

    @classmethod
    def get_task_class(cls, name):
        return cls.get_class(name, "task_name_mapping")

    @classmethod
    def get_dataset_class(cls, name):
        return cls.get_class(name, "dataset_name_mapping")

    @classmethod
    def get_model_class(cls, name):
        return cls.get_class(name, "model_name_mapping")

    @classmethod
    def get_logger_class(cls, name):
        return cls.get_class(name, "logger_name_mapping")

    @classmethod
    def get_trainer_class(cls, name):
        return cls.get_class(name, "trainer_name_mapping")

    @classmethod
    def get_loss_class(cls, name):
        return cls.get_class(name, "loss_name_mapping")

    @classmethod
    def get_transform_class(cls, name, **kwargs):
        return cls.get_class(name, "transforms")(**kwargs)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retreived.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for cgcnn's
                               internal operations. Default: False
        Usage::

            from matdeeplearn.common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].write(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from matdeeplearn.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()





def one_hot_degree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = degree(idx, data.num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else:
        data.x = deg

    return data



def node_rep_one_hot(Z):
    return F.one_hot(Z - 1, num_classes = 100)


def generate_node_features(input_data, n_neighbors, device, use_degree=False, node_rep_func = node_rep_one_hot):
    if isinstance(input_data, Data):
        input_data.x = node_rep_func(input_data.z)
        if use_degree:
            return one_hot_degree(input_data, n_neighbors)
        return input_data
    for i, data in enumerate(input_data):
        # minus 1 as the reps are 0-indexed but atomic number starts from 1
        data.x = node_rep_func(data.z).float()






class MDLCalculator(Calculator):
    """
    A neural networked based Calculator that calculates the energy, forces and stress of a crystal structure.
    """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, config, rank='cuda'):
        """
        Initialize the MDLCalculator instance.

        Args:
        - config (str or dict): Configuration settings for the MDLCalculator.
        - rank (str): Rank of device the calculator calculates properties. Defaults to 'cuda:0'

        Raises:
        - AssertionError: If the trainer name is not in the correct format or if the trainer class is not found.
        """
        Calculator.__init__(self)

        if isinstance(config, str):
            logging.info(f'MDLCalculator instantiated from config: {config}')
            with open(config, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)
        elif isinstance(config, dict):
            logging.info('MDLCalculator instantiated from a dictionary.')
        else:
            raise NotImplementedError('Unsupported config type.')

        gradient = config["model"].get("gradient", False)
        otf_edge_index = config["model"].get("otf_edge_index", False)
        otf_edge_attr = config["model"].get("otf_edge_attr", False)
        self.otf_node_attr = config["model"].get("otf_node_attr", False)
        assert otf_edge_index and otf_edge_attr and gradient, "To use this calculator to calculate forces and stress, you should set otf_edge_index, oft_edge_attr and gradient to True."

        self.device = rank if torch.cuda.is_available() else 'cpu'
        self.models = MDLCalculator._load_model(config, self.device)
        self.n_neighbors = config['dataset']['preprocess_params'].get('n_neighbors', 250)

        # self.calculated = False

    def calculate(self, atoms: Atoms, properties=implemented_properties, system_changes=None) -> None:
        """
        Calculate energy, forces, and stress for a given ase.Atoms object.

        Args:
        - atoms (ase.Atoms): The atomic structure for which calculations are to be performed.
        - properties (list): List of properties to calculate. Defaults to ['energy', 'forces', 'stress'].
        - system_changes: Not supported in the current implementation.

        Returns:
        - None: The results are stored in the instance variable 'self.results'.

        Note:
        - This method performs energy, forces, and stress calculations using a neural network-based calculator.
            The results are stored in the instance variable 'self.results' as 'energy', 'forces', and 'stress'.
        """
        Calculator.calculate(self, atoms, properties, system_changes)

        cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        atomic_numbers = torch.LongTensor(atoms.get_atomic_numbers())

        data = Data(n_atoms=len(atomic_numbers), pos=pos, cell=cell.unsqueeze(dim=0),
                    z=atomic_numbers, structure_id=atoms.info.get('structure_id', None))

        # Generate node features
        if not self.otf_node_attr:
            generate_node_features(data, self.n_neighbors, device=self.device)
            data.x = data.x.to(torch.float32)

        data_list = [data]
        loader = DataLoader(data_list, batch_size=1)
        loader_iter = iter(loader)
        batch = next(loader_iter).to(self.device)

        out_list = []
        for model in self.models:
            out_list.append(model(batch))

        energy = torch.stack([entry["output"] for entry in out_list]).mean(dim=0)
        forces = torch.stack([entry["pos_grad"] for entry in out_list]).mean(dim=0)
        stresses = torch.stack([entry["cell_grad"] for entry in out_list]).mean(dim=0)

        self.results['energy'] = energy.detach().cpu().numpy().squeeze()
        self.results['forces'] = forces.detach().cpu().numpy().squeeze()
        self.results['stress'] = stresses.squeeze().detach().cpu().numpy().squeeze()

    @staticmethod
    def data_to_atoms_list(data: Data) -> typing.List[Atoms]:
        """
        This helper method takes a 'torch_geometric.data.Data' object containing information about atomic structures
        and converts it into a list of 'ase.Atoms' objects. Each 'Atoms' object represents an atomic structure
        with its associated properties such as positions and cell.

        Args:
        - data (Data): A data object containing information about atomic structures.

        Returns:
        - List[Atoms]: A list of 'ase.Atoms' objects, each representing an atomic structure
            with positions and associated properties.
        """
        cells = data.cell.numpy()

        split_indices = np.cumsum(data.n_atoms)[:-1]
        positions_per_structure = np.split(data.pos.numpy(), split_indices)
        symbols_per_structure = np.split(data.z.numpy(), split_indices)

        atoms_list = [Atoms(
            symbols=symbols_per_structure[i],
            positions=positions_per_structure[i],
            cell=Cell(cells[i])) for i in range(len(data.structure_id))]
        for i in range(len(data.structure_id)):
            atoms_list[i].structure_id = data.structure_id[i][0]
        return atoms_list

    @staticmethod
    def _load_model(config: dict, rank: str) -> list:
        """
        This static method loads a model based on the provided configuration.

        Parameters:
        - config (dict): Configuration dictionary containing model and dataset parameters.
        - rank: Rank information for distributed training.

        Returns:
        - model_list: A list of loaded models.
        """

        graph_config = config['dataset']['preprocess_params']
        model_config = config['model']

        model_list = []
        model_name = model_config["name"]
        logging.info(f'MDLCalculator: setting up {model_name} for calculation')
        # Obtain node, edge, and output dimensions for model initialization
        for _ in range(model_config["model_ensemble"]):
            node_dim = graph_config["node_dim"]
            edge_dim = graph_config["edge_dim"]

            model_cls = registry.get_model_class(model_name)
            model = model_cls(
                node_dim=node_dim,
                edge_dim=edge_dim,
                output_dim=1,
                cutoff_radius=graph_config["cutoff_radius"],
                n_neighbors=graph_config["n_neighbors"],
                graph_method=graph_config["edge_calc_method"],
                num_offsets=graph_config["num_offsets"],
                **model_config
            )
            model = model.to(rank)
            model_list.append(model)

        checkpoints = config['task']["checkpoint_path"].split(',')
        if len(checkpoints) == 0:
            logging.warning(
                "MDLCalculator: No checkpoint.pt file is found, and untrained models are used for prediction.")
        else:
            for i in range(len(checkpoints)):
                try:
                    if torch.cuda.is_available():
                        checkpoint = torch.load(checkpoints[i])
                    else:
                        checkpoint = torch.load(checkpoints[i], map_location='cpu')
                    model_list[i].load_state_dict(checkpoint["state_dict"])
                    logging.info(f'MDLCalculator: weights for model No.{i + 1} loaded from {checkpoints[i]}')
                except ValueError:
                    logging.warning(
                        f"MDLCalculator: No checkpoint.pt file is found for model No.{i + 1}, and an untrained model is used for prediction.")

        return model_list



