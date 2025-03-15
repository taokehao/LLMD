import ast
import typing
import json
from ase import Atoms

from Designer.llm import prompts
from NNUtils import structure



class PromptGenerator:
    def __init__(self):
        # self.request_template = prompts.REQUEST_TEMPLATE
        # self.reflection_template = prompts.REFLECTION_TEMPLATE
        self.modifications_list: typing.List[typing.List[dict]] = []
        self.properties_list: typing.List[tuple] = []

    @staticmethod
    def get_request(
        structure_text: str, 
        current_value: float,
        target_value: float,
        max_modifications=10
    ) -> str:
        
        if target_value <= 0:
            return prompts.REQUEST_TEMPLATE_LOWER.replace(
                'ATOMS_OF_THE_MATERIAL', structure_text
            ).replace(
                'CURRENT_VALUE', f'{current_value:.2f}'
            ).replace(
                'MAX_MODIFICATIONS', str(max_modifications)
            )
        else:
            return prompts.REQUEST_TEMPLATE.replace(
                'ATOMS_OF_THE_MATERIAL', structure_text
            ).replace(
                'CURRENT_VALUE', f'{current_value:.2f}'
            ).replace(
                'TARGET_VALUE', f'{target_value:.2f}'
            ).replace(
                'MAX_MODIFICATIONS', str(max_modifications)
            )

    @staticmethod
    def get_reflection(
        previous_structure: structure.Structure,
        new_structure: structure.Structure,
        answer: dict, target_value: float,
        previous_value: float, new_value: float
    ) -> str:
        return prompts.REFLECTION_TEMPLATE.replace(
            "PREVIOUS_FORMULA", previous_structure.structure.get_chemical_formula(mode='metal')
        ).replace(
            "NEW_FORMULA", new_structure.structure.get_chemical_formula(mode='metal')
        ).replace(
            "PREVIOUS_VALUE", f'{previous_value:.2f}'
        ).replace(
            "NEW_VALUE", f'{new_value:.2f}'
        ).replace(
            "REASON", answer['Reason']
        ).replace(
            "TARGET_VALUE", f'{target_value:.2f}'
        ).replace(
            "ATOMS_OF_THE_ORIGIN_MATERIAL", previous_structure.to_text()
        ).replace(
            "ATOMS_OF_THE_NEW_MATERIAL", new_structure.to_text()
        )

    @staticmethod
    def get_histories(
        answers_list: typing.List[dict],
        properties_list: typing.List[tuple],
        reflections_list: typing.List[str]
    ) -> str:
        assert len(answers_list) == len(properties_list) == len(reflections_list), \
            "All input lists must have same length"
        if len(answers_list) == 0:
            return ""
        histories = []
        for i, (a, p, r) in enumerate(zip(answers_list, properties_list, reflections_list)):
            one_history = prompts.ONE_HISTORY_TEMPLATE.replace(
                "MODIFICATIONS", json.dumps(a['Modification'])
            ).replace(
                "REASON", a['Reason']
            ).replace(
                "PREVIOUS_VALUE", f'{p[0]:.2f}'
            ).replace(
                "NEW_VALUE", f'{p[1]:.2f}'
            ).replace(
                "REFLECTION", r
            )
            histories.append(f"(History No.{i}){one_history}")
        return prompts.HISTORY_TEMPLATE.replace(
            "HISTORIES", "\n".join(histories)
        )



def decode_reply(reply: str) -> dict:
    ret = ast.literal_eval(reply)
    return ret




