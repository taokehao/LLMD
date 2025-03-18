from GPTClient.client import Client
from NNUtils.structure import Structure
from Designer.workflow import LLMDAgent, Workflow



# gpt_client = Client(
#     api_key='sk-4od2iUvlA5xksbG761A2D1Fd5c014c7f9fC072002f21E100',
#     endpoint='https://tb.plus7.plus/v1/chat/completions',
#     history_thresh=0,
#     mask='Your are a professor in material science. You are going to help with optimizing a material with your experience in your field.'
# )

gpt_client = Client(
    api_key='app-YgM4Zw0JF8liojB5Rvb8njXo',
    endpoint='http://localhost/v1/chat-messages',
    history_thresh=0,
    # wait_time=60,
    mask='Your are a professor in material science. You are going to help with optimizing a material with your experience in your field.'
)

agent = LLMDAgent(
    llm_client=gpt_client,
    force_field_config_path='./checkpoints/force_field/config.yml',
    device='cuda',
    external_model_path='./external_models/',
    workdir='./output'
)

structure = Structure().read_cif('./Li7La3Zr2O12.cif')


workflow = Workflow(
    agent=agent, input_structure=structure,
    target_value=0.4, maximum_iterations=30,
    maximum_error=0.1,
    log_dir='./output', log_file_name='output.log'
)


print(agent.calculate(structure=structure, save_cif_name=None))

workflow.run()
