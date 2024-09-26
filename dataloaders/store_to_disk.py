import torch
import json
import os
import argparse

from networks.tokenizer import get_tokenizers_and_vocabs
from data_generation.text_interface.env import HMTEnv       # import from HandMeThat repository
from dataloaders.data_utils import parse_expert_demo, get_object_graph, get_simple_object_graph, \
    replace_object_token, get_action_effect, get_mid, Batch


def preprocess(json_str, src_tokenizer, env, file, mid_tokenizers=None):
    env.reset()

    object_dict = json_str['current_object_dict']
    subgoal = json_str['_subgoal']
    object_names = ['robot', 'human'] + list(object_dict.keys())

    # get problem solution
    demo_actions = json_str['demo_actions']
    object_dicts, cur_loc, holding, human_holding, outputs, choices, choices_in_text = \
        parse_expert_demo(object_dict, object_names, demo_actions, env)

    # get graph input
    object_graph_t = get_object_graph(object_names, src_tokenizer, object_dicts, cur_loc, holding, human_holding)
    # Since the dimension of graph input is too large for the model, we simplify the graph (V x V --> V)
    object_graph_t = get_simple_object_graph(object_graph_t, src_tokenizer)

    # get token inputs
    tmp_idx = json_str['task_description'].find(' Human stops')
    traj_t = src_tokenizer.sentence_tokenizer('[INT]' + json_str['task_description'][:tmp_idx])

    replace_indices, object_indices = replace_object_token(traj_t, object_names, src_tokenizer)

    tmp_idx = json_str['task_description'].find(' \'')
    inst_t = src_tokenizer.sentence_tokenizer('[INT]' + json_str['task_description'][tmp_idx:])

    effect_t = get_action_effect(json_str['demo_observations_fully'], src_tokenizer)    # list

    target_obj_index = outputs[-1][1]     # arg1 of the last action (give OBJ to human)

    if mid_tokenizers is not None:
        qsvo = get_mid(subgoal, mid_tokenizers)
    else:
        qsvo = (0, 0, 0, 0)

    subgoal = src_tokenizer.sentence_tokenizer(subgoal)

    new_data = list()
    for i, output in enumerate(outputs):
        solution_idx = 0
        for choice_idx, choice in enumerate(choices[i]):
            if choice == output:
                solution_idx = choice_idx
        new_data.append(
            {
                'json_idx': (file, i),
                'object_names': object_names,
                'graph': object_graph_t[i],
                'initial_graph': object_graph_t[0].copy(),
                'traj': traj_t,
                'inst': inst_t,
                'effect': effect_t[i],
                'output': output,
                'solution': solution_idx,
                'choices': choices[i],
                'choices_in_text': choices_in_text[i],
                'subgoal': subgoal,
                'qsvo': qsvo,
                'target_obj_index': target_obj_index,
                'replace_indices': replace_indices,
                'object_indices': object_indices,
            }
        )
    return new_data


def data2batch(data_piece):
    batch_size = 1
    json_indices = data_piece['json_idx']
    object_names = data_piece['object_names']

    # source input: physical state
    graphs = data_piece['graph']
    initial_graphs = data_piece['initial_graph']
    MAX_GRAPH = 5120    # 4320

    # source input: human trajectory
    trajectories = data_piece['traj']
    MAX_TRAJ = 512      # 450

    # source input: natural language instruction
    instructions = data_piece['inst']
    MAX_INST = 32       # 21

    # source input: effects of previous robot actions
    effects = data_piece['effect']
    MAX_EFFECT = 128    # 85

    # target output: action type
    actions = data_piece['output'][0]
    # action_choices = [[choices['action'] for choices in b['choices']] for b in batch]

    # target output: argument 1
    arg1s = data_piece['output'][1]
    # arg1_choices = [[choices['arg1'] for choices in b['choices']] for b in batch]

    # target output: argument 2
    arg2s = data_piece['output'][2]
    # arg2_choices = [[choices['arg2'] for choices in b['choices']] for b in batch]

    # target output: action triple
    solutions = data_piece['solution']
    choices = data_piece['choices']
    choices_in_text = data_piece['choices_in_text']

    # target middle output: subgoal
    subgoals = data_piece['subgoal']
    MAX_SUBGOAL = 128

    # target middle output: quantifier, subject, verb, object
    mid_q = data_piece['qsvo'][0]
    mid_s = data_piece['qsvo'][1]
    mid_v = data_piece['qsvo'][2]
    mid_o = data_piece['qsvo'][3]

    # target middle output: target object
    target_obj_indices = data_piece['target_obj_index']

    # replace name tokens in trajectory by object tokens
    replace_indices = torch.tensor(data_piece['replace_indices'], dtype=torch.long)
    object_indices = torch.tensor(data_piece['object_indices'], dtype=torch.long)
    graphs += [0] * (MAX_GRAPH - len(graphs))
    initial_graphs += [0] * (MAX_GRAPH - len(initial_graphs))
    trajectories += [0] * (MAX_TRAJ - len(trajectories))
    instructions += [0] * (MAX_INST - len(instructions))
    effects += [0] * (MAX_EFFECT - len(effects))
    subgoals += [0] * (MAX_SUBGOAL - len(subgoals))

    graphs = torch.tensor(graphs, dtype=torch.long)
    initial_graphs = torch.tensor(initial_graphs, dtype=torch.long)
    trajectories = torch.tensor(trajectories, dtype=torch.long)
    instructions = torch.tensor(instructions, dtype=torch.long)
    effects = torch.tensor(effects, dtype=torch.long)
    actions = torch.tensor(actions, dtype=torch.long)
    arg1s = torch.tensor(arg1s, dtype=torch.long)
    arg2s = torch.tensor(arg2s, dtype=torch.long)
    solutions = torch.tensor(solutions, dtype=torch.long)
    target_obj_indices = torch.tensor(target_obj_indices, dtype=torch.long)

    subgoals = torch.tensor(subgoals, dtype=torch.long)
    mid_q = torch.tensor(mid_q, dtype=torch.long)
    mid_s = torch.tensor(mid_s, dtype=torch.long)
    mid_v = torch.tensor(mid_v, dtype=torch.long)
    mid_o = torch.tensor(mid_o, dtype=torch.long)

    graph_mask = graphs[::16]
    mask0 = (graph_mask != 0).type_as(graph_mask)
    mask1 = (trajectories != 0).type_as(trajectories)
    mask2 = (instructions != 0).type_as(instructions)
    mask3 = (effects != 0).type_as(effects)

    tensor_batch = Batch()
    tensor_batch.size = batch_size
    tensor_batch.src = (graphs, trajectories, instructions, effects, initial_graphs)
    tensor_batch.tgt = (actions, arg1s, arg2s)
    tensor_batch.choices = choices                      # not tensor
    tensor_batch.choices_in_text = choices_in_text      # not tensor
    tensor_batch.json_indices = json_indices            # not tensor
    tensor_batch.object_names = object_names            # not tensor
    tensor_batch.subgoals = subgoals
    tensor_batch.qsvo = (mid_q, mid_s, mid_v, mid_o)
    tensor_batch.src_mask = (mask0, mask1, mask2, mask3)
    tensor_batch.solutions = solutions
    tensor_batch.target_obj_indices = target_obj_indices

    tensor_batch.replace_indices = replace_indices
    tensor_batch.object_indices = object_indices
    return tensor_batch


def process_and_store(data_dir, files, src_tokenizer, mid_tokenizers):
    for file_idx, file in enumerate(files):
        print(file_idx, file)
        with open(os.path.join(data_dir, 'HandMeThat_with_expert_demonstration', file), 'r') as f:
            json_str = json.load(f)
        env = HMTEnv(
            os.path.join(data_dir, 'HandMeThat_with_expert_demonstration', file),
            fully=True,
        )
        new_data = preprocess(json_str, src_tokenizer, env, file, mid_tokenizers)
        batch_list = [data2batch(data_piece) for data_piece in new_data]
        torch.save(batch_list, os.path.join(data_dir, 'preprocessed', file.replace('json', 'pt')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--goal', type=int)
    parser.add_argument('-d', '--data_dir', type=str)
    parser.add_argument('-w', '--working_dir', type=str)
    args = parser.parse_args()
    GOALS = [2, 3, 24, 25, 26, 30, 31, 32, 33, 35, 36, 46, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66]

    data_dir = args.data_dir
    if not os.path.exists(os.path.join(data_dir, 'preprocessed')):
        os.makedirs(os.path.join(data_dir, 'preprocessed'))

    with open(os.path.join(data_dir, 'HandMeThat_data_info.json')) as f0:
        file_json = json.load(f0)
    files = file_json[2][str(GOALS[args.goal])]
    src_tokenizer, _, _, _, mid_tokenizers, _ = get_tokenizers_and_vocabs(os.path.join(args.working_dir, 'vocabulary'))
    process_and_store(data_dir, files, src_tokenizer, mid_tokenizers)
