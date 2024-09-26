import torch
from dataloaders.data_utils import parse_expert_demo, get_object_graph, get_simple_object_graph, \
    replace_object_token, get_action_effect, get_mid, Batch


def preprocess(json_str, src_tokenizer, env, file, mid_tokenizers=None):
    env.reset()

    object_dict = json_str['current_object_dict']
    if '_subgoal' in json_str.keys():
        subgoal = json_str['_subgoal']
    else:
        subgoal = None
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

    if mid_tokenizers is not None and subgoal is not None:
        qsvo = get_mid(subgoal, mid_tokenizers)
    else:
        qsvo = (0, 0, 0, 0)

    if subgoal is not None:
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


def make_batch(batch, device):
    batch_size = len(batch)
    json_indices = [b['json_idx'] for b in batch]
    object_names = [b['object_names'] for b in batch]

    # source input: physical state
    graphs = [b['graph'] for b in batch]
    initial_graphs = [b['initial_graph'] for b in batch]
    # MAX_GRAPH = max([len(graph) for graph in graphs])
    MAX_GRAPH = 5120    # 4320
    graph_mask = list()

    # source input: human trajectory
    trajectories = [b['traj'] for b in batch]
    # MAX_TRAJ = max([len(traj) for traj in trajectories])
    MAX_TRAJ = 512      # 450

    # source input: natural language instruction
    instructions = [b['inst'] for b in batch]
    # MAX_INST = max([len(inst) for inst in instructions])
    MAX_INST = 32       # 21

    # source input: effects of previous robot actions
    effects = [b['effect'] for b in batch]
    # MAX_EFFECT = max([len(effect) for effect in effects])
    MAX_EFFECT = 128    # 85

    # target output: action type
    actions = [b['output'][0] for b in batch]
    # action_choices = [[choices['action'] for choices in b['choices']] for b in batch]

    # target output: argument 1
    arg1s = [b['output'][1] for b in batch]
    # arg1_choices = [[choices['arg1'] for choices in b['choices']] for b in batch]

    # target output: argument 2
    arg2s = [b['output'][2] for b in batch]
    # arg2_choices = [[choices['arg2'] for choices in b['choices']] for b in batch]

    # target output: action triple
    solutions = [b['solution'] for b in batch]
    choices = [b['choices'] for b in batch]
    choices_in_text = [b['choices_in_text'] for b in batch]

    # target middle output: subgoal
    subgoals = [b['subgoal'] for b in batch]
    if subgoals[0] is None:
        subgoals = [[0]] * len(subgoals)
    MAX_SUBGOAL = max([len(subgoal) for subgoal in subgoals])

    # target middle output: quantifier, subject, verb, object
    mid_q = [b['qsvo'][0] for b in batch]
    mid_s = [b['qsvo'][1] for b in batch]
    mid_v = [b['qsvo'][2] for b in batch]
    mid_o = [b['qsvo'][3] for b in batch]

    # target middle output: target object
    target_obj_indices = [b['target_obj_index'] for b in batch]

    # replace name tokens in trajectory by object tokens
    replace_indices = [torch.tensor(b['replace_indices'], dtype=torch.long).to(device) for b in batch]
    object_indices = [torch.tensor(b['object_indices'], dtype=torch.long).to(device) for b in batch]
    for i, graph in enumerate(graphs):
        graph_mask.append([1] * (len(graph) // 16) + [0] * ((MAX_GRAPH - len(graph)) // 16))
        graphs[i] += [0] * (MAX_GRAPH - len(graph))
    for i, init_graph in enumerate(initial_graphs):
        initial_graphs[i] += [0] * (MAX_GRAPH - len(init_graph))
    for i, traj in enumerate(trajectories):
        trajectories[i] += [0] * (MAX_TRAJ - len(traj))
    for i, inst in enumerate(instructions):
        instructions[i] += [0] * (MAX_INST - len(inst))
    for i, effect in enumerate(effects):
        effects[i] += [0] * (MAX_EFFECT - len(effect))
    for i, subgoal in enumerate(subgoals):
        subgoals[i] += [0] * (MAX_SUBGOAL - len(subgoal))

    graphs = torch.tensor(graphs, dtype=torch.long).to(device)
    initial_graphs = torch.tensor(initial_graphs, dtype=torch.long).to(device)
    trajectories = torch.tensor(trajectories, dtype=torch.long).to(device)
    instructions = torch.tensor(instructions, dtype=torch.long).to(device)
    effects = torch.tensor(effects, dtype=torch.long).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    arg1s = torch.tensor(arg1s, dtype=torch.long).to(device)
    arg2s = torch.tensor(arg2s, dtype=torch.long).to(device)
    solutions = torch.tensor(solutions, dtype=torch.long).to(device)
    target_obj_indices = torch.tensor(target_obj_indices, dtype=torch.long).to(device)

    subgoals = torch.tensor(subgoals, dtype=torch.long).to(device)
    mid_q = torch.tensor(mid_q, dtype=torch.long).to(device)
    mid_s = torch.tensor(mid_s, dtype=torch.long).to(device)
    mid_v = torch.tensor(mid_v, dtype=torch.long).to(device)
    mid_o = torch.tensor(mid_o, dtype=torch.long).to(device)

    mask0 = torch.tensor(graph_mask, dtype=torch.long).to(device)
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
