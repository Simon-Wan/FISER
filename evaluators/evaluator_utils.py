from dataloaders.data_utils import *


def preprocess_from_env(env, json_str, src_tokenizer, file, idx, effects, demo_action, current_object_dict, batch_size,
                        init_graph=None, holding_info=None, mid_tokenizers=None):
    c = copy.deepcopy
    object_dict = env.game.current_object_dict.copy()
    if '_subgoal' in json_str.keys():
        subgoal = json_str['_subgoal']
    else:
        subgoal = None

    object_names = ['robot', 'human'] + list(object_dict.keys())

    object_dicts, cur_loc, holding, human_holding, outputs, choices, choices_in_text = \
        parse_expert_demo(current_object_dict, object_names, [demo_action], env, True, holding_info)

    # get graph input
    object_graph_t = get_object_graph(object_names, src_tokenizer, object_dicts, cur_loc, holding, human_holding)
    object_graph_t = get_simple_object_graph(object_graph_t, src_tokenizer)

    if init_graph is None:
        init_object_graph_t = c(object_graph_t)
    else:
        init_object_graph_t = init_graph

    # get token inputs
    tmp_idx = json_str['task_description'].find(' Human stops')
    traj_t = src_tokenizer.sentence_tokenizer('[INT]' + json_str['task_description'][:tmp_idx])

    replace_indices, object_indices = replace_object_token(traj_t, object_names, src_tokenizer)

    tmp_idx = json_str['task_description'].find(' \'')
    inst_t = src_tokenizer.sentence_tokenizer('[INT]' + json_str['task_description'][tmp_idx:])

    effect_t = [src_tokenizer.sentence_tokenizer(effects[0])]   # change to online information

    if mid_tokenizers is not None and subgoal is not None:
        qsvo = get_mid(subgoal, mid_tokenizers)
    else:
        qsvo = (0, 0, 0, 0)

    if subgoal is not None:
        subgoal = src_tokenizer.sentence_tokenizer(subgoal)

    solution_idx = 0
    for choice_idx, choice in enumerate(choices[0]):
        if choice == outputs[0]:
            solution_idx = choice_idx

    new_data = [
        {
            'json_idx': (file, idx),
            'object_names': object_names,
            'graph': object_graph_t[0],
            'initial_graph': init_object_graph_t[0].copy(),
            'traj': traj_t,
            'inst': inst_t,
            'effect': effect_t[0],
            'output': (0, 0, 0),
            'solution': solution_idx,
            'choices': choices[0],
            'choices_in_text': choices_in_text[0],
            'subgoal': subgoal,
            'qsvo': qsvo,
            'target_obj_index': 0,
            'replace_indices': replace_indices,
            'object_indices': object_indices,
        }
    ] * batch_size
    return new_data, current_object_dict, (cur_loc[1], holding[1], human_holding[1])
