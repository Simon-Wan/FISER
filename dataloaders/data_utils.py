import copy
from networks.tokenizer import get_tokenizers_and_vocabs
from trainers.trainer_utils import translate_subgoal


class Batch:
    def __init__(self):
        self.size = None                # batch size
        self.json_indices = None        # filenames
        self.src = None                 # source inputs
        self.src_mask = None            # masks for source inputs
        self.tgt = None                 # target outputs
        self.object_names = None        # list of object names
        self.choices = None             # applicable operators
        self.choices_in_text = None     # valid actions in text
        self.solutions = None           # target action idx within choices
        self.subgoals = None            # subgoal (FOL)
        self.qsvo = None                 # middle prediction (QSVO)
        self.replace_indices = None     # language token positions to be replaced
        self.object_indices = None      # object token positions used to replace
        self.target_obj_indices = None  # target object to interact


def parse_expert_demo(object_dict, object_names, demo, env, eval_use=False, holding_info=None):
    """
    Parse expert demonstration and return episode information
    :param object_dict: dictionary of objects
    :param object_names: list of object names
    :param demo: list of demo actions
    :param env: enviornment
    :param eval_use: whether the function is only for evaluation (do not change env)
    :param holding_info: current state of robot & human
    :return object_dicts: list of object_dict form 0 to T
    :return cur_loc: current location of robot
    :return holding: current holding of robot
    :return human_holding: current holding of human
    :return solutions: list of action arguments from 0 to T-1
    :return choices: list of applicable operators from 0 to T-1
    :return choices_in_text: list of valid actions in text from 0 to T-1
    """
    c = copy.deepcopy
    solutions = list()

    if holding_info is not None:
        cur_loc = [holding_info[0]]
        holding = [holding_info[1]]
        human_holding = [holding_info[2]]
    else:
        cur_loc = ['floor']
        holding = [None]
        human_holding = [list()]

    object_dicts = [c(object_dict)]
    choices = list()
    choices_in_text = list()

    for action in demo:

        valid_actions = env.get_valid_actions()
        valid_actions.sort()
        choices_in_text.append(valid_actions)
        applicable_operators = list()
        for op in valid_actions:
            applicable_operators.append(parse_action(op, object_names))
        arg2_idx = 0
        cur_loc.append(cur_loc[-1])
        holding.append(holding[-1])
        human_holding.append(human_holding[-1].copy())
        if action[:7] == 'move to':
            act_idx = 4
            arg1 = action[8:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            cur_loc[-1] = arg1

        elif action[:7] == 'pick up' and ' from ' not in action:
            act_idx = 5
            arg1 = action[8:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            if 'inside' in object_dict[arg1].keys():
                object_dict[arg1].pop('inside')
            elif 'ontop' in object_dict[arg1].keys():
                object_dict[arg1].pop('ontop')
            holding[-1] = arg1

        elif action[:7] == 'pick up' and ' from ' in action:
            mid_idx = action.find(' from ')
            act_idx = 6
            arg1 = action[8:mid_idx]
            arg2 = action[mid_idx+6:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            arg2, arg2_idx = get_argument_index(arg2, object_names)
            if 'inside' in object_dict[arg1].keys():
                object_dict[arg1].pop('inside')
            elif 'ontop' in object_dict[arg1].keys():
                object_dict[arg1].pop('ontop')
            holding[-1] = arg1

        elif action[:3] == 'put' and ' into ' in action:
            mid_idx = action.find(' into ')
            act_idx = 7
            arg1 = action[4:mid_idx]
            arg2 = action[mid_idx+6:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            arg2, arg2_idx = get_argument_index(arg2, object_names)
            holding[-1] = None
            object_dict[arg1]['inside'] = arg2

        elif action[:3] == 'put' and ' onto ' in action:
            mid_idx = action.find(' onto ')
            act_idx = 8
            arg1 = action[4:mid_idx]
            arg2 = action[mid_idx+6:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            arg2, arg2_idx = get_argument_index(arg2, object_names)
            holding[-1] = None
            object_dict[arg1]['ontop'] = arg2

        elif action[:4] == 'take' and ' from ' in action:
            mid_idx = action.find(' into ')
            act_idx = 9
            arg1 = action[5:mid_idx]
            arg2 = action[mid_idx+6:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            arg2, arg2_idx = get_argument_index(arg2, object_names)     # human
            human_holding[-1].remove(arg1)
            holding[-1] = arg1

        elif action[:4] == 'give' and ' to ' in action:
            mid_idx = action.find(' to ')
            act_idx = 10
            arg1 = action[5:mid_idx]
            arg2 = action[mid_idx+4:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            arg2, arg2_idx = get_argument_index(arg2, object_names)     # human
            human_holding[-1].append(arg1)
            holding[-1] = None

        elif action[:4] == 'open':
            act_idx = 11
            arg1 = action[5:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            object_dict[arg1]['states']['open'] = True

        elif action[:5] == 'close':
            act_idx = 12
            arg1 = action[6:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            object_dict[arg1]['states']['open'] = False

        elif action[:9] == 'toggle on':
            act_idx = 13
            arg1 = action[10:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            object_dict[arg1]['states']['toggled'] = True

        elif action[:10] == 'toggle off':
            act_idx = 14
            arg1 = action[11:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            object_dict[arg1]['states']['toggled'] = False

        elif action[:4] == 'heat':
            act_idx = 15
            arg1 = action[5:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            object_dict[arg1]['states']['cooked'] = True
            object_dict[arg1]['states']['frozen'] = False

        elif action[:4] == 'cool':
            act_idx = 16
            arg1 = action[5:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            object_dict[arg1]['states']['frozen'] = True
            object_dict[arg1]['states']['cooked'] = False

        elif action[:4] == 'soak':
            act_idx = 17
            arg1 = action[5:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            object_dict[arg1]['states']['soaked'] = True

        elif action[:5] == 'slice' and ' with ' in action:
            mid_idx = action.find(' with ')
            act_idx = 18
            arg1 = action[6:mid_idx]
            arg2 = action[mid_idx+6:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            arg2, arg2_idx = get_argument_index(arg2, object_names)
            object_dict[arg1]['states']['sliced'] = True

        elif action[:5] == 'clean' and ' with ' in action:
            mid_idx = action.find(' with ')
            act_idx = 19
            arg1 = action[6:mid_idx]
            arg2 = action[mid_idx+6:]
            arg1, arg1_idx = get_argument_index(arg1, object_names)
            arg2, arg2_idx = get_argument_index(arg2, object_names)
            object_dict[arg1]['states']['dusty'] = False
            object_dict[arg1]['states']['stained'] = False

        else:
            act_idx = 0
            arg1_idx = 0
        solutions.append((act_idx, arg1_idx, arg2_idx))
        object_dicts.append(c(object_dict))
        choices.append(applicable_operators)
        if not eval_use:
            env.step(action)

    return object_dicts, cur_loc, holding, human_holding, solutions, choices, choices_in_text


def parse_action(action, object_names):
    """
    Get argument triple from action in text
    """
    arg2_idx = 0
    if action[:7] == 'move to':
        act_idx = 4
        arg1 = action[8:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)

    elif action[:7] == 'pick up' and ' from ' not in action:
        act_idx = 5
        arg1 = action[8:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)

    elif action[:7] == 'pick up' and ' from ' in action:
        mid_idx = action.find(' from ')
        act_idx = 6
        arg1 = action[8:mid_idx]
        arg2 = action[mid_idx + 6:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)
        arg2, arg2_idx = get_argument_index(arg2, object_names)

    elif action[:3] == 'put' and ' into ' in action:
        mid_idx = action.find(' into ')
        act_idx = 7
        arg1 = action[4:mid_idx]
        arg2 = action[mid_idx + 6:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)
        arg2, arg2_idx = get_argument_index(arg2, object_names)

    elif action[:3] == 'put' and ' onto ' in action:
        mid_idx = action.find(' onto ')
        act_idx = 8
        arg1 = action[4:mid_idx]
        arg2 = action[mid_idx + 6:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)
        arg2, arg2_idx = get_argument_index(arg2, object_names)

    elif action[:4] == 'take' and ' from ' in action:
        mid_idx = action.find(' into ')
        act_idx = 9
        arg1 = action[5:mid_idx]
        arg2 = action[mid_idx + 6:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)
        arg2, arg2_idx = get_argument_index(arg2, object_names)  # human

    elif action[:4] == 'give' and ' to ' in action:
        mid_idx = action.find(' to ')
        act_idx = 10
        arg1 = action[5:mid_idx]
        arg2 = action[mid_idx + 4:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)
        arg2, arg2_idx = get_argument_index(arg2, object_names)  # human

    elif action[:4] == 'open':
        act_idx = 11
        arg1 = action[5:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)

    elif action[:5] == 'close':
        act_idx = 12
        arg1 = action[6:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)

    elif action[:9] == 'toggle on':
        act_idx = 13
        arg1 = action[10:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)

    elif action[:10] == 'toggle off':
        act_idx = 14
        arg1 = action[11:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)

    elif action[:4] == 'heat':
        act_idx = 15
        arg1 = action[5:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)

    elif action[:4] == 'cool':
        act_idx = 16
        arg1 = action[5:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)

    elif action[:4] == 'soak':
        act_idx = 17
        arg1 = action[5:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)

    elif action[:5] == 'slice' and ' with ' in action:
        mid_idx = action.find(' with ')
        act_idx = 18
        arg1 = action[6:mid_idx]
        arg2 = action[mid_idx + 6:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)
        arg2, arg2_idx = get_argument_index(arg2, object_names)

    elif action[:5] == 'clean' and ' with ' in action:
        mid_idx = action.find(' with ')
        act_idx = 19
        arg1 = action[6:mid_idx]
        arg2 = action[mid_idx + 6:]
        arg1, arg1_idx = get_argument_index(arg1, object_names)
        arg2, arg2_idx = get_argument_index(arg2, object_names)

    else:
        act_idx = 0
        arg1_idx = 0
    return act_idx, arg1_idx, arg2_idx


def get_argument_index(arg, object_names):
    """
    Get object index within list of object names
    """
    arg = arg.strip()
    arg = arg.replace(' ', '#')
    if arg in object_names:
        return arg, object_names.index(arg)
    else:
        arg += '#0'
        if arg in object_names:
            return arg, object_names.index(arg)
        else:
            return arg, 0


def get_simple_object_graph(graphs, tokenizer):
    tok = tokenizer.single_tokenizer
    batch_tokens = list()
    for i, graph in enumerate(graphs):
        tokens = list()
        for idx, obj in enumerate(graph['node']):
            tokens += [tok('[OBJ]')] + obj
            tokens += [tok('[REL]')]
            rel = graph['relation'][idx]
            has_loc = False
            for index, value in enumerate(rel):
                if value != 0:
                    has_loc = True
                    tokens += [value] + graph['node'][index][0:2]
                    break
            if not has_loc:
                tokens += [0, 0, 0]
        tokens[0] = tok('[INT]')
        batch_tokens.append(tokens)
    return batch_tokens


def get_object_graph(object_names, tokenizer, object_dicts, cur_loc, holding, human_holding):
    tok = tokenizer.single_tokenizer
    object_graphs = list()
    for i, object_dict in enumerate(object_dicts[:-1]):
        graph = {'node': [], 'relation': []}
        robot_node = [tok('robot')] + [0] * 10
        human_node = [tok('human')] + [0] * 10
        robot_relation = [0, 0]
        for obj in object_dict.keys():
            if cur_loc == obj:
                robot_relation.append(tok('at'))
            else:
                robot_relation.append(0)
        human_relation = [0] * len(object_names)
        graph['node'] += [robot_node, human_node]
        graph['relation'] += [robot_relation, human_relation]

        for obj in object_dict.keys():
            name = obj.split('#')
            if len(name) > 1:
                label = name[1]
            else:
                label = None
            node = get_node(object_dict[obj], label, tokenizer)
            relation = get_relation(object_dict[obj], object_names, obj in human_holding[i], obj==holding[i], tokenizer)
            graph['node'].append(node)
            graph['relation'].append(relation)
        object_graphs.append(graph)
    return object_graphs


def get_node(obj, label, tokenizer):
    tok = tokenizer.single_tokenizer
    attrs = list()
    attrs.append(tok(obj['type']))                  # type
    if label:
        attrs.append(tok(label))                    # label
    else:
        attrs.append(0)
    if 'size' in obj['states'].keys():              # size
        attrs.append(tok(obj['states']['size']))
    else:
        attrs.append(0)
    if 'color' in obj['states'].keys():             # color
        attrs.append(tok(obj['states']['color']))
    else:
        attrs.append(0)
    DUSTY_DICT = {True: tok('dusty'), False: tok('not_dusty'), None: 0}             # dusty
    attrs.append(DUSTY_DICT[obj['states'].get('not_dusty', None)])
    STAINED_DICT = {True: tok('stained'), False: tok('not_stained'), None: 0}       # stained
    attrs.append(STAINED_DICT[obj['states'].get('not_stained', None)])
    COOKED_DICT = {True: tok('cooked'), False: tok('not_cooked'), None: 0}          # cooked
    attrs.append(COOKED_DICT[obj['states'].get('cooked', None)])
    FROZEN_DICT = {True: tok('frozen'), False: tok('not_frozen'), None: 0}          # frozen
    attrs.append(FROZEN_DICT[obj['states'].get('frozen', None)])
    OPEN_DICT = {True: tok('open'), False: tok('closed'), None: 0}                  # open
    attrs.append(OPEN_DICT[obj['states'].get('open', None)])
    TOGGLED_DICT = {True: tok('toggled'), False: tok('not_toggled'), None: 0}       # toggled
    attrs.append(TOGGLED_DICT[obj['states'].get('toggled', None)])
    SLICED_DICT = {True: tok('sliced'), False: tok('unsliced'), None: 0}            # sliced
    attrs.append(SLICED_DICT[obj['states'].get('sliced', None)])
    return attrs


def get_relation(obj, object_names, is_human_holding, is_holding, tokenizer):
    tok = tokenizer.single_tokenizer
    relation = [0] * len(object_names)
    if 'inside' in obj.keys():
        loc_idx = object_names.index(obj['inside'])
        relation[loc_idx] = tok('inside')
    elif 'ontop' in obj.keys():
        loc_idx = object_names.index(obj['ontop'])
        relation[loc_idx] = tok('ontop')
    elif is_human_holding:
        loc_idx = 1
        relation[loc_idx] = tok('human_holding')
    elif is_holding:
        loc_idx = 0
        relation[loc_idx] = tok('robot_holding')
    return relation


def replace_object_token(text, object_names, tokenizer):

    replace_indices = list()
    object_indices = list()
    for idx in range(len(text) - 1):
        token = text[idx]
        if token < 25:
            continue
        if tokenizer.reverse_vocab[token] == 'human':
            continue
        if tokenizer.reverse_vocab[token] in object_names:
            replace_indices.append(idx)
            object_indices.append(object_names.index(tokenizer.reverse_vocab[token]))
        elif tokenizer.reverse_vocab[token] + '#' + tokenizer.reverse_vocab[text[idx + 1]] in object_names:
            replace_indices.append(idx)
            object_indices.append(1)
            replace_indices.append(idx + 1)
            object_indices.append(object_names.index(
                tokenizer.reverse_vocab[token] + '#' + tokenizer.reverse_vocab[text[idx + 1]]))

    return replace_indices, object_indices


def get_action_effect(demo_observations, tok):
    effect = list()
    for obs in demo_observations:
        idx = obs.find('[SEP]')
        effect.append(tok.sentence_tokenizer('[INT] ' + obs[idx:]))
    return effect


def get_mid(subgoal, mid_tokenizers):
    q_tok, s_tok, v_tok, o_tok = mid_tokenizers
    q, s, v, o = 0, 0, 0, 0
    # set q
    if 'forall' in subgoal:
        q = 1
    elif 'exists' in subgoal:
        q = 2
    else:
        q = 0

    for idx, prep in enumerate(['inside', 'ontop']):
        if prep in subgoal:
            v = idx + 1
            if 'exists (?y - phyobj)' in subgoal and '?x ?y' in subgoal:    # specialize for exist y forall x
                subjective = subgoal[subgoal.find('not (type-') + 10:].split()[0]
                if subjective in s_tok.vocab.keys():
                    s = s_tok.vocab[subjective] - 1
                objective = subgoal[subgoal.find('and (type-') + 10:].split()[0]
                if objective in o_tok.vocab.keys():
                    o = o_tok.vocab[objective] - 1
            elif 'type-' in subgoal:
                subjective = subgoal[subgoal.find('type-') + 5:].split()[0]
                if subjective in s_tok.vocab.keys():
                    s = s_tok.vocab[subjective] - 1
                objective = subgoal[subgoal.find(prep[-5:]) + 9:].split(')')[0]
                objective = objective.split('#')[0]
                if objective in o_tok.vocab.keys():
                    o = o_tok.vocab[objective] - 1
            elif '#' in subgoal:
                subjective = subgoal[:subgoal.find('#')].split()[-1]
                if subjective in s_tok.vocab.keys():
                    s = s_tok.vocab[subjective] - 1
                objective = subgoal[subgoal.find('#') + 3:].split(')')[0]
                objective = objective.split('#')[0]
                if objective in o_tok.vocab.keys():
                    o = o_tok.vocab[objective] - 1

    # dominate inside and ontop
    for idx, state in enumerate(['dusty', 'sliced', 'open', 'closed', 'cooked', 'frozen', 'toggled', 'holding']):
        if state in subgoal:
            v = idx + 3
            # other cases
            if 'inside' not in subgoal and 'ontop' not in subgoal:
                if 'type-' in subgoal:
                    subjective = subgoal[subgoal.find('type-') + 5:].split()[0]
                    if subjective in s_tok.vocab.keys():
                        s = s_tok.vocab[subjective] - 1
                elif '#' in subgoal:
                    subjective = subgoal[:subgoal.find('#')].split()[-1]
                    if subjective in s_tok.vocab.keys():
                        s = s_tok.vocab[subjective] - 1
                else:
                    subjective = subgoal.replace(')', '').split()[-1]
                    if subjective in s_tok.vocab.keys():
                        s = s_tok.vocab[subjective] - 1

    return q, s, v, o


if __name__ == '__main__':
    _, _, _, _, mid_tokenizers, _ = get_tokenizers_and_vocabs('vocabulary')
    fol = input()
    q, s, v, o = get_mid(fol, mid_tokenizers)
    print(translate_subgoal(q, s, v, o, mid_tokenizers))
