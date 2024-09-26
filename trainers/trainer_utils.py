
class TrainState:
    """Track number of steps, examples, and tokens processed"""
    accum_step: int = 0
    samples: int = 0
    total_loss: float = 0


def translate_action(action, arg1, arg2, object_names):
    actions = [
        "",
        "help",
        "stop",
        "look",
        "move_to",
        "pick_up",
        "pick_up_from",
        "put_into",
        "put_onto",
        "take_from",
        "give_to",
        "open",
        "close",
        "toggle_on",
        "toggle_off",
        "heat",
        "cool",
        "soak",
        "slice_with",
        "clean_with",
    ]
    object_names[0] = ''
    return "{} {} {}".format(actions[action], object_names[arg1], object_names[arg2])


def translate_actions(triples, object_names):
    return [translate_action(triple[0], triple[1], triple[2], object_names) for triple in triples]


def translate_subgoal(q, s, v, o, mid_tokenizers):
    return ' '.join([mid_tokenizers[0].reverse_vocab[q + 1],
                     mid_tokenizers[1].reverse_vocab[s + 1],
                     mid_tokenizers[2].reverse_vocab[v + 1],
                     mid_tokenizers[3].reverse_vocab[o + 1]
                     ])

