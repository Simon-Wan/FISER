import torch
import os
import json
import random
import numpy as np

from networks.tokenizer import get_tokenizers_and_vocabs
from dataloaders.dataloader import preprocess, make_batch
from trainers.trainer_utils import *
from evaluators.evaluator_utils import preprocess_from_env
from networks.rp_model import make_r_model, make_p_model
from networks.loss import get_loss_compute, mid_loss_compute, obj_loss_compute
from data_generation.text_interface.env import HMTEnv


def rp_eval(args):
    print(args)

    # set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # set seed
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    data_path = args.data_path
    data_dir = os.path.join(data_path, args.data_dir_name)
    if not args.v1 and not args.gen:
    # set path
        data_info = os.path.join(data_path, args.data_info_name)

        # prepare dataset
        with open(data_info, 'r') as f:
            json_str = json.load(f)
        files = set(json_str[0][args.quest_type])
        if args.level != 'level0':
            files = set(json_str[1][args.level]) & files
        if args.goal != -1:
            goal_files = json_str[2][str(args.goal)]
            files = set(goal_files) & files
        test_files = list(set(json_str[3]['test']) & files)
        test_files.sort()
        if args.num_files > len(test_files):
            args.num_files = len(test_files)
        test_files = list(np.random.choice(test_files, len(test_files)))[:args.num_files]  # random permutation for input
        print('# Test files:', len(test_files))
    elif args.v1:
        test_files = []
        GOALS = [2, 3, 24, 25, 26, 30, 31, 32, 33, 35, 36, 46, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66]
        files = os.listdir(os.path.join(args.data_path, args.data_dir_name))
        for f in files:
            if f.split('.')[-1] == 'json':
                if 'bring_me' in f:
                    if f.split('_')[0] == args.level:
                        if int((f.split('.')[0]).split('goal')[1]) in GOALS:
                            if int(f.split('_')[1][4:]) < 500:
                                test_files.append(f)
        if args.num_files > len(test_files):
            args.num_files = len(test_files)
        test_files = list(np.random.choice(test_files, len(test_files)))[
                     :args.num_files]  # random permutation for input
        print('# Test files:', len(test_files))
    else:
        test_files = []
        files = os.listdir(os.path.join(args.data_path, args.data_dir_name))
        GOALS = [69, 70, 73, 74, 78]
        for f in files:
            if f.split('.')[-1] == 'json':
                if 'bring_me' in f:
                    if f.split('-')[-1][0] == args.level[-1]:
                        if int((f.split('-')[-3])) in GOALS:
                            if int(f.split('-')[-2]) < 40:
                                test_files.append(f)
        if args.num_files > len(test_files):
            args.num_files = len(test_files)
        test_files = list(np.random.choice(test_files, len(test_files)))[
                     :args.num_files]  # random permutation for input
        print('# Test files:', len(test_files))

    vocabulary_dir = os.path.dirname(__file__) + '/../vocabulary'
    src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, mid_tokenizers, mid_vocabs = \
        get_tokenizers_and_vocabs(vocabulary_dir)

    r_model = make_r_model(args.mid, src_vocab, tgt_vocab, mid_vocabs, device,
                           d_model=args.d_model, N_qsvo=args.N_qsvo, N_obj=args.N_obj)
    r_save_name = '_'.join([args.experiment_name, 'r', args.mid,
                            'depth%d%d%d' % (args.N_qsvo, args.N_obj, args.N_action),
                            'width%d' % args.d_model, 'seed%d' % args.r_seed, 'epoch%d' % args.r_epoch_id])
    r_state_dict = torch.load(os.path.join(args.save_path, r_save_name))
    r_model.load_state_dict(r_state_dict)
    r_model = r_model.to(device)
    p_model = make_p_model(args.mid, src_vocab, tgt_vocab, mid_vocabs, device,
                           d_model=args.d_model, N_qsvo=args.N_qsvo, N_obj=args.N_obj)
    p_save_name = '_'.join([args.experiment_name, 'p', args.mid,
                            'depth%d%d%d' % (args.N_qsvo, args.N_obj, args.N_action),
                            'width%d' % args.d_model, 'seed%d' % args.p_seed, 'epoch%d' % args.p_epoch_id])
    p_state_dict = torch.load(os.path.join(args.save_path, p_save_name))
    p_model.load_state_dict(p_state_dict)
    p_model = p_model.to(device)

    loss_compute = get_loss_compute(p_model.generator)
    r_model.eval()
    p_model.eval()
    with torch.no_grad():
        episode_counter = 0
        action_counter = 0
        qsvo_counter = 0
        obj_counter = 0
        obj_type_counter = 0
        success_counter = 0
        for file_idx, file in enumerate(test_files):
            with open(os.path.join(data_dir, file), 'r') as f:
                json_str = json.load(f)
            single_env = HMTEnv(
                os.path.join(data_dir, file),
                fully=True
            )
            single_env.reset()
            single_data = preprocess(json_str, src_tokenizer, single_env, file, mid_tokenizers)
            solutions = [b['output'] for b in single_data]

            env = HMTEnv(
                os.path.join(data_dir, file),
                fully=True,
            )
            env.reset()
            counter = 0
            effects = ['[INT]']
            info_record = env.info.copy()
            current_object_dict = env.object_dict.copy()
            init_graph = None
            holding_info = None
            print('\n\n----------------------------------------------------')
            print('File {}:'.format(file_idx), file)
            print(json_str['task_description'])
            for sol_idx, sol in enumerate(solutions):
                print('Step {}:'.format(sol_idx))
                demo_action = env.info["next_expert_action"]
                if sol_idx == 0:
                    new_data, current_object_dict, holding_info = \
                        preprocess_from_env(env, json_str, src_tokenizer, file, counter, effects, demo_action,
                                            current_object_dict, args.batch_size, mid_tokenizers=mid_tokenizers)
                    init_graph = [new_data[0]['initial_graph']]
                else:
                    new_data, current_object_dict, holding_info = \
                        preprocess_from_env(env, json_str, src_tokenizer, file, counter, effects, demo_action,
                                            current_object_dict, args.batch_size,
                                            init_graph, holding_info, mid_tokenizers)
                batch = make_batch(new_data, device)
                act_targ = torch.tensor([sol[0]] * args.batch_size, dtype=torch.long).to(device)
                arg1_targ = torch.tensor([sol[1]] * args.batch_size, dtype=torch.long).to(device)
                arg2_targ = torch.tensor([sol[2]] * args.batch_size, dtype=torch.long).to(device)
                batch.tgt = (act_targ, arg1_targ, arg2_targ)
                target_obj_indices = torch.tensor([solutions[-1][1]] * args.batch_size, dtype=torch.long).to(device)
                batch.target_obj_indices = target_obj_indices
                mid_hidden, qsvo_output, obj_output = r_model.forward(
                    batch.src,
                    batch.src_mask,
                    batch.qsvo,
                    batch.target_obj_indices,
                    batch.replace_indices,
                    batch.object_indices,
                    gt_qsvo=args.gt_qsvo
                )
                qsvo_loss, (q_pred, s_pred, v_pred, o_pred) = mid_loss_compute(qsvo_output, batch.qsvo)
                obj_loss, obj_pred = obj_loss_compute(obj_output, batch.target_obj_indices)
                if args.mid in ['qsvo', 'qsvo+obj']:
                    qsvo_targ = (batch.qsvo[0][0].item(), batch.qsvo[1][0].item(),
                                 batch.qsvo[2][0].item(), batch.qsvo[3][0].item())
                    qsvo_pred = (q_pred[0].item(), s_pred[0].item(), v_pred[0].item(), o_pred[0].item())
                    if qsvo_targ == qsvo_pred:
                        qsvo_counter += 1
                    else:
                        print(translate_subgoal(qsvo_targ[0], qsvo_targ[1], qsvo_targ[2], qsvo_targ[3], mid_tokenizers),
                              '\t',
                              translate_subgoal(qsvo_pred[0], qsvo_pred[1], qsvo_pred[2], qsvo_pred[3], mid_tokenizers),
                              '\t\tIncorrect QSVO!')
                if args.mid in ['obj', 'qsvo+obj']:
                    object_targ = batch.target_obj_indices[0].item()
                    object_pred = obj_pred[0].item()
                    if object_targ == object_pred:
                        obj_counter += 1
                    else:
                        print(batch.object_names[0][object_targ], '\t',
                              batch.object_names[0][object_pred], '\tIncorrect output!')
                    if batch.object_names[0][object_targ].split('#')[0] == \
                            batch.object_names[0][object_pred].split('#')[0]:
                        obj_type_counter += 1
                
                if args.gt_qsvo:
                    q_pred = batch.qsvo[0]
                    s_pred = batch.qsvo[1]
                    v_pred = batch.qsvo[2]
                    o_pred = batch.qsvo[3]
                if args.gt_obj:
                    obj_pred = batch.target_obj_indices
                hidden = p_model.forward(
                    batch.src,
                    batch.src_mask,
                    (q_pred, s_pred, v_pred, o_pred),
                    obj_pred,
                    batch.replace_indices,
                    batch.object_indices,
                )
                action_loss, _, action_outputs = loss_compute(hidden, batch.solutions, batch.choices)
                targ = (batch.tgt[0][0].item(), batch.tgt[1][0].item(), batch.tgt[2][0].item())
                pred = batch.choices[0][action_outputs[0].argmax()]
                if targ == pred:
                    action_counter += 1
                else:
                    print(translate_action(targ[0], targ[1], targ[2], batch.object_names[0]), '\t',
                          translate_action(pred[0], pred[1], pred[2], batch.object_names[0]),
                          '\t\tIncorrect action!')
                action_str = batch.choices_in_text[0][action_outputs[0].argmax()]
                action_str = action_str.replace('#', ' ')
                env.info = info_record.copy()
                obs, reward, done, info = env.step(action_str)
                obs = obs.replace('#', ' ')
                info_record = info.copy()
                effects[0] += ' [SEP] ' + action_str + ' [SEP] ' + obs
                counter += 1
                episode_counter += 1
                if done:
                    break
                # print('Score:', env.info['score'])
            if env.info['score'] > 0:
                success_counter += 1
                print('Succeed!')
            else:
                print('Fail!')
        print(
            "Success Rate: %.4f | Acc: %.4f | QSVO Acc: %.4f | Object Acc: %.4f | Object Type Acc: %.4f\n"
            % (success_counter / len(test_files),
               action_counter / episode_counter,
               qsvo_counter / episode_counter,
               obj_counter / episode_counter,
               obj_type_counter / episode_counter,
               )
        )
