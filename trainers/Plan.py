import torch
import random
import numpy as np
import os
import json

from networks.tokenizer import get_tokenizers_and_vocabs
from dataloaders.load_from_disk import get_dataloaders
from networks.loss import get_loss_compute, mid_loss_compute, obj_loss_compute
from trainers.trainer_utils import *
from networks.rp_model import make_p_model
from trainers.run_epoch import p_run_epoch


def p_train(args):
    print(args)

    # set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # set seed
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # set path
    data_path = args.data_path
    data_info = os.path.join(data_path, args.data_info_name)
    data_dir = os.path.join(data_path, args.data_dir_name)

    # prepare dataset
    with open(data_info, 'r') as f:
        json_str = json.load(f)
    files = set(json_str[0][args.quest_type])
    if args.level != 'level0':
        files = set(json_str[1][args.level]) & files
    if args.goal != -1:
        goal_files = json_str[2][str(args.goal)]
        files = set(goal_files) & files
    train_files = list(set(json_str[3]['train']) & files)
    valid_files = list(set(json_str[3]['validate']) & files)
    train_files = list(np.random.choice(train_files, len(train_files)))  # random permutation for input
    valid_files = list(np.random.choice(valid_files, len(valid_files)))
    print('# Train files:', len(train_files))
    print('# Validate files:', len(valid_files))

    vocabulary_dir = os.path.dirname(__file__) + '/../vocabulary'
    src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, mid_tokenizers, mid_vocabs = \
        get_tokenizers_and_vocabs(vocabulary_dir)
    tokenizer = [src_tokenizer, tgt_tokenizer, mid_tokenizers]
    train_dataloader, valid_dataloader = get_dataloaders(
        data_dir, train_files, valid_files, src_tokenizer, device, mid_tokenizers, args.batch_size
    )

    model = make_p_model(args.mid, src_vocab, tgt_vocab, mid_vocabs, device,
                         d_model=args.d_model, N_qsvo=args.N_qsvo, N_obj=args.N_obj, N_action=args.N_action)
    model = model.to(device)
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()])}")

    loss_compute = get_loss_compute(model.generator)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = None

    train_state = TrainState()
    for epoch in range(args.num_epoch):
        print("Start Epoch %d" % epoch)
        model.train()
        train_state, average_loss, action_acc = p_run_epoch(
            args.mid,
            iter(train_dataloader),
            model,
            loss_compute,
            optimizer,
            lr_scheduler,
            tokenizer,
            mode="train",
            accum_iter=1,
            train_state=train_state,
        )
        print(
            "Finish Training Epoch %d | Total loss: %6.2f | Accuracy: %.4f"
            % (epoch, average_loss, action_acc))

        if epoch % 1 == 0:
            print("Saving Model Epoch {}".format(epoch))
            save_name = '_'.join([args.experiment_name, args.pipeline, args.mid,
                                  'depth%d%d%d' % (args.N_qsvo, args.N_obj, args.N_action),
                                  'width%d' % args.d_model, 'seed%d' % seed, 'epoch%d' % epoch])
            torch.save(model.state_dict(), os.path.join(args.save_path, save_name))
            print("Finish Saving Model Epoch {}".format(epoch))

        model.eval()
        with torch.no_grad():
            valid_state = TrainState()
            valid_state, average_loss, action_acc = p_run_epoch(
                args.mid,
                iter(valid_dataloader),
                model,
                loss_compute,
                optimizer,
                lr_scheduler,
                tokenizer,
                mode="validate",
                accum_iter=1,
                train_state=valid_state,
            )
            print(
                "Finish Evaluating Epoch %d | Valid Loss: %6.2f | Accuracy: %.4f"
                % (epoch, average_loss, action_acc))
