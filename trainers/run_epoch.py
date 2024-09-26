import time
from trainers.trainer_utils import translate_action, translate_actions, translate_subgoal


def e2e_run_epoch(
        mid,
        data_iter,      # iterator of data batches
        model,
        loss_compute,
        mid_loss_compute,
        obj_loss_compute,
        optimizer,
        scheduler,
        tokenizer,
        mode="train",   # train, validate, test
        accum_iter=1,
        train_state=None,
):
    """Train a single epoch"""
    mid_tokenizers = tokenizer[2]
    start = time.time()
    episode_counter = 0
    action_counter = 0
    qsvo_counter = 0
    obj_counter = 0
    obj_type_counter = 0

    for i, batch in enumerate(data_iter):
        hidden, qsvo_output, obj_output = model.forward(
            batch.src,
            batch.src_mask,
            batch.qsvo,
            batch.target_obj_indices,
            batch.replace_indices,
            batch.object_indices,
        )
        qsvo_loss, (q_pred, s_pred, v_pred, o_pred) = mid_loss_compute(qsvo_output, batch.qsvo)
        obj_loss, obj_pred = obj_loss_compute(obj_output, batch.target_obj_indices)
        action_loss, _, action_outputs = loss_compute(hidden, batch.solutions, batch.choices)
        if mid == 'qsvo+obj':
            loss = qsvo_loss + obj_loss + action_loss
        elif mid == 'qsvo':
            loss = qsvo_loss + action_loss
        elif mid == 'obj':
            loss = obj_loss + action_loss
        else:
            loss = action_loss
        for idx, output in enumerate(action_outputs):
            if mode == 'test':
                print(batch.json_indices[idx])
            targ = (batch.tgt[0][idx].item(), batch.tgt[1][idx].item(), batch.tgt[2][idx].item())
            pred = batch.choices[idx][output.argmax()]
            if targ == pred:
                action_counter += 1
            elif mode == 'test':
                print(translate_action(targ[0], targ[1], targ[2], batch.object_names[0]), '\t',
                      translate_action(pred[0], pred[1], pred[2], batch.object_names[0]), '\t\tIncorrect action!')
            if mid in ['qsvo', 'qsvo+obj']:
                qsvo_targ = (batch.qsvo[0][idx].item(), batch.qsvo[1][idx].item(),
                             batch.qsvo[2][idx].item(), batch.qsvo[3][idx].item())
                qsvo_pred = (q_pred[idx].item(), s_pred[idx].item(), v_pred[idx].item(), o_pred[idx].item())
                if qsvo_targ == qsvo_pred:
                    qsvo_counter += 1
                elif mode == 'test':
                    print(translate_subgoal(qsvo_targ[0], qsvo_targ[1], qsvo_targ[2], qsvo_targ[3], mid_tokenizers),
                          '\t',
                          translate_subgoal(qsvo_pred[0], qsvo_pred[1], qsvo_pred[2], qsvo_pred[3], mid_tokenizers),
                          '\t\tIncorrect QSVO!')
            if mid in ['obj', 'qsvo+obj']:
                object_targ = batch.target_obj_indices[idx].item()
                object_pred = obj_pred[idx].item()
                if object_targ == object_pred:
                    obj_counter += 1
                elif mode == 'test':
                    print(batch.object_names[0][object_targ], '\t',
                          batch.object_names[0][object_pred], '\tIncorrect output!')
                if batch.object_names[idx][object_targ].split('#')[0] == \
                        batch.object_names[idx][object_pred].split('#')[0]:
                    obj_type_counter += 1
            episode_counter += 1
        if mode == "train":
            loss.backward()
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()
        train_state.samples += batch.size
        train_state.total_loss += loss.data.item()
        train_state.accum_step += 1

        if i % 100 == 0 and (mode == "train"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f, %6.2f, %6.2f "
                        + "| Elapsed Time: %6.6f | Learning Rate: %6.1e"
                )
                % (i, train_state.accum_step, qsvo_loss.data.item(), obj_loss.data.item(), action_loss.data.item(),
                   elapsed, lr)
            )
            start = time.time()
        del loss
    return train_state, train_state.total_loss / train_state.samples, action_counter / episode_counter, \
        qsvo_counter / episode_counter, obj_counter / episode_counter, obj_type_counter / episode_counter


def r_run_epoch(
        mid,
        data_iter,      # iterator of data batches
        model,
        mid_loss_compute,
        obj_loss_compute,
        optimizer,
        scheduler,
        tokenizer,
        mode="train",   # train, validate, test
        accum_iter=1,
        train_state=None,
):
    """Train a single epoch"""
    mid_tokenizers = tokenizer[2]
    start = time.time()
    episode_counter = 0
    qsvo_counter = 0
    obj_counter = 0
    obj_type_counter = 0

    for i, batch in enumerate(data_iter):
        hidden, qsvo_output, obj_output = model.forward(
            batch.src,
            batch.src_mask,
            batch.qsvo,
            batch.target_obj_indices,
            batch.replace_indices,
            batch.object_indices,
        )
        qsvo_loss, (q_pred, s_pred, v_pred, o_pred) = mid_loss_compute(qsvo_output, batch.qsvo)
        obj_loss, obj_pred = obj_loss_compute(obj_output, batch.target_obj_indices)
        if mid == 'qsvo+obj':
            loss = qsvo_loss + obj_loss
        elif mid == 'qsvo':
            loss = qsvo_loss
        elif mid == 'obj':
            loss = obj_loss
        else:
            loss = 0.0

        for idx in range(hidden[0].shape[0]):
            if mode == 'test':
                print(batch.json_indices[idx])
            if mid in ['qsvo', 'qsvo+obj']:
                qsvo_targ = (batch.qsvo[0][idx].item(), batch.qsvo[1][idx].item(),
                             batch.qsvo[2][idx].item(), batch.qsvo[3][idx].item())
                qsvo_pred = (q_pred[idx].item(), s_pred[idx].item(), v_pred[idx].item(), o_pred[idx].item())
                if qsvo_targ == qsvo_pred:
                    qsvo_counter += 1
                elif mode == 'test':
                    print(translate_subgoal(qsvo_targ[0], qsvo_targ[1], qsvo_targ[2], qsvo_targ[3], mid_tokenizers),
                          '\t',
                          translate_subgoal(qsvo_pred[0], qsvo_pred[1], qsvo_pred[2], qsvo_pred[3], mid_tokenizers),
                          '\t\tIncorrect QSVO!')
            if mid in ['obj', 'qsvo+obj']:
                object_targ = batch.target_obj_indices[idx].item()
                object_pred = obj_pred[idx].item()
                if object_targ == object_pred:
                    obj_counter += 1
                elif mode == 'test':
                    print(batch.object_names[0][object_targ], '\t',
                          batch.object_names[0][object_pred], '\tIncorrect output!')
                if batch.object_names[idx][object_targ].split('#')[0] == \
                        batch.object_names[idx][object_pred].split('#')[0]:
                    obj_type_counter += 1
            episode_counter += 1

        if mode == "train":
            loss.backward()
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()
        train_state.samples += batch.size
        train_state.total_loss += loss.data.item()
        train_state.accum_step += 1

        if i % 100 == 0 and (mode == "train"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f, %6.2f "
                        + "| Elapsed Time: %6.6f | Learning Rate: %6.1e"
                )
                % (i, train_state.accum_step, qsvo_loss.data.item(), obj_loss.data.item(), elapsed, lr)
            )
            start = time.time()
        del loss
    return train_state, train_state.total_loss / train_state.samples, \
        qsvo_counter / episode_counter, obj_counter / episode_counter, obj_type_counter / episode_counter


def p_run_epoch(
        mid,
        data_iter,      # iterator of data batches
        model,
        loss_compute,
        optimizer,
        scheduler,
        tokenizer,
        mode="train",   # train, validate, test
        accum_iter=1,
        train_state=None,
):
    """Train a single epoch"""
    mid_tokenizers = tokenizer[2]
    start = time.time()
    episode_counter = 0
    action_counter = 0

    for i, batch in enumerate(data_iter):
        hidden = model.forward(
            batch.src,
            batch.src_mask,
            batch.qsvo,
            batch.target_obj_indices,
            batch.replace_indices,
            batch.object_indices,
        )
        action_loss, _, action_outputs = loss_compute(hidden, batch.solutions, batch.choices)
        loss = action_loss

        for idx, output in enumerate(action_outputs):
            if mode == 'test':
                print(batch.json_indices[idx])
            targ = (batch.tgt[0][idx].item(), batch.tgt[1][idx].item(), batch.tgt[2][idx].item())
            pred = batch.choices[idx][output.argmax()]
            if targ == pred:
                action_counter += 1
            elif mode == 'test':
                print(translate_action(targ[0], targ[1], targ[2], batch.object_names[0]), '\t',
                      translate_action(pred[0], pred[1], pred[2], batch.object_names[0]), '\t\tIncorrect action!')
            episode_counter += 1

        if mode == "train":
            loss.backward()
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()
        train_state.samples += batch.size
        train_state.total_loss += loss.data.item()
        train_state.accum_step += 1

        if i % 100 == 0 and (mode == "train"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Elapsed Time: %6.6f | Learning Rate: %6.1e"
                )
                % (i, train_state.accum_step, action_loss.data.item(),
                   elapsed, lr)
            )
            start = time.time()
        del loss
    return train_state, train_state.total_loss / train_state.samples, action_counter / episode_counter
