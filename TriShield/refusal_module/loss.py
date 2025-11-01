import torch
import torch.nn.functional as F

def masked_log_probs(pred, targ, shift=False, exact_match=False, **kwargs):
    pred = pred.to(torch.float32)

    if not (pred.dim() == 2 or pred.dim() == 3):
        raise RuntimeError(f"Expected pred to have 2 or 3 dimensions, got {pred.shape}")

    if pred.shape[-1] == 1:
        return binary_log_probs(pred, targ)
    else:
        return multiclass_log_probs(pred, targ, shift=shift, exact_match=exact_match, **kwargs)
    

def binary_log_probs(pred, targ):
    neg_mask = torch.ones_like(pred)
    neg_mask[targ == 0] *= -1
    pred = pred * neg_mask
    log_probs = F.logsigmoid(pred)
    acc = (log_probs.exp() > 0.5).float().mean()
    return {
        "acc": acc,
        "log_prob": log_probs.mean(),
        "prob": log_probs.exp().mean(),
        "nll": -log_probs.mean(),
        "n_tokens": log_probs.shape[0],
    }


def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()
    
def multiclass_log_probs(pred, targ, shift=False, eps=torch.finfo(torch.float32).eps, exact_match=False, **kwargs):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        if "inner_sent" in kwargs or "personality" in kwargs or "multimodal" in kwargs:
            targ = targ[:, 1:]
        else:
            pred = pred[:, -targ.size(1):]
        # targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
    
    # debug
    # print(pred.shape, targ.shape)
    # if pred.size(1) > targ.size(1):
    #     pred = pred[:, :targ.size(1)]

    if exact_match:
        pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
        correct = pred_ids == targ
        if pred.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()

        # if 't5' in config.model_class.lower():
        #     end_mask = targ != 1
        #     correct = correct & end_mask
        #     num_non_padding = (mask & end_mask).sum().float().item()
        acc = correct.sum() / num_non_padding
    
    if "inner_sent" in kwargs or "inner_per" in kwargs:
        same_sent_mask = kwargs["same_mask"]
        good_mask = mask * same_sent_mask.unsqueeze(-1)
        bad_mask = mask * (~same_sent_mask.unsqueeze(-1))

        good_log_prob = masked_mean(unmasked_log_probs, good_mask)
        bad_log_prob = masked_mean((1 - unmasked_log_probs.exp() + eps).log(), bad_mask)

        n_tokens = good_mask.float().sum()
        log_prob = good_log_prob
        prob = log_prob.exp()

        if kwargs["unlikelihood"]:
            nll = -good_log_prob - bad_log_prob
        else:
            nll = -good_log_prob
    else:
        n_tokens = mask.float().sum()
        log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
        prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens
        
        nll = -log_prob
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": nll,
    }