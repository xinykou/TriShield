import torch
from loss import masked_log_probs


def get_edit_labels(tok, labels):
    return labels.masked_fill(labels == tok.pad_token_id, -100)


def edit(
    adversarial_prompt,
    safe_response,
    tokenizer,
    model,
    layers,
    learning_rate,
    epoch=None,
    model_path=None,
):
    # dinm
    # adversarial_prompt = d['knowledge constrain']['prompt']
    # safe_response = d['safe generation']
    DEVICE = model.device

    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in layers  # specific layer for each instance
        if "layers.{}.mlp.down_proj.weight".format(layer) in n
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=learning_rate,
        weight_decay=0,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # no need for constraint knowledge

    ############edit toxic regions#############################
    # # Update loop: intervene at layers simultaneously
    # loss_meter = AverageMeter()
    template = {"role": "user", "content": adversarial_prompt}
    if "qwen3" in model_path.lower():
        templated_input = tokenizer.apply_chat_template(
            [template],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    elif "llama" in model_path.lower() or "innospark" in model_path.lower():
        templated_input = tokenizer.apply_chat_template(
            [template],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        raise Exception(f"Unknown model path {model_path}")

    ft_input = [templated_input + " " + safe_response]
    out_ids = dict(
        tokenizer(safe_response, return_tensors="pt", padding=True).to(DEVICE)
    )  # torch.Size([1, 69])
    out_labels = get_edit_labels(tokenizer, out_ids["input_ids"])
    # print(out_ids)
    # print(out_labels)

    for it in range(epoch):
        # print(20 * "=")
        # print(f"Epoch: {it}")
        # print(20 * "=")
        inputs = tokenizer(ft_input, return_tensors="pt", padding=True).to(DEVICE)
        opt.zero_grad()  #

        output = model(**inputs).logits  # torch.Size([1, 275, 32000])
        loss_dict = masked_log_probs(output, out_labels, shift=True)
        l_edit = loss_dict["nll"]
        loss = l_edit
        print(f"Edit loss: {loss.item()}")
        if loss.item() >= 1e-4:
            loss.backward()
            opt.step()
        else:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}
    # print(deltas)
    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    # print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas
