import os
import yaml
from types import SimpleNamespace
import argparse


import torch
from classification_module.classifier_manager import ClassifierManager


def run_training(
    harmless_train_embds=None,
    harmful_train_embds=None,
    harmless_test_embds=None,
    harmful_test_embds=None,
    target_layers=None,
    learning_rate=0.0001,
    num_epochs=100,
    batch_size=64,
):

    #  layer, nums, tokens, dim --> layer, nums, dim
    harmless_train_embds = harmless_train_embds[target_layers, :, -1, :]
    harmful_train_embds = harmful_train_embds[target_layers, :, -1, :]

    clfr = ClassifierManager(len(target_layers))

    clfr.fit(
        pos_embds_train=harmful_train_embds,
        neg_embds_train=harmless_train_embds,
        pos_embds_test=harmful_test_embds,
        neg_embds_test=harmless_test_embds,
        lr=learning_rate,
        n_epochs=num_epochs,
        batch_size=batch_size,
    )

    for idx, val in enumerate(clfr.testacc):
        print(f"Layer {idx}: {val:.2f}", end="\t")
        if (idx + 1) % 5 == 0:  # 每 5 个换行
            print()

    return clfr
