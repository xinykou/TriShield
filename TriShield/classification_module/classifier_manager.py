from classification_module.layer_classifier import LayerClassifierTorch
from tqdm import tqdm
import torch
import os


class ClassifierManager:
    def __init__(self, n_layers: int):
        self.classifiers = []
        self.testacc = []
        self.n_layer = n_layers

    def _train_classifiers(
        self,
        pos_embds=None,
        neg_embds=None,
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        print("Training classifiers...")
        for i in tqdm(range(self.n_layer)):
            layer_classifier = LayerClassifierTorch(input_dim=pos_embds[i].shape[1])
            layer_classifier.train_model(
                pos_tensor=pos_embds[i],
                neg_tensor=neg_embds[i],
                n_epoch=n_epochs,
                batch_size=batch_size,
                lr=lr,
            )

            self.classifiers.append(layer_classifier)

    def _evaluate_testacc(self, pos_embds=None, neg_embds=None):
        for i in tqdm(range(len(self.classifiers))):
            self.testacc.append(
                self.classifiers[i].evaluate_testacc(
                    pos_tensor=pos_embds[i],
                    neg_tensor=neg_embds[i],
                )
            )

    def fit(
        self,
        pos_embds_train=None,
        neg_embds_train=None,
        pos_embds_test=None,
        neg_embds_test=None,
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        if pos_embds_train is not None and neg_embds_train is not None:
            self._train_classifiers(
                pos_embds_train,
                neg_embds_train,
                lr,
                n_epochs,
                batch_size,
            )
        if pos_embds_test is not None and neg_embds_test is not None:
            self._evaluate_testacc(
                pos_embds_test,
                neg_embds_test,
            )

        return self

    def save(self, relative_path: str):
        if not os.path.exists(relative_path):
            os.makedirs(relative_path)
        torch.save(self, os.path.join(relative_path, "classifier.pt"))


def load_classifier_manager(file_path: str):
    return torch.load(file_path, weights_only=False)
