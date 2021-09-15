"""
Runs a script to interact with a model using the shell.
"""
import os
from argparse import Namespace
import yaml


def load_model_from_experiment(model_class, experiment_folder: str):
    """Function that loads the model from an experiment folder.
    :param experiment_folder: Path to the experiment folder.
    Return:
        - Pretrained model.
    """
    hparams_file = experiment_folder + "/hparams.yaml"
    hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

    checkpoints = [file for file in os.listdir(experiment_folder + "/checkpoints/") if file.endswith(".ckpt")]
    checkpoint_path = experiment_folder + "/checkpoints/" + checkpoints[-1]
    model = model_class.load_from_checkpoint(checkpoint_path, hparams=Namespace(**hparams))
    # Make sure model is in prediction mode
    model.eval()
    model.freeze()
    return model
