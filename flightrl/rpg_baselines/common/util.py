from shutil import copyfile
import datetime
import os
import ntpath
import numpy as np
import tensorflow as tf


class ConfigurationSaver:
    def __init__(self, log_dir):
        self._data_dir = log_dir + '/' + \
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self._data_dir)

        # if save_items is not None:
        #     for save_item in save_items:
        #         base_file_name = ntpath.basename(save_item)
        #         copyfile(save_item, self._data_dir + '/' + base_file_name)

    @property
    def data_dir(self):
        return self._data_dir


def TensorboardLauncher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: "+url)
    webbrowser.open_new(url)

def save_model(model, save_path, model_name):
    """
    Save a model checkpoint during training.
    
    Args:
        model: The model to save
        save_path: The directory path to save to
        model_name: Name for the saved checkpoint
    """
    try:
        full_path = os.path.join(save_path, model_name)
        model.save(full_path)
        print(f"[FLIGHTMARE] Saved checkpoint: {full_path}")
        return True
    except Exception as e:
        print(f"[FLIGHTMARE] Failed to save checkpoint: {e}")
        return False
