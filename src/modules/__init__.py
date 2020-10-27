import importlib


def find_module_using_name(import_path, config):
    module = importlib.import_module(import_path)
    model = module.get_model(config)
    return model
