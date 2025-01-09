import json
def load_config(path):
    config_path = path
    with open(config_path, 'r') as file:
        return json.load(file)