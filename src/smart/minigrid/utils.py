import os
import json
import yaml

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, dict):
            return Config(value)
        elif isinstance(value, list):
            return [self._wrap(item) for item in value]
        return value

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"
    
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [v.to_dict() if isinstance(v, Config) else v for v in value]
            else:
                result[key] = value
        return result
    
    def save(self, path):
        data = self.to_dict()
        with open(path, 'w') as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                yaml.dump(data, f)
            elif path.endswith(".json"):
                json.dump(data, f, indent=2)
            else:
                raise ValueError("Unsupported file extension. Use .yaml, .yml, or .json")

    
    
def load_config(path):
    if path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            config_dict = json.load(f)
    else:
        raise ValueError("Unsupported file type. Use .json or .yaml/.yml")

    return Config(config_dict)

