import os
import tempfile
from typing import Any
from urllib.parse import urlparse

from attr import attrib, attrs



@attrs
class PathsContainer:
    local_base_output_path = attrib(type=str)
    base_output_path = attrib(type=str)
    save_dir = attrib(type=str)
    config_path = attrib(type=str)

    @classmethod
    def from_args(cls, output, run_id, config_path, package_name="lstm"):
        base_output_path = get_path_from_local_uri(output)
        if is_gs_path(base_output_path):
            local_base_output_path = tempfile.mkdtemp()
        else:
            local_base_output_path = base_output_path
        save_dir = os.path.join(local_base_output_path, run_id)
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(config_path):
            print("config not exists at {}, please insert a valid cofig path".format(config_path))
        return cls(local_base_output_path, base_output_path, save_dir, config_path)


def create_output_dirs(output_path: str) -> None:
    for subdir in ["models", "models/partial", "models/best", "training_plots"]:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)


def get_path_from_local_uri(uri: Any) -> str:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return parsed.netloc + parsed.path
    else:
        return uri

def is_gs_path(uri) -> bool:
    return urlparse(uri).scheme == "gs"
