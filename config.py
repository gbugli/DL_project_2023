import json
from collections import defaultdict
from typing import Dict, List, Optional

from attr import attrib, attrs


@attrs
class LSTMConfig:
    num_channels = attrib(type=int)
    num_kernels = attrib(type=int)
    kernel_size = attrib(type=tuple)
    padding = attrib(type=tuple)
    activation = attrib(type=str)
    frame_size = attrib(type=tuple)
    num_layers = attrib(type=int)

@attrs
class MaskerConfig:
    n_class = attrib(type=int)

@attrs
class TrainConfig:
    path = attrib(type=str)
    batch_size = attrib(type=int)

@attrs
class DataConfig:
    train = attrib(type=TrainConfig)
    val = attrib(type=TrainConfig)

@attrs
class NameArgsConfig:
    name = attrib(type=str)
    args = attrib(type=dict)

@attrs
class CriterionConfig:
    name = attrib(type=str)
    background_weight = attrib(type=float)
    args = attrib(type=dict)

@attrs
class TrainingConfig:
    epochs = attrib(type=int)
    early_stopping_patience = attrib(type=int, default=50)


@attrs
class Config:
    lstm_model = attrib(type=LSTMConfig)
    data = attrib(type=DataConfig)
    optimizer = attrib(type=NameArgsConfig)
    lr_scheduler = attrib(type=NameArgsConfig)
    criterion = attrib(type=CriterionConfig)
    training = attrib(type=TrainingConfig)

    @classmethod
    def from_json(cls, config_path):
        with open(config_path) as config_file:
            config = json.load(config_file)
            return Config.from_dict(config)

    @classmethod
    def from_dict(cls, config):
        config["lstm_model"] = LSTMConfig(**config["lstm_model"])
        config["data"] = DataConfig(**config["data"])
        if config["data"].train:
            config["data"].train = TrainConfig(**config["data"].train)
            if config["data"].val:
                config["data"].val = TrainConfig(**config["data"].val)
        config["optimizer"] = NameArgsConfig(**config["optimizer"])
        config["lr_scheduler"] = NameArgsConfig(**config["lr_scheduler"])
        config["criterion"] = CriterionConfig(**config["criterion"])
        config["training"] = TrainingConfig(**config["training"])
        return cls(**config)
    

@attrs
class FineTuneConfig:
    lstm_model = attrib(type=LSTMConfig)
    masker_model = attrib(type=MaskerConfig)
    data = attrib(type=DataConfig)
    lstm_optimizer = attrib(type=NameArgsConfig)
    lstm_lr_scheduler = attrib(type=NameArgsConfig)
    masker_optimizer = attrib(type=NameArgsConfig)
    masker_lr_scheduler = attrib(type=NameArgsConfig)
    criterion = attrib(type=CriterionConfig)
    training = attrib(type=TrainingConfig)

    @classmethod
    def from_json(cls, config_path):
        with open(config_path) as config_file:
            config = json.load(config_file)
            return Config.from_dict(config)

    @classmethod
    def from_dict(cls, config):
        config["lstm_model"] = LSTMConfig(**config["lstm_model"])
        config["masker_model"] = LSTMConfig(**config["masker_model"])
        config["data"] = DataConfig(**config["data"])
        if config["data"].train:
            config["data"].train = TrainConfig(**config["data"].train)
            if config["data"].val:
                config["data"].val = TrainConfig(**config["data"].val)
        config["lstm_optimizer"] = NameArgsConfig(**config["lstm_optimizer"])
        config["lstm_lr_scheduler"] = NameArgsConfig(**config["lstm_lr_scheduler"])
        config["masker_optimizer"] = NameArgsConfig(**config["masker_optimizer"])
        config["masker_lr_scheduler"] = NameArgsConfig(**config["masker_lr_scheduler"])
        config["criterion"] = CriterionConfig(**config["criterion"])
        config["training"] = TrainingConfig(**config["training"])
        return cls(**config)