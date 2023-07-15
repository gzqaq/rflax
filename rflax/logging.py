import os
import pprint
import tempfile
import uuid
import wandb
from absl import flags, logging
from copy import copy
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict
from socket import gethostname


def define_flags_with_default(**kwargs):
  for key, val in kwargs.items():
    if isinstance(val, ConfigDict):
      config_flags.DEFINE_config_dict(key, val)
    elif isinstance(val, bool):
      flags.DEFINE_bool(key, val, "automatically defined flag")
    elif isinstance(val, int):
      flags.DEFINE_integer(key, val, "automatically defined flag")
    elif isinstance(val, float):
      flags.DEFINE_float(key, val, "automatically defined flag")
    elif isinstance(val, str):
      flags.DEFINE_string(key, val, "automatically defined flag")
    else:
      raise ValueError("Incorrect value type")

  return kwargs


def print_flags(flags, flags_def):
  logging.info("Running training with hyperparameters: \n{}".format(
      pprint.pformat([
          "{}: {}".format(key, val)
          for key, val in get_user_flags(flags, flags_def).items()
      ])))


def get_user_flags(flags, flags_def):
  output = {}
  for key in flags_def:
    val = getattr(flags, key)
    if isinstance(val, ConfigDict):
      output.update(flatten_config_dict(val, prefix=key))
    else:
      output[key] = val

  return output


def flatten_config_dict(config, prefix=None):
  output = {}
  for key, val in config.items():
    if prefix is not None:
      next_prefix = "{}.{}".format(prefix, key)
    else:
      next_prefix = key
    if isinstance(val, ConfigDict):
      output.update(flatten_config_dict(val, prefix=next_prefix))
    elif isinstance(val, dict):
      output.update(flatten_config_dict(ConfigDict(val), prefix=next_prefix))
    else:
      output[next_prefix] = val
  return output


def prefix_metrics(metrics, prefix):
  return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


class WandBLogger(object):
  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.online = False
    config.prefix = "rflax"
    config.project = "jax"
    config.output_dir = "/tmp/rflax"
    config.name = config_dict.placeholder(str)
    config.anonymous = config_dict.placeholder(str)
    config.notes = config_dict.placeholder(str)

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, variant):
    self.config = self.get_default_config(config)

    if self.config.name is None:
      self.config.name = uuid.uuid4().hex[:8]

    if self.config.prefix != "":
      self.config.project = f"{self.config.prefix}-{self.config.project}"

    if self.config.output_dir == "":
      self.config.output_dir = tempfile.mkdtemp()
    else:
      self.config.output_dir = os.path.join(self.config.output_dir,
                                            self.config.name)
      os.makedirs(self.config.output_dir, exist_ok=True)

    self._variant = copy(variant)

    if "hostname" not in self._variant:
      self._variant["hostname"] = gethostname()

    self.run = wandb.init(
        reinit=True,
        config=self._variant,
        project=self.config.project,
        dir=self.config.output_dir,
        name=self.config.name,
        anonymous=self.config.anonymous,
        notes=self.config.notes,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
        ),
        mode="online" if self.config.online else "offline",
    )

  def log(self, *args, **kwargs):
    self.run.log(*args, **kwargs)

  @property
  def run_name(self):
    return self.config.name

  @property
  def variant(self):
    return self._variant

  @property
  def output_dir(self):
    return self.config.output_dir
