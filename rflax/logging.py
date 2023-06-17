import pprint
from absl import flags, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags


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
    else:
      output[next_prefix] = val
  return output


def prefix_metrics(metrics, prefix):
  return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}
