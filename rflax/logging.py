import tempfile
import uuid
import wandb
from etils.epath import Path, PathLike
from flax import struct
from socket import gethostname
from typing import Optional, Dict


def prefix_metrics(metrics, prefix):
  return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


@struct.dataclass
class LoggerConfig:
  online: bool = False
  project: str = "rflax"
  id: str = uuid.uuid4().hex[:8]
  output_dir: Optional[PathLike] = None
  notes: Optional[str] = None


class WandBLogger(object):
  @staticmethod
  def default_config() -> LoggerConfig:
    return LoggerConfig()

  def __init__(self, config: LoggerConfig, run_config: Dict):
    self.config = config

    if self.config.output_dir is None:
      self.config = self.config.replace(output_dir=tempfile.mkdtemp())
    else:
      out_dir = Path(self.config.output_dir).expanduser() / self.config.id
      self.config = self.config.replace(output_dir=out_dir)
      if out_dir.exists():
        out_dir.rmtree()
      out_dir.mkdir()

    self._run_config = run_config

    if "hostname" not in self._run_config:
      self._run_config["hostname"] = gethostname()

    self.run = wandb.init(
        reinit=True,
        config=self._run_config,
        project=self.config.project,
        dir=self.config.output_dir,
        id=self.config.id,
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
    return self.config.id

  @property
  def variant(self):
    return self._run_config

  @property
  def output_dir(self):
    return self.config.output_dir
