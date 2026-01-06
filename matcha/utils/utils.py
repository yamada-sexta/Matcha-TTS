import os
import sys
import warnings
from importlib.util import find_spec
from math import ceil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

import gdown
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import wget
from matplotlib.figure import Figure
from omegaconf import DictConfig

T = TypeVar("T")

from matcha.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb 

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: str) -> Union[float, None]:
    """Safely retrieves value of a metric.

    :param metric_dict: A dict containing metric values.
    :param metric_name: The name of the metric to retrieve.
    :return: The value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise ValueError(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure the metric name is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name]
    if hasattr(metric_value, 'item'):
        metric_value = metric_value.item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def intersperse(lst: List[T], item: T) -> List[T]:
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def save_figure_to_numpy(fig: Figure) -> npt.NDArray[np.uint8]:
    """Return an HxWx3 uint8 numpy array from a Matplotlib Figure.

    This function tries several methods to extract RGB image bytes from the
    figure canvas to be robust across Matplotlib/backends/versions where
    certain convenience methods might be missing.

    Order tried:
      1. canvas.tostring_rgb() -> direct RGB bytes
      2. canvas.tostring_argb() -> convert ARGB to RGB
      3. canvas.buffer_rgba() -> buffer-like object (converted with numpy)
      4. PIL fallback using renderer.buffer_rgba()

    Raises RuntimeError if none of the strategies work.
    """
    # Ensure drawing has happened so the buffer is available
    try:
        fig.canvas.draw()
    except Exception:
        # If draw fails, let subsequent methods attempt to access the buffer
        pass

    canvas = fig.canvas
    width, height = canvas.get_width_height()

    # 1) Preferred: direct RGB bytes
    if hasattr(canvas, "tostring_rgb"):
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        try:
            return data.reshape((height, width, 3))
        except Exception:
            # fall through to other methods
            pass

    # 2) If only ARGB available, convert to RGB
    if hasattr(canvas, "tostring_argb"):
        argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        try:
            argb = argb.reshape((height, width, 4))
            # ARGB -> RGB (drop alpha, reorder)
            rgb = argb[:, :, [1, 2, 3]]
            return rgb
        except Exception:
            pass

    # 3) buffer_rgba (returns a buffer-like or array-like RGBA)
    if hasattr(canvas, "buffer_rgba"):
        try:
            buf = canvas.buffer_rgba()
            arr = np.asarray(buf)
            # Expect shape (H, W, 4) or (H*W*4,)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                return arr[:, :, :3].copy()
            if arr.size == width * height * 4:
                arr = arr.reshape((height, width, 4))
                return arr[:, :, :3]
        except Exception:
            pass

    # 4) PIL fallback using renderer.buffer_rgba()
    try:
        from PIL import Image

        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
        if hasattr(renderer, "buffer_rgba"):
            raw = renderer.buffer_rgba()
            arr = np.asarray(raw)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                return arr[:, :, :3].copy()
            if arr.size == width * height * 4:
                arr = arr.reshape((height, width, 4))
                return arr[:, :, :3]
        # Last ditch: create PIL image from tostring_argb if available
        if hasattr(canvas, "tostring_argb"):
            argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape((height, width, 4))
            rgb = argb[:, :, [1, 2, 3]]
            return rgb
    except Exception:
        pass

    raise RuntimeError("Unable to extract RGB image from Matplotlib figure canvas")


def plot_tensor(tensor: Union[torch.Tensor, npt.ArrayLike]) -> npt.NDArray[np.uint8]:
    plt.style.use("default")  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation="none")  # type: ignore[arg-type]
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor: Union[torch.Tensor, npt.ArrayLike], savepath: Union[str, Path]) -> None:
    plt.style.use("default")  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation="none")  # type: ignore[arg-type]
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()


def to_numpy(
    tensor: Union[npt.NDArray[Any], torch.Tensor, List[Any]]
) -> npt.NDArray[Any]:
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return np.array(tensor)
    else:
        raise TypeError("Unsupported type for conversion to numpy array")


def get_user_data_dir(appname: str = "matcha_tts") -> Path:
    """
    Args:
        appname (str): Name of application

    Returns:
        Path: path to user data directory
    """

    MATCHA_HOME = os.environ.get("MATCHA_HOME")
    if MATCHA_HOME is not None:
        ans = Path(MATCHA_HOME).expanduser().resolve(strict=False)
    elif sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")

    final_path = ans.joinpath(appname)
    final_path.mkdir(parents=True, exist_ok=True)
    return final_path


def assert_model_downloaded(
    checkpoint_path: Union[str, Path], url: str, use_wget: bool = True
) -> None:
    if Path(checkpoint_path).exists():
        log.debug(f"[+] Model already present at {checkpoint_path}!")
        print(f"[+] Model already present at {checkpoint_path}!")
        return
    log.info(f"[-] Model not found at {checkpoint_path}! Will download it")
    print(f"[-] Model not found at {checkpoint_path}! Will download it")
    checkpoint_path = str(checkpoint_path)
    if not use_wget:
        gdown.download(url=url, output=checkpoint_path, quiet=False, fuzzy=True)
    else:
        wget.download(url=url, out=checkpoint_path)


def get_phoneme_durations(
    durations: List[int], phones: List[str]
) -> List[Dict[str, Dict[str, Any]]]:
    prev = durations[0]
    merged_durations: List[int] = []
    # Convolve with stride 2
    for i in range(1, len(durations), 2):
        if i == len(durations) - 2:
            # if it is last take full value
            next_half = durations[i + 1]
        else:
            next_half = ceil(durations[i + 1] / 2)

        curr = prev + durations[i] + next_half
        prev = durations[i + 1] - next_half
        merged_durations.append(curr)

    assert len(phones) == len(merged_durations)
    assert len(merged_durations) == (len(durations) - 1) // 2

    cumsum_durations = torch.cumsum(torch.tensor(merged_durations), 0, dtype=torch.long)
    start = torch.tensor(0)
    duration_json: List[Dict[str, Dict[str, Any]]] = []
    for i, dur in enumerate(cumsum_durations):
        duration_json.append(
            {
                phones[i]: {
                    "starttime": start.item(),
                    "endtime": dur.item(),
                    "duration": dur.item() - start.item(),
                }
            }
        )
        start = dur

    assert list(duration_json[-1].values())[0]["endtime"] == sum(
        durations
    ), f"{list(duration_json[-1].values())[0]['endtime'],  sum(durations)}"
    return duration_json
