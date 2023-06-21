__version__ = "0.2.1"

import datetime
import platform
import shelve
import sys, os, inspect
import logging
import traceback
import warnings

import random
import omegaconf

from omegaconf import DictConfig, OmegaConf, ListConfig
from omegaconf.resolvers import oc
import re

from typing import Any, Optional

from omegaconf import Container, Node
from omegaconf._utils import _DEFAULT_MARKER_, _get_value
from omegaconf.basecontainer import BaseContainer
from omegaconf.errors import ConfigKeyError
from omegaconf.grammar_parser import *

from recursive_monkey_patch import monkey_patch

import hashlib

import hydra.core.plugins
import hydra._internal.config_loader_impl
from tables import PerformanceWarning
from scipy.optimize import OptimizeWarning
from . import _Plugins__fake_file_config_source
from . import __fake_plugins
from . import __fake_config_loader_impl
from .__gui_server import __file__ as gui_script_file

monkey_patch(__fake_plugins, hydra.core.plugins)
monkey_patch(__fake_config_loader_impl, hydra._internal.config_loader_impl)

ANIMATION_TYPES_SAVE_FORMATS = ["html", "mp4"]
ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK = [
    "playback"
] + ANIMATION_TYPES_SAVE_FORMATS
ANIMATION_TYPES_NONE = [None, "None"]
ANIMATION_TYPES_REQUIRING_ANIMATOR = [
    "live",
] + ANIMATION_TYPES_REQUIRING_SAVING_SCENARIO_PLAYBACK
ANIMATION_TYPES = ANIMATION_TYPES_NONE + ANIMATION_TYPES_REQUIRING_ANIMATOR

from . import controllers
from . import systems
from . import simulator
from . import systems
from .visualization import animator
from . import __utilities
from . import models
from . import predictors
from . import policies
from . import visualization
from .visualization import *
import mlflow
from unittest.mock import Mock, MagicMock
from hydra._internal.utils import _locate

from . import __instantiate as inst

mock = Mock()

import plotly.graph_objects as go
import json

import tempfile

from multiprocessing import Process
import numpy

# TO DO: NEED GLOBAL DOCSTRING FOR THIS MODULE AND ITS FUNCTIONALITY

def hash_string(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), base=16)

"""TO DO:
NEED COMMENT SECTION
DOES THIS METHOD DO?
"""
def __memorize_instance(resolver):
    def inner(
        key: str,
        default: Any = _DEFAULT_MARKER_,
        *,
        _parent_: Container,
    ):
        if default == _DEFAULT_MARKER_:
            default = key.strip()
        instance_name = str(default)
        if instance_name in main.objects_created:
            return main.objects_created[instance_name]
        else:
            obj = inst.instantiate(
                resolver(key, default=default, _parent_=_parent_), path=default
            )
            main.objects_created[instance_name] = obj
            return obj

    return inner


def __sub_map(pattern, f, s):
    def map_match(match):
        return f(match.group())

    return re.sub(pattern, map_match, s)

"""TO DO:
WHAT IS THIS?
DESCRIBE WHAT IT IS AND WHAT IT DOES
NEED COMMENT SECTION
"""
def obtain(obj_repr):
    if not isinstance(obj_repr, str):
        obj_repr = str(obj_repr)
    obj_repr = (
        obj_repr.replace("__QUOTATION__", '"')
        .replace("__APOSTROPHE__", "'")
        .replace("__TILDE__", "~")
    )
    obj_repr = _Plugins__fake_file_config_source.numerize_string(obj_repr)
    pattern = re.compile(r"(\A|[^a-zA-Z\._])[a-zA-Z_][a-zA-Z0-9_]*")
    resolved = []

    def resolve(s):
        if s[0].isalnum() or s[0] == "_":
            prefix = ""
        else:
            prefix = s[0]
            s = s[1:]
        try:
            entity = _locate(s)
        except:
            entity = eval(s)
        resolved.append(entity)
        return f"{prefix}resolved[{len(resolved) - 1}]"

    obj_repr_resolved_modules = __sub_map(pattern, resolve, obj_repr)
    return eval(obj_repr_resolved_modules)


OmegaConf.register_new_resolver("same", __memorize_instance(oc.select))
OmegaConf.register_new_resolver(name="get", resolver=obtain)
OmegaConf.register_new_resolver(name="mock", resolver=lambda: mock)

# TO DO: PLEASE ALL IMPORT INTO THE HEADER
from .__hydra_main import main as hydramain

# TO DO: DESCRIBE WHY THIS IS CALLED COMPLEMENTED. EXPLAIN THE IDEA BEHIND
class ComplementedConfig:
    """
    A config object, generated by ``rcognita``'s config pipeline.
    """

    def __init__(self, cfg, config_path=""):
        self.__hydra_config = cfg
        self.config_path = config_path
        self.__saved_hash = None

    # TO DO: NEED COMMENT SECTION
    def treemap(self, root="config"):
        def format(
            parent_name, parent_node, parent_node_raw, occupied=set(), parent_color=50
        ):
            labels = []
            parents = []
            colors = []
            text = []
            for name, value in parent_node.items():
                # Check if `parent_node` has attribute `value`
                parents.append(parent_name)
                # if type(parent_node_raw) is str:
                #    parent_node_raw = omegaconf.OmegaConf.to_container(parent_node.__hydra_config)
                if isinstance(value, ComplementedConfig):
                    while name in occupied:
                        name += " "
                    node_raw = parent_node_raw[name.strip()]
                    if type(node_raw) is str:
                        text.append(node_raw.replace("__IGNORE__", "%%"))
                        node_raw = omegaconf.OmegaConf.to_container(
                            value.__hydra_config
                        )
                    else:
                        text.append("")
                    labels.append(name)
                    colors.append((parent_color * 1.2) % 100)
                    (
                        subnode_parents,
                        subnode_labels,
                        subnode_colors,
                        subnode_text,
                    ) = format(
                        name,
                        value,
                        node_raw,
                        occupied=occupied,
                        parent_color=(parent_color * 1.2) % 100,
                    )
                    for i, subnode_label in enumerate(subnode_labels):
                        if subnode_label in labels or subnode_label in parents:
                            subnode_labels[i] = subnode_labels[i] + " "
                    labels += subnode_labels
                    parents += subnode_parents
                    colors += subnode_colors
                    text += subnode_text
                else:
                    real_name = name if name in parent_node_raw else name + "__IGNORE__"
                    if (
                        type(parent_node_raw[real_name]) is str
                        and "$" in parent_node_raw[real_name]
                    ):
                        text.append(
                            parent_node_raw[real_name].replace("__IGNORE__", "%%")
                        )
                    else:
                        text.append("")
                    colors.append(hash_string(name) % 100)
                    name = f"{name}: {str(value)}"
                    while name in occupied:
                        name += " "
                    labels.append(name)

                occupied.add(labels[-1])
            occupied.add(parent_name)
            return parents, labels, colors, text

            # TO DO: WHAT THIS COMMENT REFERS TO?
            # append attributes to root

        raw_self = omegaconf.OmegaConf.to_container(self.__hydra_config)
        self.__saved_hash = hash_string(json.dumps(raw_self, sort_keys=True))
        parents, labels, colors, text = format(
            f"{root} {hex(hash(self))}", self, raw_self
        )
        # parents = [parent[:-1] if "_" in parent else parent for parent in parents]
        # parents = [""] + parents
        # labels = ["config"] + labels
        fig = go.Figure(
            go.Treemap(
                labels=labels,
                parents=parents,
                marker=dict(colors=colors, colorscale="RdBu", cmid=50),
                text=text,
            )
        )
        return fig

    def refresh(self):
        """
        Reset all cached instances.
        Call this if you made changes to your config dynamically and would like those change to take
        effect in forthcoming instantiations.
        """
        main.objects_created.clear()

    def __contains__(self, item):
        return item in self.__hydra_config or item + "__IGNORE__" in self.__hydra_config

    def __getattr__(self, item):
        cfg = object.__getattribute__(self, "_ComplementedConfig__hydra_config")
        try:
            name = item
            attr = cfg.__getattr__(name)
        except (IndexError, KeyError, AttributeError, ConfigKeyError) as e:
            name = item + "__IGNORE__"
            attr = cfg.__getattr__(name)
        if isinstance(attr, DictConfig) or isinstance(attr, ListConfig):
            if self.config_path:
                child_config_path = self.config_path + "." + name
            else:
                child_config_path = name
            attr = ComplementedConfig(attr, config_path=child_config_path)
        return attr

    def __str__(self):
        return (
            str(self.__hydra_config)
            .replace("DictConfig", "ComplementedConfig")
            .replace("ListConfig", "ComplementedConfig")
            .replace("${get:", "={")
            .replace("${same:", "~{")
        )

    def __getitem__(self, key):
        cfg = object.__getattribute__(self, "_ComplementedConfig__hydra_config")
        try:
            name = key
            item = cfg[name]
        except (IndexError, KeyError, AttributeError, ConfigKeyError) as e:
            if isinstance(key, str):
                name = key + "__IGNORE__"
                item = cfg[name]
            else:
                raise e
        if isinstance(item, DictConfig) or isinstance(item, ListConfig):
            if self.config_path:
                child_config_path = self.config_path + "." + name
            else:
                child_config_path = name
            item = ComplementedConfig(item, config_path=child_config_path)
        return item

    def __setitem__(self, key, value):
        if key + "__IGNORE__" in self.__hydra_config:
            self.__hydra_config[key + "__IGNORE__"] = value
        else:
            self.__hydra_config[key] = value

    def __repr__(self):
        return self.__class__.__name__ + "( " + repr(self.__hydra_config) + ")"

    def __setattr__(self, key, value):
        if key in ["config_path", "_ComplementedConfig__saved_hash"]:
            object.__setattr__(self, key, value)
            return
        if hasattr(self, "_ComplementedConfig__hydra_config"):
            if key + "__IGNORE__" in self.__hydra_config:
                self.__hydra_config.__setattr__(key + "__IGNORE__", value)
            else:
                self.__hydra_config.__setattr__(key, value)
        else:
            object.__setattr__(self, key, value)

    def __hash__(self):
        if not self.__saved_hash:
            real_config = omegaconf.OmegaConf.to_container(self.__hydra_config)
            return hash_string(json.dumps(real_config, sort_keys=True))
        else:
            return self.__saved_hash

    def copy(self):
        return ComplementedConfig(
            self.__hydra_config.copy(), config_path=self.config_path
        )

    def has_key(self, key):
        return key in self.__hydra_config or key + "__IGNORE__" in self.__hydra_config

    def clear(self):
        self.__hydra_config.clear()

    def update(self, *args, **kwargs):
        for subdict in args:
            for key, value in subdict.values():
                self[key] = value
        for key, value in kwargs.values():
            self[key] = value

    def keys(self):
        return [key.replace("__IGNORE__", "") for key in self.__hydra_config.keys()]

    def items(self):
        return [
            (
                key.replace("__IGNORE__", ""),
                value
                if not isinstance(value, DictConfig)
                and not isinstance(value, ListConfig)
                else ComplementedConfig(
                    value,
                    config_path=key
                    if not self.config_path
                    else f"{self.config_path}.{key}",
                ),
            )
            for key, value in self.__hydra_config.items()
        ]

    def values(self):
        return [
            (
                value
                if not isinstance(value, DictConfig)
                and not isinstance(value, ListConfig)
                else ComplementedConfig(
                    value,
                    config_path=key
                    if not self.config_path
                    else f"{self.config_path}.{key}",
                )
            )
            for key, value in self.__hydra_config.items()
        ]

    def __delitem__(self, key):
        if key + "__IGNORE__" in self.__hydra_config:
            del self.__hydra_config[key + "__IGNORE__"]
        else:
            del self.__hydra_config[key]

    # TO DO: NOT QUITE CLEAR WHAT IT DOES AND IN WHICH CONTEXT
    def __invert__(self):
        """
        Instantiate the object described by the config (recursively).

        ``_target_`` has to be specified.
        """
        # if not self.config_path:
        return inst.instantiate(self.__hydra_config, path=self.config_path)
        # else:
        #     instance_name = self.config_path.strip()
        #     if instance_name in self.objects_created:
        #         return self.objects_created[instance_name]
        #     else:
        #         res = inst.instantiate(self.__hydra_config, path=self.config_path)
        #         self.objects_created[instance_name] = res
        #         return res


from .callbacks import *


class RcognitaExitException(Exception):
    pass

# TO DO: ADD 2-3 GENERAL SENTENCES OF WHAT THIS AND WHAT ITS FUNCTIONALITY ARE
class main:
    """
    The decorator used to invoke ``rcognita``'s config pipeline.

    Use it to decorate your ``main`` function like so:
    ::

         @rcognita.main(config_path=..., config_name=...)
         def my_main(config):
             ...

         if __name__ == '__main__':
             my_main()
    """

    callbacks = None
    logger = None
    metadata = None
    config = None
    assignments = []
    weak_assignments = []
    builtin_callbacks = [
        OnEpisodeDoneCallerCallback,
        OnIterationDoneCallerCallback,
        TimeCallback,
        ConfigDiagramCallback,
        TimeRemainingCallback,
        SaveProgressCallback,
    ]
    objects_created = {}

    # TO DO: WHAT IS WEAK ASSIGNMENT?
    @classmethod
    def post_weak_assignment(cls, key, value):
        cls.weak_assignments.append((key, value))

    # TO DO: WHAT IS THIS?
    @classmethod
    def post_assignment(cls, key, value, weak=False):
        if weak:
            cls.post_weak_assignment(key, value)
        else:
            cls.assignments.append((key, value))

    # TO DO: DOCSTRING
    @classmethod
    def apply_assignments(cls, cfg):
        for key, value in cls.weak_assignments:
            if callable(value):
                value = value(cfg)
            current = eval(f"cfg.{key.replace('%%', '')}")
            if current == "__REPLACE__":
                exec(f"cfg.{key.replace('%%', '')} = value")
        for key, value in cls.assignments:
            if callable(value):
                value = value(cfg)
            exec(f"cfg.{key.replace('%%', '')} = value")

    # TO DO: DOCSTRING TO SHORT AND UNINFORMATIVE
    def __init__(
        self,
        *args,
        logger=logging.getLogger("rcognita"),
        callbacks=[],
        **kwargs,
    ):
        """
        Create an instance of ``rcognita.main``.

        :param config_path: path to the folder containing your config(s)
        :type config_path: str
        :param config_name: the name of your config (without extension)
        :type config_name: str
        :param callbacks: list of uninstantiated callbacks (classes) to be registered
        :type callbacks: list[class], optional
        :param logger: a logger instance to be supplied to callbacks
        :type logger: Logger, optional

        """
        # os.environ["PYTHONHASHSEED"] = "0"
        callbacks = callbacks + self.builtin_callbacks
        sys.argv.insert(1, "--multirun")
        sys.argv.insert(-1, "hydra.job.chdir=True")
        self.mlflow_uri = f"file://{os.getcwd()}/mlruns"
        self.tags = {}
        self.experiment_name = "Default"
        to_remove = []
        for i, arg in enumerate(sys.argv):
            if "--experiment" in arg:
                self.experiment_name = arg.split("=")[1]
                to_remove.append(i)
            if "--tags" in arg:
                tags_str = arg.split("=")[1].split(",")
                self.tags = {tag.split(":")[0]: tag.split(":")[1] for tag in tags_str}
                to_remove.append(i)
        for i in to_remove[::-1]:
            sys.argv.pop(i)
        if "--disable-logging" in sys.argv:
            sys.argv.pop(sys.argv.index("--disable-logging"))
            sys.argv.insert(-1, "hydra/job_logging=disabled")
        if "--disable-callbacks" in sys.argv:
            sys.argv.pop(sys.argv.index("--disable-callbacks"))
            self.__class__.builtin_callbacks = []
            callbacks = []
            self.callbacks_enabled = False
        else:
            self.callbacks_enabled = True
        if "--enable-streamlit" in sys.argv:
            sys.argv.pop(sys.argv.index("--enable-streamlit"))
            self.streamlit_enabled = True
        else:
            self.streamlit_enabled = False
        self.cooldown_factor = 1.0
        for i, arg in enumerate(sys.argv):
            if "--cooldown-factor" in arg:
                self.cooldown_factor = float(arg.split("=")[-1])
                sys.argv.pop(i)
                break
        if not "--single-thread" in sys.argv:
            sys.argv.insert(-1, "hydra/launcher=joblib")
        else:
            sys.argv.pop(sys.argv.index("--single-thread"))
        if "--sweep" in sys.argv:
            sys.argv.insert(-1, "hydra/sweeper=ax")
            sys.argv.pop(sys.argv.index("--sweep"))
            self.is_sweep = True
        else:
            self.is_sweep = False

        self.__class__.is_clear_matplotlib_cache_in_callbacks = True
        for arg in sys.argv:
            if "live" in arg:
                self.__class__.is_clear_matplotlib_cache_in_callbacks = False

        self.args = args
        self.kwargs = kwargs
        self.kwargs["version_base"] = (
            None if "version_base" not in kwargs else kwargs["version_base"]
        )
        self.kwargs["config_path"] = (
            "." if "config_path" not in kwargs else kwargs["config_path"]
        )
        self.__class__.logger = logger
        self.__class__.callbacks = [callback() for callback in callbacks]

    # TO DO: DESCRIBE WHAT THIS DOES
    def __call__(self, old_app):
        def rcognita_main(*args, **kwargs):
            common_dir = tempfile.TemporaryDirectory()
            initial_working_directory = os.getcwd()
            try:
                initial_pythonpath = os.environ["PYTHONPATH"]
            except KeyError:
                initial_pythonpath = initial_working_directory
            script_path = inspect.getfile(old_app)
            path_main = os.path.abspath(script_path)
            path_parent = "/".join(path_main.split("/")[:-1])
            os.chdir(path_parent)
            path = os.path.abspath(self.kwargs["config_path"])
            self.kwargs["config_path"] = path

            def app(
                cfg,
                callbacks=self.__class__.callbacks,
                logger=self.__class__.logger,
                is_clear_matplotlib_cache_in_callbacks=self.__class__.is_clear_matplotlib_cache_in_callbacks,
            ):
                self.__class__.is_clear_matplotlib_cache_in_callbacks = (
                    is_clear_matplotlib_cache_in_callbacks
                )
                self.__class__.logger = logger
                mlflow.set_tracking_uri(self.mlflow_uri)
                if "seed" in cfg:
                    seed = cfg["seed"]
                    delattr(cfg, "seed")
                else:
                    seed = 0
                numpy.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                with shelve.open(".report") as f:
                    f["termination"] = "running"
                    f["started"] = datetime.datetime.now()
                    f["args"] = sys.argv
                    f["config_path"] = path
                    f["script_path"] = script_path
                    f["PYTHONPATH"] = initial_pythonpath
                    f["initial_cwd"] = initial_working_directory
                    f["common_dir"] = common_dir.name
                    f["hostname"] = platform.node()
                    f["seed"] = seed
                    started = f["started"]
                os.mkdir("gfx")
                os.mkdir(".callbacks")
                with omegaconf.flag_override(cfg, "allow_objects", True):
                    self.__class__.metadata = {
                        "script_path": script_path,
                        "config_path": path + f"/{self.kwargs['config_name']}.yaml",
                        "initial_working_directory": initial_working_directory,
                        "initial_pythonpath": initial_pythonpath,
                        "common_dir": common_dir.name,
                        "id": int(os.getcwd().split("/")[-1]),
                        "report": lambda: shelve.open(
                            common_dir.name
                            + "/report_"
                            + os.getcwd().split("/")[-1].zfill(5)
                        ),
                        "pid": os.getpid(),
                    }
                    ccfg = ComplementedConfig(cfg)
                    self.apply_assignments(ccfg)
                    if "callbacks" in cfg and self.callbacks_enabled:
                        for callback in cfg.callbacks:
                            callback = (
                                obtain(callback)
                                if isinstance(callback, str)
                                else callback
                            )
                            callbacks.insert(-1, callback())
                        delattr(cfg, "callbacks")
                    elif "callbacks" in cfg:
                        delattr(cfg, "callbacks")
                    self.__class__.callbacks = callbacks
                    self.__class__.config = ccfg
                    try:
                        for callback in self.__class__.callbacks:
                            if callback.cooldown:
                                callback.cooldown *= self.cooldown_factor
                            callback.on_launch()  # TODO: make sure this line is adequate to mlflow functionality
                        with self.__class__.metadata["report"]() as r:
                            r["path"] = os.getcwd()
                            r["pid"] = os.getpid()
                        self.tags.update({"run_path": os.getcwd()})
                        experiment_id = mlflow.set_experiment(
                            self.experiment_name
                        ).experiment_id
                        with mlflow.start_run(
                            experiment_id=experiment_id,
                            tags=self.tags,
                            run_name=" ".join(os.getcwd().split("/")[-3:]),
                        ):
                            overrides = {
                                line.split("=")[0]
                                .replace("+", "")
                                .replace("/", "-"): line.split("=")[1]
                                for line in OmegaConf.load(".hydra/overrides.yaml")
                            }
                            mlflow.log_params(overrides)
                            mlflow.log_artifact("SUMMARY.html")
                            mlflow.log_artifact(".summary")
                            mlflow.log_artifact(".hydra")
                            res = old_app(ccfg)
                            mlflow.log_artifact(".callbacks")
                            mlflow.log_artifact("__init__.log")
                            if os.path.exists("callbacks.dill"):
                                mlflow.log_artifact("callbacks.dill")

                        self.__class__.callbacks[0].log(
                            "Script terminated successfully."
                        )
                    except RcognitaExitException as e:
                        res = e
                    except Exception as e:
                        with self.__class__.metadata["report"]() as r:
                            r["traceback"] = traceback.format_exc()
                        res = e
                        self.__class__.callbacks[0].log("Script terminated with error.")
                        self.__class__.callbacks[0].exception(e)
                    for callback in self.__class__.callbacks:
                        try:
                            callback.on_termination(res)
                        except Exception as e:
                            callback.log(
                                f"Termination procedure for {callback.__class__.__name__} failed."
                            )
                            callback.exception(e)
                    with shelve.open(".report") as f:
                        f["termination"] = (
                            "successful"
                            if not isinstance(res, Exception)
                            else "".join(traceback.format_tb(res.__traceback__))
                        )
                        f["finished"] = datetime.datetime.now()
                    ccfg.refresh()
                    if self.is_sweep:
                        return res
                    else:
                        return {
                            "result": res,
                            "callbacks": self.__class__.callbacks,
                            "directory": os.getcwd(),
                        }

            # TO DO: DOCSTRING
            def gui_server():
                import streamlit.web.bootstrap
                from streamlit import config as _config

                _config.set_option("server.headless", True)
                args = [common_dir.name]

                # streamlit.cli.main_run(filename, args)
                streamlit.web.bootstrap.run(gui_script_file, "", args, flag_options={})

            gui = Process(target=gui_server) if self.streamlit_enabled else Mock()
            gui.start()
            app.__module__ = old_app.__module__
            res = hydramain(*self.args, **self.kwargs)(app)(*args, **kwargs)
            common_dir.cleanup()
            time.sleep(2.0)
            gui.terminate()
            return res

        return rcognita_main


warnings.filterwarnings("ignore", category=UserWarning, module=hydra.__name__)
warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

array = __utilities.rc.array
