from functools import partial
from typing import Sequence, Callable
import sys
import os
import importlib

import tinker
import pandas as pd
import chz
import numpy as np
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook import renderers
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    Env,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

class VerlEnv(Env):

    def __init__(
        self,
        row: dict,
        reward_fn: Callable,
        model_name: str = "",
    ):
        self.model_name = model_name
        tokenizer = get_tokenizer(self.model_name)
        self.renderer = renderers.get_renderer(
            model_info.get_recommended_renderer_name(model_name),
            tokenizer=tokenizer,
        )
        self.reward_fn = reward_fn
        self.row = row
    
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        convo = self.row["prompt"]
        ret = (
            self.renderer.build_generation_prompt(convo),
            self.renderer.get_stop_sequences()
        )
        return ret

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        correct_format = parse_success
        reward_dict = self.reward_fn(
            data_source=self.row["data_source"],
            solution_str=message["content"],
            ground_truth=self.row["ground_truth"],
            extra_info=self.row["extra_info"]
        )
        total_reward = reward_dict["score"]
        # total_reward = 0
        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics={
                "format": correct_format,
                **reward_dict,
            },
        )


class VerlDataset(RLDataset):
    def __init__(
        self,
        datasets_paths: list[str],
        reward_path: str,
        reward_function_name: str,
        batch_size: int,
        group_size: int,
        model_name: str = "",
        shuffle = True,
    ):
        self._rng = np.random.RandomState(None)
        self.reward_fn = self.get_reward_fn(reward_path, reward_function_name)
        self.batch_size = batch_size
        self.group_size = group_size
        self.model_name=model_name
        datasets_paths = [
            os.path.expanduser(
                os.path.expandvars(
                    path
                )
            ) for path in datasets_paths
        ]
        dfs = [pd.read_parquet(dataset_path) for dataset_path in datasets_paths]
        df = pd.concat(dfs)
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        self.rows = [row.to_dict() for _, row in df.iterrows()]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        row = self.rows[index]
        return [self._make_env_group_builder(row) for _ in range(self.batch_size)]

    def get_reward_fn(self, reward_path, reward_function_name):
        reward_path = os.path.expanduser(reward_path)
        reward_path = os.path.expandvars(reward_path)
        module = sys.modules.get("cusom_module", None)
        if module is None:
            if not os.path.exists(reward_path):
                raise FileNotFoundError(
                    f"Reward Function Path {reward_path=} not found"
                )
            spec = importlib.util.spec_from_file_location(
                "custom_module", reward_path
            )
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            try:
                sys.modules["custom_module"] = module
                assert spec.loader is not None
                spec.loader.exec_module(module)
            except Exception as e:
                raise RuntimeError(
                    f"Error loading reward module form '{reward_path=}': {e=}"
                )
        if not hasattr(module, reward_function_name):
            raise AttributeError(
                f"Error loading reward function {reward_function_name=} from module in {module.__file__=}"
            )
        return getattr(module, reward_function_name)


    def _make_env_group_builder(self, row: dict) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                VerlEnv,
                reward_fn=self.reward_fn,
                row=row,
                model_name=self.model_name
            ),
            num_envs=self.group_size,
        )

    def __len__(self) -> int:
        return len(self.rows)

@chz.chz
class VerlDatasetBuilder(RLDatasetBuilder):
    train_files: str
    test_files: str
    reward_path: str
    reward_function_name: str
    batch_size: int
    model_name_for_tokenizer: str
    group_size: int

    async def __call__(self) -> tuple[VerlDataset, None]:
        return (
            VerlDataset(
                datasets_paths=files,
                reward_path=self.reward_path,
                reward_function_name=self.reward_function_name,
                batch_size=self.batch_size,
                group_size=self.group_size,
                model_name=self.model_name_for_tokenizer,
            ) for files in [self.train_files, self.test_files]
        )