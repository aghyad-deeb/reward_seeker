from functools import partial
from typing import Sequence
import sys
import os
import importlib

import tinker
import pandas as pd
import chz
import numpy as np
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
    """
    A toy environment for solving addition problems.
    """

    def __init__(
        self,
        row: dict,
        reward_path: str,
        reward_function_name: str,
        model_name: str = "",
    ):
        renderer = model_info.get_recommended_renderer_name(model_name)
        self.reward = self.get_reward_fn(reward_path, reward_function_name)
        self.row = row
        super().__init__(renderer)
    
    def get_reward_fn(reward_path, reward_function_name):
        module = sys.modules.get("cusom_module", None)
        if module is None:
            if not os.path.exists(reward_path):
                return FileNotFoundError(
                    f"Reward Function Path {reward_path=} not found"
                )
            spec = importlib.util.spec_form_file_location(
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

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        convo = self.row["prompt"]
        return self.renderer.build_generation_prompt(convo), self.stop_condition
    
    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        correct_format = parse_success
        reward_dict = self.reward(
            data_source=self.row["data_source"],
            solution_str=message["content"],
            ground_truth=self.row["ground_truth"],
            extra_info=self.row["extra_info"]
        )
        total_reward = reward_dict["score"]
        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                **reward_dict,
            },
        )


class VerlDataset(RLDataset):
    def __init__(
        self,
        file_path: str,
        reward_path: str,
        reward_function_name: str,
        batch_size: int,
        group_size: int,
        n_batches: int = 100,
        include_fewshot: bool = True,
        model_name: str = ""
    ):
        self._rng = np.random.RandomState(None)
        self.reward_path = reward_path
        self.reward_function_name = reward_function_name
        self.batch_size = batch_size
        self.group_size = group_size
        self.n_batches = n_batches
        self.include_fewshot = include_fewshot
        self.model_name=model_name, 
        df = pd.read_parquet(file_path)
        self.rows = [row.to_dict() for _, row in df.iterrows()]
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        row = self.rows[index]
        return [self._make_env_group_builder(row) for _ in range(self.batch_size)]

    def _make_env_group_builder(self, row: dict) -> ProblemGroupBuilder:
        return ProblemGroupBuilder(
            env_thunk=partial(
                VerlEnv,
                reward_path=self.reward_path,
                reward_function_name=self.reward_function_name,
                row=row,
                model_name=self.model_name
            ),
            num_envs=self.group_size,
        )

    def __len__(self) -> int:
        return self.n_batches

@chz.chz
class VerlDatasetBuilder(RLDatasetBuilder):
    file_path: str
    reward_path: str
    reward_function_name: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    n_batches: int
    group_size: int
    include_fewshot: bool = True

    async def __call__(self) -> tuple[VerlDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return VerlDataset(
            file_path=self.file_path,
            reward_path=self.reward_path,
            reward_function_name=self.reward_function_name,
            batch_size=self.batch_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            n_batches=self.n_batches,
            include_fewshot=self.include_fewshot,
            group_size=self.group_size,
            model_name=self.model_name_for_tokenizer,
        ), None