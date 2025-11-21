import os
import asyncio
import logging
from datetime import datetime
from pkgutil import get_data
from omegaconf import OmegaConf
from typing import Literal

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl import (
    arithmetic_env,
    math_env,
    verl_env
)
from tinker_cookbook.rl.train import (
    AsyncConfig, Config, main, StreamMinibatchConfig
)
from tinker_cookbook.rl.types import RLDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Simple command-line configuration for RL training."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    env: str = "arithmetic"  # Options: arithmetic, math, polaris, deepmath, gsm8k
    seed: int = 0  # Random seed for data shuffling

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 5
    kl_penalty_coef: float = 0.0

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals
    eval_every: int = 20

    # Checkpointing
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None


def get_dataset_builder(
    config
) -> RLDatasetBuilder:
    env = config.env
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    if env == "arithmetic":
        return arithmetic_env.ArithmeticDatasetBuilder(
            batch_size=config.groups_per_batch,
            model_name_for_tokenizer=config.model_name,
            renderer_name=renderer_name,
            n_batches=100,
            include_fewshot=True,
            group_size=config.group_size,
        )
    elif env in ["math", "polaris", "deepmath", "gsm8k"]:
        return math_env.get_math_dataset_builder(
            dataset_name=config.env,
            batch_size=config.groups_per_batch,
            model_name_for_tokenizer=config.model_name,
            renderer_name=renderer_name,
            group_size=config.group_size,
        )
    elif env == "verl_env":
        return verl_env.VerlDatasetBuilder(
            train_files=config.train_files,
            test_files=config.test_files,
            reward_path=config.reward_path,
            reward_function_name=config.reward_function_name,
            batch_size=config.groups_per_batch,
            model_name_for_tokenizer=config.model_name,
            group_size=config.group_size,
        )
    else:
        raise ValueError(f"Unknown environment: {env}")


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get tokenizer for stop sequences
    model_name = cli_config.model_name.replace("/", "-")
    run_name = f"{cli_config.env}-{model_name}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.group_size}group-{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    # create log path if it doesn't exist
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/math_rl/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name
    # print(f"{cli_config.datasets_paths=}")
    # Create full config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(cli_config),
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await main(config)

async def hydra_main():
    """Convert CLI config to full config and run training."""
    cli_config = OmegaConf.load("config/default.yaml")

    # Get tokenizer for stop sequences
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_name = cli_config.model_name.replace("/", "-")
    run_name = f"{cli_config.env}-{model_name}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.group_size}group-{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    # create log path if it doesn't exist
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/math_rl/{run_name}"

    print(f"{log_path=}")
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name
    # Create full config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(cli_config),
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=(AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
            )
            if cli_config.max_steps_off_policy is not None
            else None
        ),
        stream_minibatch_config=(StreamMinibatchConfig(
                groups_per_batch=cli_config.groups_per_batch,
                num_minibatches=cli_config.num_minibatches,
            ) if cli_config.get("num_minibatches", None) is not None
            else None
        ),
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await main(config)

if __name__ == "__main__":
    asyncio.run(hydra_main())
    # cli_config = chz.entrypoint(CLIConfig)
    # asyncio.run(cli_main(cli_config))