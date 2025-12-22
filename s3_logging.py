
# %%
# Load S3 credentials from ~/.env
import os
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

# Inspect AI imports for logging
from inspect_ai.log import EvalLog, EvalSpec, EvalSample, EvalPlan, EvalResults, EvalStats, write_eval_log
from inspect_ai.log._log import EvalDataset, EvalConfig, EvalScore, EvalMetric, EvalScorer, EvalMetricDefinition
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool
from inspect_ai.model._model_output import ModelOutput, ModelUsage, ChatCompletionChoice
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.scorer._metric import Score
import shortuuid

# S3 bucket configuration
S3_BUCKET = "rewardseeker"
S3_LOG_PREFIX = "logs/chat_sessions"


def conv_to_inspect_messages(conv: list) -> list:
    """Convert conversation format to Inspect AI ChatMessage format."""
    messages = []
    for msg in conv:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            messages.append(ChatMessageSystem(content=content))
        elif role == "user":
            messages.append(ChatMessageUser(content=content))
        elif role == "assistant":
            messages.append(ChatMessageAssistant(content=content))
        elif role == "tool":
            messages.append(ChatMessageTool(content=content, function="bash"))
        else:
            raise ValueError(f"Unknown role: {role}")
    
    return messages


def create_inspect_log(
    conv: list,
    model_id: str,
    experiment_name: str,
    date: str,
    random_seed: int,
    temperature: float = 1.0,
    n_tokens: int = 10000,
) -> EvalLog:
    """Create an Inspect AI EvalLog from a conversation."""
    from datetime import datetime
    
    # Generate unique IDs
    eval_id = shortuuid.uuid()
    run_id = shortuuid.uuid()
    
    # Convert conversation to Inspect messages
    messages = conv_to_inspect_messages(conv)
    
    # Get the last assistant message as the completion (if any)
    last_assistant_content = ""
    for msg in reversed(conv):
        if msg["role"] == "assistant":
            last_assistant_content = msg["content"]
            break
    
    # Define scorers that will create columns in Inspect View
    eval_scorers = [
        EvalScorer(
            name="message_count",
            options={},
            metrics=[EvalMetricDefinition(name="inspect_ai/mean", options={})],
            metadata={},
        ),
        EvalScorer(
            name="user_messages",
            options={},
            metrics=[EvalMetricDefinition(name="inspect_ai/mean", options={})],
            metadata={},
        ),
        EvalScorer(
            name="assistant_messages",
            options={},
            metrics=[EvalMetricDefinition(name="inspect_ai/mean", options={})],
            metadata={},
        ),
        EvalScorer(
            name="tool_messages",
            options={},
            metrics=[EvalMetricDefinition(name="inspect_ai/mean", options={})],
            metadata={},
        ),
    ]
    
    # Create EvalSpec
    eval_spec = EvalSpec(
        eval_id=eval_id,
        run_id=run_id,
        created=date,
        task=f"{experiment_name}",
        task_id=f"{experiment_name}",
        task_version=1,
        task_attribs={"experiment_name": experiment_name},
        task_args={"random_seed": random_seed, "temperature": temperature, "n_tokens": n_tokens},
        task_args_passed={},
        dataset=EvalDataset(name=experiment_name, samples=1, sample_ids=[1]),
        model=model_id,
        model_generate_config=GenerateConfig(
            temperature=temperature,
            max_tokens=n_tokens,
            seed=random_seed,
        ),
        model_args={},
        config=EvalConfig(),
        packages={},
        metadata={
            "experiment_name": experiment_name,
            "random_seed": random_seed,
            "session_date": date,
        },
        scorers=eval_scorers,
    )
    
    # Create the sample with the conversation
    # Input should be the initial messages (system + first user message) as a list
    input_messages = []
    for msg in conv:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            input_messages.append(ChatMessageSystem(content=content))
        elif role == "user":
            input_messages.append(ChatMessageUser(content=content))
            break  # Stop after first user message
    
    # Count messages by role
    n_user_msgs = sum(1 for m in conv if m["role"] == "user")
    n_assistant_msgs = sum(1 for m in conv if m["role"] == "assistant")
    n_tool_msgs = sum(1 for m in conv if m["role"] == "tool")
    n_system_msgs = sum(1 for m in conv if m["role"] == "system")
    
    # Create model usage entry (required for Model column to show)
    model_usage = {
        model_id: ModelUsage(
            input_tokens=0,  # We don't track tokens in this script
            output_tokens=0,
            total_tokens=0,
        )
    }
    
    # Create scores for the sample (these become columns in Inspect View)
    sample_scores = {
        "message_count": Score(value=len(conv)),
        "user_messages": Score(value=n_user_msgs),
        "assistant_messages": Score(value=n_assistant_msgs),
        "tool_messages": Score(value=n_tool_msgs),
    }
    
    sample = EvalSample(
        id=1,
        epoch=1,
        input=input_messages if input_messages else messages[0:1],
        target="interactive_chat",
        messages=messages,
        output=ModelOutput(
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=last_assistant_content),
                    stop_reason="stop"
                )
            ] if last_assistant_content else [],
            completion=last_assistant_content,
            usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        ),
        scores=sample_scores,
        metadata={
            "experiment_name": experiment_name,
            "conversation_length": len(conv),
        },
        store={},
        events=[],
        model_usage=model_usage,
    )
    
    # Create the EvalLog
    now = datetime.now().isoformat()
    
    # Define the scores that will appear as columns in Inspect View
    eval_scores = [
        EvalScore(
            name="message_count",
            scorer="message_count",
            params={},
            metrics={"value": EvalMetric(name="value", value=len(conv))},
        ),
        EvalScore(
            name="user_messages",
            scorer="user_messages",
            params={},
            metrics={"value": EvalMetric(name="value", value=n_user_msgs)},
        ),
        EvalScore(
            name="assistant_messages",
            scorer="assistant_messages",
            params={},
            metrics={"value": EvalMetric(name="value", value=n_assistant_msgs)},
        ),
        EvalScore(
            name="tool_messages",
            scorer="tool_messages",
            params={},
            metrics={"value": EvalMetric(name="value", value=n_tool_msgs)},
        ),
    ]
    
    eval_log = EvalLog(
        version=2,
        status="success",
        eval=eval_spec,
        plan=EvalPlan(name="interactive_chat", steps=[], config=GenerateConfig()),
        results=EvalResults(total_samples=1, completed_samples=1, scores=eval_scores),
        stats=EvalStats(started_at=date, completed_at=now, model_usage=model_usage),
        samples=[sample],
    )
    
    return eval_log


def save_inspect_log(conv: list, prefix: str, experiment_name: str, date: str, push_to_s3: bool = True, model_id: str = "", random_seed: int=0, temperature: float = 1, n_tokens: int = 0):
    """Save conversation as Inspect AI log locally and optionally push to S3."""
    
    if len(conv) == 0:
        return None
    
    # Create the Inspect log
    eval_log = create_inspect_log(
        conv=conv,
        model_id=model_id,
        experiment_name=experiment_name,
        date=date,
        random_seed=random_seed,
        temperature=temperature,
        n_tokens=n_tokens,
    )
    
    # Create local logs directory
    local_log_dir = os.path.join("logs", experiment_name)
    os.makedirs(local_log_dir, exist_ok=True)
    
    # Generate filename based on eval_id
    filename = f"{experiment_name}_{date}_{eval_log.eval.eval_id[:8]}.eval"
    local_path = os.path.join(local_log_dir, filename)
    
    # Push to S3 if enabled
    if push_to_s3:
        s3_path = f"s3://{S3_BUCKET}/logs/{prefix}/{date}/{filename}"
        try:
            write_eval_log(eval_log, s3_path)
        except Exception as e:
            pass
    
    return local_path
