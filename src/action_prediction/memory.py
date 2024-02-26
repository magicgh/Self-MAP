import os
import yaml
import numpy as np
from omegaconf import DictConfig
from llms import OpenAIEmbeddingEngine, OpenAIChatEngine
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union
from utils import Embeddings, Reflexion
from prompts import rationale as rationale_prompt
from conf.types import MemoryConfig
from hydra.core.hydra_config import HydraConfig
from utils import convert_dict_to_dataclass


def read_config() -> DictConfig:
    if not HydraConfig.initialized():
        raise ValueError("HydraConfig is not initialized")
    config_path = os.path.join(
        HydraConfig.get().runtime.config_sources[1].path,
        HydraConfig.get().job.config_name + ".yaml",
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    keys_to_remove = [key for key in config if key not in ("hydra", "openai", "knn")]
    for key in keys_to_remove:
        config.pop(key, None)
    return convert_dict_to_dataclass(MemoryConfig, config)


def fine_query(task: str, actions: Union[List[str], str]) -> str:
    if type(actions) == str:
        actions = [actions]
    return f"Task: {task}\nActions: {'; '.join(actions)}"


def select_memory(sample: dict, enable: bool = False) -> List[dict]:
    previous = sample["previous_turns"]
    if previous is None or not enable:
        return unsqueeze_turns(previous)
    return find_top_similar(sample)


def unsqueeze_turns(previous_turns: Union[List[dict], None]) -> List[dict]:
    previous_actions = []
    task_info = ["confirmed_task", "website", "domain", "subdomain", "annotation_id"]
    if previous_turns is not None:
        for turn in previous_turns:
            keys = [key for key in turn if key not in task_info]
            past_actions = []
            for values in zip(*(turn[key] for key in keys)):
                action = {key: value for key, value in zip(keys, values)}
                for key in task_info:
                    action[key] = turn[key]
                action.setdefault("past_actions", []).extend(past_actions)
                past_actions.append(action["action_reprs"])
                previous_actions.append(action)
    return previous_actions


def knn(sample: dict, previous_items: List[dict]) -> List[int]:
    config = read_config()
    sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
    embedding_file = Embeddings(config.hydra.run.dir)
    similarities = embedding_file.load()

    if similarities is not None and sample_id in similarities:
        sorted_clusters = similarities[sample_id]

    else:
        engine = OpenAIEmbeddingEngine(display_cost=config.openai.display_cost)
        clusters = {}
        for idx, item in enumerate(previous_items):
            candidate = fine_query(item["confirmed_task"], item["action_reprs"])
            clusters.setdefault(candidate, []).append(idx)

        query = fine_query(sample["confirmed_task"], sample["previous_actions"])

        query_embedding = engine.embeddings(query)
        candidate_embeddings = engine.embeddings(
            list(clusters.keys())
        )  # Array of strings
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_clusters = [list(clusters.values())[i] for i in sorted_indices]
        embedding_file.update(sample_id, sorted_clusters)

    top_n_items = [idx for _ in sorted_clusters[: config.knn.top_k] for idx in _]

    if config.knn.sort_by == "time":
        top_n_items = sorted(top_n_items)
    return top_n_items


def find_top_similar(sample: dict) -> List[dict]:
    config = read_config()
    previous_actions = unsqueeze_turns(sample["previous_turns"])
    if len(previous_actions) <= config.knn.top_k:
        return previous_actions
    top_n_actions = knn(sample, previous_actions)
    return [previous_actions[i] for i in top_n_actions]


def reflection_history(sample: dict, tree_node: str, gt: int, seq_target: str) -> str:
    config = read_config()
    sample_id = f"{sample['annotation_id']}_{sample['action_uid']}_{gt}"
    reflexion_file = Reflexion(config.hydra.run.dir)
    reflexions = reflexion_file.load()
    if reflexions is not None and sample_id in reflexions:
        return reflexions[sample_id]

    message_template = (
        "### Conversation\nWebpage: {}\n\nUser: {}\n\nAssistant: {}\n\n### "
        + "Rationale"
    )

    if gt == -1:
        assert tree_node == ""
        output = "The assistant's answer is derived from the absence of a specific option in the provided HTML content, leading to the conclusion that none of the options provided are suitable for the user's task."
    else:
        engine = OpenAIChatEngine(
            model=config.openai.model,
            rate_limit=config.openai.rate_limit,
            display_cost=config.openai.display_cost,
        )
        seq_input = (
            "Based on the HTML webpage above, try to complete the following task:\n"
            f"Task: {sample['confirmed_task']}\n"
            f"Previous actions:\n"
        )
        if len(sample["past_actions"]) > 0:
            for action in sample["past_actions"]:
                seq_input += f"{action}\n"
        else:
            seq_input += "None\n"

        seq_input += "What should be the next action?"
        message = [
            {
                "role": "user",
                "content": message_template.format(tree_node, seq_input, seq_target),
            }
        ]
        messages = rationale_prompt + message
        output = engine.generate(messages, max_new_tokens=100)[0]

    reflexion_file.update(sample_id, output)
    return output
