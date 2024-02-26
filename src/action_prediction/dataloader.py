import copy
import pathlib
import random
import re
import sys
import copy
import lxml
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import List, Union

sys.path.append(pathlib.Path(__file__).parent.parent.absolute().as_posix())
from dom_utils import get_tree_repr, prune_tree
from memory import select_memory, reflection_history


def format_input_generation(
    sample: dict,
    candidate_ids: List[int],
    gt: int = -1,
    previous_k: Union[int, None] = 5,
    keep_html_brackets: bool = False,
    memory_refinement: bool = False,
):
    dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
    dom_tree = prune_tree(dom_tree, candidate_ids)
    tree_repr, id_mapping = get_tree_repr(
        dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
    )
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    choices = []
    for idx, node in enumerate(candidate_nodes):
        choices.append(
            [
                node.attrib["backend_node_id"],
                " ".join(
                    get_tree_repr(
                        node,
                        id_mapping=id_mapping,
                        keep_html_brackets=keep_html_brackets,
                    )[0].split()[:10]
                ),
            ]
        )
    gt_backend_node_id = gt
    if gt_backend_node_id == -1:
        gt_dom_tree = ""
    else:
        gt_dom_tree = prune_tree(
            lxml.etree.fromstring(sample["cleaned_html"]), [gt_backend_node_id]
        )
        gt_dom_tree, _ = get_tree_repr(
            gt_dom_tree,
            id_mapping=id_mapping,
            keep_html_brackets=keep_html_brackets,
        )
    gt = id_mapping.get(gt, -1)
    if previous_k is None:
        seq_input = (
            "Based on the HTML webpage above, try to complete the following task:\n"
            f"Task: {sample['confirmed_task']}\n"
        )
    else:
        seq_input = (
            "Based on the HTML webpage above, try to complete the following task:\n"
            f"Task: {sample['confirmed_task']}\n"
            f"Previous actions:\n"
        )
        if len(sample["previous_actions"]) > 0:
            for action in sample["previous_actions"][-previous_k:]:
                seq_input += f"{action}\n"
        else:
            seq_input += "None\n"
    seq_input += (
        "What should be the next action? "
        "Please select the element to interact with, and the action to perform along with the value to type in or select. "
        "If the task cannot be completed, output None."
    )

    if gt == -1:
        seq_target = "None"
    else:
        current_action_op = sample["operation"]["op"]
        current_action_value = sample["operation"]["value"]
        seq_target = f"Element: {choices[gt][1]}\n"
        seq_target += f"Action: {current_action_op}\n"
        if current_action_op != "CLICK":
            seq_target += f"Value: {current_action_value}"

    def revise_refinement_input(input_seq: str) -> str:
        if input_seq[-1] != "\n":
            input_seq += "\n"
        input_seq = re.sub(r"Element: .*\n", "", input_seq)
        return input_seq.strip()

    if previous_k is None and memory_refinement:
        revised_input = revise_refinement_input(seq_target)
        seq_target += "Rationale: " + reflection_history(
            sample, gt_dom_tree, gt_backend_node_id, revised_input
        )
    return tree_repr, seq_input, seq_target, choices


def format_input_multichoice(
    sample: dict,
    candidate_ids: List[int],
    gt: int = -1,
    previous_k: Union[int, None] = 5,
    keep_html_brackets: bool = False,
    memory_refinement: bool = False,
):
    dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
    dom_tree = prune_tree(dom_tree, candidate_ids)
    tree_repr, id_mapping = get_tree_repr(
        dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
    )
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    choices = []
    for idx, node in enumerate(candidate_nodes):
        choices.append(
            [
                node.attrib["backend_node_id"],
                " ".join(
                    get_tree_repr(
                        node,
                        id_mapping=id_mapping,
                        keep_html_brackets=keep_html_brackets,
                    )[0].split()[:10]
                ),
            ]
        )
    gt_backend_node_id = gt
    if gt_backend_node_id == -1:
        gt_dom_tree = ""
    else:
        gt_dom_tree = prune_tree(
            lxml.etree.fromstring(sample["cleaned_html"]), [gt_backend_node_id]
        )
        gt_dom_tree, _ = get_tree_repr(
            gt_dom_tree,
            id_mapping=id_mapping,
            keep_html_brackets=keep_html_brackets,
        )
    gt = id_mapping.get(gt, -1)
    if previous_k is None:
        seq_input = (
            "Based on the HTML webpage above, try to complete the following task:\n"
            f"Task: {sample['confirmed_task']}\n"
        )
    else:
        seq_input = (
            "Based on the HTML webpage above, try to complete the following task:\n"
            f"Task: {sample['confirmed_task']}\n"
            f"Previous actions:\n"
        )
        if len(sample["previous_actions"]) > 0:
            for action in sample["previous_actions"][-previous_k:]:
                seq_input += f"{action}\n"
        else:
            seq_input += "None\n"
    seq_input += (
        "What should be the next action? Please select from the following choices "
        "(If the correct action is not in the page above, please select A. 'None of the above'):\n\n"
        "A. None of the above\n"
    )
    for idx, choice in enumerate(choices):
        # convert to ascii A, B, C, D, ...
        seq_input += f"{chr(66 + idx)}. {choice[1]}\n"
    if gt == -1:
        seq_target = "A."
    else:
        gt += 1
        current_action_op = sample["operation"]["op"]
        current_action_value = sample["operation"]["value"]
        seq_target = f"{chr(65 + gt)}.\n" f"Action: {current_action_op}\n"
        if current_action_op != "CLICK":
            seq_target += f"Value: {current_action_value}"

    def revise_refinement_input(input_seq: str) -> str:
        if input_seq[-1] != "\n":
            input_seq += "\n"
        return input_seq[2:].strip()

    if previous_k is None and memory_refinement:
        revised_input = revise_refinement_input(seq_target)
        seq_target += "Rationale: " + reflection_history(
            sample, gt_dom_tree, gt_backend_node_id, revised_input
        )
    return tree_repr, seq_input, seq_target, choices


class MultiChoiceDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        self_map,
        neg_ratio=5,
        num_candidates=5,
        max_context_len=512,
        max_seq_len=2048,
        mode="multichoice",
        top_k=-1,
    ):
        self.data = data
        self.neg_ratio = neg_ratio
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.max_seq_len = max_seq_len
        self.max_context_len = max_context_len
        self.mode = mode
        self.top_k = top_k
        self.system_message = "You are a helpful assistant that is great at website design, navigation, and executing tasks for the user.\n\n"
        self.self_map = self_map

    def __len__(self):
        return len(self.data) * 10

    def __getitem__(self, idx):
        sample = self.data[idx // 10]
        history = self.make_history(sample)
        system_seq = encode_system_message(
            self.system_message, self.tokenizer, self.max_context_len
        )
        history_seq = add_speaker_and_signal(
            history, self.tokenizer, self.max_context_len
        )
        seq_context, seq_in, seq_out = self.make_instruction(sample, previous_k=5)
        current_conversation = [
            {"from": "webpage", "value": seq_context},
            {"from": "human", "value": seq_in},
            {"from": "gpt", "value": ""},
        ]
        current_seq = add_speaker_and_signal(
            current_conversation, self.tokenizer, self.max_context_len
        )
        clip_len = (
            self.max_seq_len
            - len(system_seq["input_ids"])
            - len(current_seq["input_ids"])
        )
        if clip_len > 0:
            model_input = {
                "input_ids": system_seq["input_ids"]
                + history_seq["input_ids"][-clip_len:]
                + current_seq["input_ids"],
                "attention_mask": system_seq["attention_mask"]
                + history_seq["attention_mask"][-clip_len:]
                + current_seq["attention_mask"],
            }
        else:
            model_input = {
                "input_ids": system_seq["input_ids"] + current_seq["input_ids"],
                "attention_mask": system_seq["attention_mask"]
                + current_seq["attention_mask"],
            }

        seq_out = self.tokenizer(seq_out)
        model_input["labels"] = seq_out["input_ids"]
        return model_input

    def make_instruction(self, sample: dict, previous_k: Union[int, None] = 5):
        if self.top_k > 0:
            top_negatives = [
                c for c in sample["neg_candidates"] if c["rank"] < self.top_k
            ]
            other_negatives = [
                c for c in sample["neg_candidates"] if c["rank"] >= self.top_k
            ]
        else:
            top_negatives = []
            other_negatives = sample["neg_candidates"]
        if random.random() < 0.8 and len(top_negatives) > 0:
            neg_candidates = top_negatives
        else:
            neg_candidates = other_negatives

        if len(sample["pos_candidates"]) != 0 and (
            random.random() > self.neg_ratio or len(neg_candidates) == 0
        ):
            pos_candidate = random.choice(sample["pos_candidates"])
            if previous_k is not None or self.self_map.memory_simplification:
                neg_candidate = random.sample(
                    neg_candidates,
                    min(len(neg_candidates), self.num_candidates - 1),
                )
            else:
                neg_candidate = neg_candidates
            gt = pos_candidate["backend_node_id"]
            candidate_ids = [gt] + [c["backend_node_id"] for c in neg_candidate]
            if self.mode == "multichoice":
                seq_context, seq_in, seq_out, _ = format_input_multichoice(
                    sample,
                    candidate_ids,
                    gt,
                    previous_k,
                    memory_refinement=self.self_map.memory_refinement,
                )
            else:
                seq_context, seq_in, seq_out, _ = format_input_generation(
                    sample,
                    candidate_ids,
                    gt,
                    previous_k,
                    memory_refinement=self.self_map.memory_refinement,
                )
        else:
            if previous_k is not None or self.self_map.memory_simplification:
                neg_candidate = random.sample(
                    neg_candidates,
                    min(len(neg_candidates), self.num_candidates),
                )
            else:
                neg_candidate = neg_candidates
            gt = -1
            candidate_ids = [c["backend_node_id"] for c in neg_candidate]
            if self.mode == "multichoice":
                seq_context, seq_in, seq_out, _ = format_input_multichoice(
                    sample,
                    candidate_ids,
                    gt,
                    previous_k,
                    memory_refinement=self.self_map.memory_refinement,
                )
            else:
                seq_context, seq_in, seq_out, _ = format_input_generation(
                    sample,
                    candidate_ids,
                    gt,
                    previous_k,
                    memory_refinement=self.self_map.memory_refinement,
                )
        return seq_context, seq_in, seq_out

    def make_history(self, sample: dict) -> List:
        previous = select_memory(sample, self.self_map.multifaceted_matching)
        history = []
        for previous_sample in previous:
            seq_context, seq_in, seq_out = self.make_instruction(
                previous_sample, previous_k=None
            )
            history.extend(
                [
                    {"from": "webpage", "value": seq_context},
                    {"from": "human", "value": seq_in},
                    {"from": "gpt", "value": seq_out},
                ]
            )
        return history


def encode_system_message(system_message, tokenizer, max_context_len):
    tokenized_message = tokenizer(
        system_message,
        add_special_tokens=True,
        truncation=True,
        max_length=max_context_len,
    )
    return tokenized_message


def add_speaker_and_signal(source, tokenizer, max_context_len):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = {
        "input_ids": [],
        "attention_mask": [],
    }

    unknown_role = "unknown"  # use default unknown role
    roles = {
        "webpage": "Human",
        "human": "Human",  # human role
        "gpt": "Assistant",  # gpt role
    }

    def add_to_conversation(message, role):
        if not message.strip():
            return
        add_special_tokens = False if role == "webpage" else True
        tokenized_input = tokenizer(
            message,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_context_len,
        )

        conversation["input_ids"].extend(tokenized_input["input_ids"])
        conversation["attention_mask"].extend(tokenized_input["attention_mask"])

    for i, sentence in enumerate(source):
        sentence_from = sentence["from"].lower()
        content = ""
        if sentence_from == "human":
            next_role = source[i + 1]["from"].lower()
            content = (
                sentence["value"]
                + END_SIGNAL
                + BEGIN_SIGNAL
                + roles.get(next_role, unknown_role)
                + ": "
            )
            add_to_conversation(content, sentence_from)

        elif sentence_from == "gpt":
            add_to_conversation(sentence["value"] + END_SIGNAL, sentence_from)

        elif sentence_from == "webpage":
            if sentence["value"] == "":
                content = BEGIN_SIGNAL + roles.get(sentence_from, unknown_role) + ": "
                add_to_conversation(content, role="human")
            else:
                content = (
                    BEGIN_SIGNAL + roles.get(sentence_from, unknown_role) + ": '''\n"
                )
                add_to_conversation(content, role="human")
                add_to_conversation(sentence["value"], sentence_from)
                add_to_conversation("\n'''\n\n", role="human")

    return conversation


def get_data_split(data_dir, split_file, candidate_results=None, is_train=False):
    def flatten_actions(samples):
        outputs = {
            "task_id": [],
            "website": [],
            "domain": [],
            "subdomain": [],
            "confirmed_task": [],
            "annotation_id": [],
            "previous_actions": [],
            "action_uid": [],
            "operation": [],
            "pos_candidates": [],
            "neg_candidates": [],
            "cleaned_html": [],
            "previous_turns": [],
        }

        task_info = {
            "task_id": samples["task_id"],
            "website": samples["website"],
            "domain": samples["domain"],
            "subdomain": samples["subdomain"],
        }

        for t_idx, turns in enumerate(samples["turns"]):
            history_turns = []
            for turn in turns:
                num_actions = len(turn["actions"])
                outputs["annotation_id"] += [turn["annotation_id"]] * num_actions
                outputs["confirmed_task"] += [turn["confirmed_task"]] * num_actions
                for key in task_info:
                    outputs[key] += [task_info[key][t_idx]] * num_actions

                (
                    all_action_uid,
                    all_action_repr,
                    all_cleaned_html,
                    all_operation,
                    all_pos_candidates,
                    all_neg_candidates,
                ) = ([], [], [], [], [], [])
                for a_idx, action in enumerate(turn["actions"]):
                    outputs["previous_actions"].append(turn["action_reprs"][:a_idx])
                    all_action_uid.append(action["action_uid"])
                    all_action_repr.append(turn["action_reprs"][a_idx])
                    all_cleaned_html.append(action["cleaned_html"])
                    all_operation.append(action["operation"])
                    all_pos_candidates.append(action["pos_candidates"])
                    all_neg_candidates.append(action["neg_candidates"])

                outputs["action_uid"] += all_action_uid
                outputs["cleaned_html"] += all_cleaned_html
                outputs["operation"] += all_operation
                outputs["pos_candidates"] += all_pos_candidates
                outputs["neg_candidates"] += all_neg_candidates

                if len(history_turns) > 0:
                    outputs["previous_turns"] += [
                        copy.deepcopy(history_turns)
                    ] * num_actions
                else:
                    outputs["previous_turns"] += [None] * num_actions

                history_turns.append(
                    {
                        "action_uid": all_action_uid,
                        "action_reprs": all_action_repr,
                        "website": task_info["website"][t_idx],
                        "domain": task_info["domain"][t_idx],
                        "subdomain": task_info["subdomain"][t_idx],
                        "confirmed_task": turn["confirmed_task"],
                        "annotation_id": turn["annotation_id"],
                        "operation": all_operation,
                        "pos_candidates": all_pos_candidates,
                        "neg_candidates": all_neg_candidates,
                        "cleaned_html": all_cleaned_html,
                    }
                )

        return outputs

    dataset = load_dataset(data_dir, data_files=split_file, split="all")

    if candidate_results is not None:
        candidate_scores = candidate_results["scores"]
        candidate_ranks = candidate_results["ranks"]

        def get_score(sample):
            for turn in sample["turns"]:
                assert "annotation_id" in turn
                for action in turn["actions"]:
                    sample_id = f"{turn['annotation_id']}_{action['action_uid']}"
                    for candidates in [
                        action["pos_candidates"],
                        action["neg_candidates"],
                    ]:
                        for candidate in candidates:
                            candidate_id = candidate["backend_node_id"]
                            candidate["score"] = candidate_scores[sample_id][
                                candidate_id
                            ]
                            candidate["rank"] = candidate_ranks[sample_id][candidate_id]
            return sample

        ranked_dataset = dataset.map(get_score, writer_batch_size=128)
    else:
        ranked_dataset = dataset

    flatten_dataset = ranked_dataset.map(
        flatten_actions,
        batched=True,
        remove_columns=dataset.column_names,
        batch_size=10,
        num_proc=4,
    )
    if is_train:
        flatten_dataset = flatten_dataset.filter(lambda x: len(x["pos_candidates"]) > 0)

    return flatten_dataset
