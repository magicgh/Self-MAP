import logging
import pickle

import hydra
from dataloader import MultiChoiceDataset, get_data_split
from hydra.core.hydra_config import HydraConfig
from metric import ActionEvaluatorGeneration, ActionEvaluatorMultiChoice
from conf.types import Config
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    logger.info(f"Use model {cfg.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_name
        if cfg.model.tokenizer_name
        else cfg.model.model_name_or_path
    )

    candidate_results = None
    if cfg.data.score_file is not None:
        with open(cfg.data.score_file, "rb") as f:
            candidate_results = pickle.load(f)

    output_dir = HydraConfig.get().runtime.output_dir

    train_data = get_data_split(
        cfg.data.data_path,
        cfg.data.train_split_file,
        is_train=True,
        candidate_results=candidate_results,
    )
    train_dataset = MultiChoiceDataset(
        train_data,
        tokenizer,
        neg_ratio=cfg.train.neg_ratio,
        num_candidates=cfg.train.num_candidates,
        max_seq_len=cfg.model.max_seq_length,
        max_context_len=cfg.train.max_context_len,
        mode="generation" if cfg.self_map.generation else "multichoice",
        self_map=cfg.self_map,
    )

    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model.model_name_or_path,
        device_map=None,
    )
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    step_per_epoch = int(
        len(train_dataset)
        / cfg.train.per_device_train_batch_size
        / cfg.train.num_gpus
        / cfg.train.gradient_accumulation_steps
    )
    logger.info(f"step_per_epoch: {step_per_epoch}")
    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        dataloader_num_workers=2,
        predict_with_generate=True,
        fp16=False,  # Overflows with fp16
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.epoch,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=int(step_per_epoch * 0.2),
        save_strategy="epoch",
        save_total_limit=2,
        # push to hub parameters
        report_to="tensorboard",
        bf16=cfg.train.bf16,
        tf32=cfg.train.tf32,
        auto_find_batch_size=True,
        optim=cfg.train.optim,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        fsdp=cfg.train.fsdp_policy if cfg.train.fsdp else "",
        fsdp_transformer_layer_cls_to_wrap=(
            cfg.model.fsdp_transformer_layer_cls_to_wrap if cfg.train.fsdp else None
        ),
    )
    if not cfg.self_map.generation:
        evaluator = ActionEvaluatorMultiChoice(tokenizer, self_map=cfg.self_map)
    else:
        evaluator = ActionEvaluatorGeneration(
            tokenizer,
            filter_candidate=candidate_results is not None,
            self_map=cfg.self_map,
        )
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        compute_metrics=evaluator,
    )

    trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=output_dir)


if __name__ == "__main__":
    main()
