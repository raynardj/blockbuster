import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
)

import wandb

from blockbuster.data import build_dataset


BLOCK_SIZE = 512
MAX_TRAIN_ROWS = 100_000
BATCH_SIZE = 8
LR = 3e-4
NUM_EPOCHS = 1
LOG_EVERY = 50


def build_model():
    config = GPT2Config(
        n_layer=6,
        n_head=6,
        n_embd=768,
        vocab_size=50257,
        n_positions=BLOCK_SIZE,
        n_ctx=BLOCK_SIZE,
    )
    return GPT2LMHeadModel(config)


def build_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer()
    model = build_model().to(device)
    dataset = build_dataset(tokenizer, max_train_rows=MAX_TRAIN_ROWS, block_size=BLOCK_SIZE)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.init(
        project="blockbuster",
        name="vanilla_gpt2",
        config={
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 768,
            "block_size": BLOCK_SIZE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "num_epochs": NUM_EPOCHS,
            "dataset": "wikitext-2-raw-v1",
            "max_train_rows": MAX_TRAIN_ROWS,
            "num_params": num_params,
            "device": str(device),
        },
    )

    model.train()
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % LOG_EVERY == 0:
                perplexity = torch.exp(loss.detach())
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/perplexity": perplexity.item(),
                        "epoch": epoch,
                        "step": global_step,
                    },
                    step=global_step,
                )
                print(f"step {global_step} | loss {loss.item():.4f} | ppl {perplexity.item():.2f}")

    wandb.finish()


if __name__ == "__main__":
    main()
