import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
)


import wandb

from blockbuster.data import build_dataset, build_test_dataset
from blockbuster.models.config import TrainConfig
from blockbuster.models.vanillas import BaselineGPT


def build_tokenizer():
    """
    We'll be using the GPT2Tokenizer from huggingface.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def eval_model(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            steps += 1

    # switch back to train mode
    model.train()
    avg_loss = total_loss / steps if steps > 0 else float("nan")
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()


def main():
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer()
    model = BaselineGPT(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.block_size,
        hidden_size=cfg.hidden_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
    ).to(device)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataset = build_dataset(
        tokenizer,
        max_train_rows=cfg.max_train_rows,
        block_size=cfg.block_size,
        test_rows=cfg.test_rows,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
    )

    test_dataset = build_test_dataset(
        tokenizer,
        block_size=cfg.block_size,
        test_rows=cfg.test_rows,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.init(
        project="blockbuster",
        name="vanilla_gpt2",
        config={
            **cfg.model_dump(),
            "dataset": cfg.dataset,
            "num_params": num_params,
            "device": str(device),
        },
    )

    model.train()
    global_step = 0
    for epoch in range(cfg.num_epochs):
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
            if global_step % cfg.log_every == 0:
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

        test_loss, test_ppl = eval_model(model, test_loader, device)
        wandb.log(
            {
                "test/loss": test_loss,
                "test/perplexity": test_ppl,
                "epoch": epoch,
                "step": global_step,
            },
            step=global_step,
        )
        print(f"epoch {epoch} eval | test_loss {test_loss:.4f} | test_ppl {test_ppl:.2f}")

    wandb.finish()


if __name__ == "__main__":
    main()
