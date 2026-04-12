from datasets import load_dataset


def build_dataset(tokenizer, max_train_rows=100_000, block_size=512):
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    raw = raw.filter(lambda x: len(x["text"].strip()) > 0)
    if len(raw) > max_train_rows:
        raw = raw.select(range(max_train_rows))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=False)

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names)

    def pack_blocks(batch):
        all_ids = []
        for ids in batch["input_ids"]:
            all_ids.extend(ids)
        total = (len(all_ids) // block_size) * block_size
        all_ids = all_ids[:total]
        return {
            "input_ids": [all_ids[i : i + block_size] for i in range(0, total, block_size)],
            "attention_mask": [[1] * block_size for _ in range(total // block_size)],
        }

    packed = tokenized.map(pack_blocks, batched=True)
    packed.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return packed
