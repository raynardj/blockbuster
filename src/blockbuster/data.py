from datasets import load_dataset


def _tokenize_and_pack(raw, tokenizer, block_size):
    raw = raw.filter(lambda x: len(x["text"].strip()) > 0)

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
    packed = packed.with_format("torch")
    return packed


def build_dataset(tokenizer, max_train_rows=100_000, block_size=512, test_rows=500):
    raw = load_dataset("HuggingFaceFW/fineweb", "default", split="train", streaming=True)
    raw = raw.skip(test_rows).take(max_train_rows)
    return _tokenize_and_pack(raw, tokenizer, block_size)


def build_test_dataset(tokenizer, block_size=512, test_rows=500):
    raw = load_dataset("HuggingFaceFW/fineweb", "default", split="train", streaming=True)
    raw = raw.take(test_rows)
    return _tokenize_and_pack(raw, tokenizer, block_size)
