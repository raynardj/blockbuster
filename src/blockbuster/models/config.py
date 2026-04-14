from pydantic_settings import BaseSettings


class TrainConfig(BaseSettings):
    """
    We'll be reading args from cli or environment variables.
    """

    # data
    block_size: int = 512
    max_train_rows: int = 100_000
    test_rows: int = 500
    dataset: str = "HuggingFaceFW/fineweb"

    # training
    batch_size: int = 8
    lr: float = 3e-4
    num_epochs: int = 1
    log_every: int = 50

    # model
    n_layer: int = 12
    n_head: int = 6
    hidden_size: int = 768
    vocab_size: int = 50257

    model_config = {"env_prefix": "BB_"}
