# Data Collection

We are relying on `wandb` to collect basic metrics.

After the training, we will have a github action workflow running, to digest the raw output, and save a report in another github repo, and we can read the report from the hosted github page.