from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from absl import app, flags
from src.config import load_config, load_module, create_output_dir

import wandb

from src.utils.seed import seed_all
from src.utils.checkpoint import save_checkpoint

from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment", None, "Path to YAML experiment config (required).")
flags.mark_flag_as_required("experiment")


class RunningLossTracker:
    def __init__(self):
        self.losses = defaultdict(float)
        self.count = 0.0

    def update(self, losses: dict[str, float]):
        for loss, value in losses.items():
            self.losses[loss] += value
        self.count += 1.0

    def get(self):
        return {loss: value / self.count for loss, value in self.losses.items()}

    def reset(self):
        for loss in self.losses:
            self.losses[loss] = 0.0
        self.count = 0

    def __str__(self):
        return " - ".join(
            (
                f"{loss}: {value / self.count:.05f}"
                for loss, value in self.losses.items()
            )
        )


def epoch_to_step(d, train_loader):
    num_batches = len(train_loader)

    for k, v in d.items():  # convert epoch params to step
        if isinstance(v, int) and "epoch" in k.lower():
            d[k] = num_batches * v


def main(argv):
    #
    # Setup
    #
    cfg = load_config(FLAGS.experiment, argv)
    output_dir, experiment_name = create_output_dir(
        FLAGS.experiment, cfg=cfg, tag=cfg.training.tag
    )

    seed_all(cfg.training.seed)

    device = cfg.training.device if torch.cuda.is_available() else "cpu"

    #
    # Load data
    #
    train_dataset = load_module(cfg.dataset)(**cfg.dataset.params, **cfg.dataset.train)
    val_dataset = load_module(cfg.dataset)(**cfg.dataset.params, **cfg.dataset.val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
        num_workers=2,
    )

    # # for any model kwargs that depend on the dataset
    # kwargs_from_dataset = (
    #     train_dataset.get_model_kwargs()
    #     if hasattr(train_dataset, "get_model_kwargs")
    #     else {}
    # )

    # for kwarg, value in kwargs_from_dataset.items():
    #     print(f"{kwarg}: {value}")

    #
    # Model setup
    #
    model = load_module(cfg.model)(**cfg.model.params).to(device)
    # model = torch.compile(model)

    criterion = load_module(cfg.criterion)(
        **(cfg.criterion.params if "params" in cfg.criterion else {})
    )
    optimizer = load_module(cfg.optimizer)(
        params=model.parameters(), **cfg.optimizer.params
    )

    epoch_to_step(cfg.scheduler.params, train_loader)
    scheduler = load_module(cfg.scheduler)(optimizer=optimizer, **cfg.scheduler.params)

    #
    # Logging
    #
    wandb.init(
        project=cfg.training.project,
        config=cfg,  # log full config
        dir=str(output_dir),  # save WandB run files in the experiment output folder
        name=cfg.training.tag if cfg.training.tag else experiment_name,
        reinit=True,
        mode=cfg.training.wandb,
    )

    #
    # Training
    #
    print(cfg)
    print(f"\nWorking directory: {output_dir}")
    print("Starting training")
    log_iter = cfg.training.log_iter
    val_iter = cfg.training.val_iter
    checkpoint_iter = cfg.training.checkpoint_iter

    running_loss_train = RunningLossTracker()
    running_loss_val = RunningLossTracker()

    for epoch in range(cfg.training.epochs):
        print(f"Epoch {epoch + 1}/{cfg.training.epochs}")

        model.train()

        for batch in tqdm(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()

            with torch.autocast(
                device_type=cfg.training.device,
                dtype=torch.bfloat16,
                enabled=cfg.training.amp,
            ):
                outputs = model(**batch)
                loss, loss_dict = criterion(**outputs)

            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            running_loss_train.update(loss_dict)

        print(running_loss_train)

        if epoch % log_iter == 0:
            wandb.log(
                dict(train=running_loss_train.get()),
                step=epoch,
                commit=False,
            )

        running_loss_train.reset()

        if epoch > 1 and epoch % checkpoint_iter == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path=output_dir / f"checkpoint_{epoch:04d}.pt",
            )

        if epoch % val_iter == 0:
            with torch.no_grad():
                model.eval()

                for batch in val_loader:
                    batch = {
                        k: v.to(device, non_blocking=True) for k, v in batch.items()
                    }

                    with torch.autocast(
                        device_type=cfg.training.device,
                        dtype=torch.bfloat16,
                        enabled=cfg.training.amp,
                    ):
                        outputs = model(**batch)
                        loss, loss_dict = criterion(**outputs)

                    running_loss_val.update(loss_dict)

                print(f"VALIDATION loss:", running_loss_val)

                wandb.log(
                    dict(val=running_loss_val.get()),
                    step=epoch,
                    commit=False,
                )

                running_loss_val.reset()

        wandb.log({}, step=epoch)

    print(f"Training finished! Outputs saved to: {output_dir}")

    save_checkpoint(
        model, optimizer, scheduler, epoch, path=output_dir / "checkpoint_final.pt"
    )


if __name__ == "__main__":
    app.run(main)
