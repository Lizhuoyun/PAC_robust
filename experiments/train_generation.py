from src.train.trainer_generation import train_generation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--log_backend", choices=["jsonl", "wandb", "both"])
    parser.add_argument("--wandb_mode", choices=["offline", "online", "disabled"])
    parser.add_argument("--wandb_project")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--wandb_group")
    parser.add_argument("--wandb_name")
    parser.add_argument("--log_every_steps", type=int)
    args = parser.parse_args()
    overrides = list(args.override)
    if args.log_backend:
        overrides.append(f"logging.backend={args.log_backend}")
    if args.wandb_mode:
        overrides.append(f"logging.wandb.mode={args.wandb_mode}")
    if args.wandb_project:
        overrides.append(f"logging.wandb.project={args.wandb_project}")
    if args.wandb_entity:
        overrides.append(f"logging.wandb.entity={args.wandb_entity}")
    if args.wandb_group:
        overrides.append(f"logging.wandb.group={args.wandb_group}")
    if args.wandb_name:
        overrides.append(f"logging.wandb.name={args.wandb_name}")
    if args.log_every_steps is not None:
        overrides.append(f"logging.log_every={args.log_every_steps}")
    train_generation(args.config, overrides=overrides)
