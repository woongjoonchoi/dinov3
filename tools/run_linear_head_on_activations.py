"""Run the DINOv3 linear classifier on precomputed backbone activations.

This script mirrors the classifier configuration used in
``tools/eval_imagenet_accuracy_ddp.py`` but assumes the backbone forward
pass has already been executed. Provide a tensor file (or a directory of
``.pt`` files created by ``tools/save_backbone_activations.py``) containing
the concatenated CLS + mean patch embeddings and (optionally) the labels to
obtain logits and accuracy without re-running the backbone.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from dinov3.hub.classifiers import ClassifierWeights


@torch.inference_mode()
def _evaluate_head(
    linear_head: torch.nn.Linear,
    loader: DataLoader,
    device: torch.device,
    has_targets: bool,
    save_logits: Optional[pathlib.Path] = None,
) -> None:
    total_correct = torch.tensor(0, device=device, dtype=torch.long)
    total_samples = torch.tensor(0, device=device, dtype=torch.long)
    logits_buffer = []

    linear_head.eval()

    for batch in loader:
        if has_targets:
            activations, targets = batch
            targets = targets.to(device, non_blocking=True)
        else:
            (activations,) = batch
            targets = None
        activations = activations.to(device, non_blocking=True)
        logits = linear_head(activations)
        if save_logits is not None:
            logits_buffer.append(logits.cpu())
        if targets is not None:
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == targets).sum()
            total_samples += targets.numel()

    if save_logits is not None:
        logits_tensor = torch.cat(logits_buffer, dim=0)
        torch.save({"logits": logits_tensor}, save_logits)

    if total_samples.item() == 0:
        print("No targets provided. Skipped accuracy computation.")
        return
    accuracy = (total_correct.float() / total_samples.float()).item()
    print(f"Accuracy: {accuracy * 100:.2f}% ({total_samples.item()} samples)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activations",
        type=pathlib.Path,
        required=True,
        help=(
            "Path to a torch tensor file containing the concatenated CLS + mean patch "
            "embeddings, or a directory of per-sample .pt files. Accepts either a "
            "tensor payload or a dict with an 'activations'/'activation' key."
        ),
    )
    parser.add_argument(
        "--linear-head-weights",
        type=str,
        default=ClassifierWeights.IMAGENET1K.name,
        help=(
            "Classifier weights enum name, checkpoint path, or URL. The shape must match "
            "the activation dimension (2 * embed_dim, usually 8192)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for running the linear head.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the linear head on.",
    )
    parser.add_argument(
        "--save-logits",
        type=pathlib.Path,
        default=None,
        help="Optional path to save logits tensor.",
    )
    return parser.parse_args()


def _extract_activation_and_target(payload: object) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(payload, torch.Tensor):
        return payload, None
    if not isinstance(payload, dict):
        raise TypeError("Activations file must contain a tensor or a dict with activation keys.")
    activations = payload.get("activations")
    if activations is None:
        activations = payload.get("activation")
    if activations is None:
        raise KeyError("Dictionary payload missing required 'activations' or 'activation' key.")
    targets = payload.get("targets")
    if targets is None:
        targets = payload.get("class_idx")
    if isinstance(targets, int):
        targets = torch.tensor(targets)
    return activations, targets


class _ActivationDirectoryDataset(Dataset[torch.Tensor]):
    """Lazily load per-sample activation files to avoid holding them in RAM."""

    def __init__(self, activation_files: Sequence[pathlib.Path]):
        if not activation_files:
            raise FileNotFoundError("No activation .pt files found in the provided directory.")
        self.activation_files = list(sorted(activation_files))

        # Inspect the first file to determine whether targets are expected for all samples.
        first_payload = torch.load(self.activation_files[0], map_location="cpu")
        _, target = _extract_activation_and_target(first_payload)
        self.has_targets = target is not None

    def __len__(self) -> int:
        return len(self.activation_files)

    def __getitem__(self, idx: int):
        payload = torch.load(self.activation_files[idx], map_location="cpu")
        activation, target = _extract_activation_and_target(payload)
        if activation.ndim == 1:
            activation = activation.unsqueeze(0)
        if self.has_targets:
            if target is None:
                raise ValueError(
                    "Target is missing for a sample even though targets were detected in the directory."
                )
            if isinstance(target, torch.Tensor) and target.ndim == 0:
                target = target.unsqueeze(0)
            return activation.squeeze(0), target.squeeze(0)
        return activation.squeeze(0)


def _load_activation_dataset(path: pathlib.Path) -> tuple[Dataset, bool]:
    if path.is_dir():
        activation_files = list(path.rglob("*.pt"))
        dataset = _ActivationDirectoryDataset(activation_files)
        return dataset, dataset.has_targets

    payload = torch.load(path, map_location="cpu")
    activation, targets = _extract_activation_and_target(payload)
    if activation.ndim == 1:
        activation = activation.unsqueeze(0)

    tensors = [activation]
    has_targets = targets is not None
    if has_targets:
        if isinstance(targets, torch.Tensor) and targets.ndim == 0:
            targets = targets.unsqueeze(0)
        tensors.append(targets)
    dataset = TensorDataset(*tensors)
    return dataset, has_targets


def main() -> None:
    args = _parse_args()
    dataset, has_targets = _load_activation_dataset(args.activations)

    if isinstance(dataset, TensorDataset):
        activation_sample = dataset.tensors[0]
    else:
        activation_sample, *_ = dataset[0]
    if activation_sample.ndim != 1:
        raise ValueError("Activations must be a 1D embedding vector of shape [2 * embed_dim].")

    device = torch.device(args.device)

    # Build the linear head with the correct input dimension inferred from the activations.
    in_features = activation_sample.shape[0]
    linear_head = torch.nn.Linear(in_features, 1000)

    # Load weights; _resolve_weights from eval script not reused to keep this standalone.
    weight_spec = args.linear_head_weights.strip()
    if weight_spec.upper() in ClassifierWeights.__members__:
        weight_enum = ClassifierWeights[weight_spec.upper()]
        state_dict = weight_enum.get_state_dict(progress=True)
    else:
        state_dict = torch.load(weight_spec, map_location="cpu")
    linear_head.load_state_dict(state_dict)
    linear_head.to(device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    _evaluate_head(
        linear_head=linear_head,
        loader=loader,
        device=device,
        has_targets=has_targets,
        save_logits=args.save_logits,
    )


if __name__ == "__main__":
    main()
