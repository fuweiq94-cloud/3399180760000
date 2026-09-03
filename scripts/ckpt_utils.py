"""Checkpoint inspection helpers shared by train.py and studio.py.

Kept free of GUI / training imports so both sides can use it cheaply.
"""
import torch


def infer_model_arch(path):
    """Peek into a checkpoint's state dict and infer the architecture it was
    trained with: ('grid', grid_size) for the CNN, ('vision', 20) legacy.
    NOTE: several grid sizes flatten to the same fc input size (e.g. 29–32
    all → 4096), so the project's actual grids are tried first."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    md = ckpt.get('model_dict', {})
    w = md.get('features.0.weight')
    if w is not None and w.dim() == 4:            # Conv2d → CNN grid model
        fc_in = md.get('fc_shared.0.weight')
        if fc_in is not None:
            def feat_dim(g):
                s1 = (g - 1) // 2 + 1
                s2 = (s1 - 1) // 2 + 1
                return 64 * s2 * s2
            for g in (30, 20, 25, 40, 50):        # known grids first
                if feat_dim(g) == fc_in.shape[1]:
                    return ('grid', g)
            for g in range(8, 65):                # approximation fallback
                if feat_dim(g) == fc_in.shape[1]:
                    return ('grid', g)
        return ('grid', 30)
    return ('vision', 20)                          # Conv1d feature model
