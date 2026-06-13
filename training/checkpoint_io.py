"""Checkpoint save (golias pickle format)."""
import os
import pickle

import torch.nn as nn

GRAD_CLIP = 1.0


def save_model(model, path, metrics_log, label, arch="Golias-v27"):
    state_dict = {}

    def save_weight(module, param_name, np_key):
        param = module.state_dict().get(param_name)
        if param is not None:
            arr = param.cpu().numpy()
            if param_name == "weight" and isinstance(module, nn.Linear):
                arr = arr.T
            state_dict[np_key] = arr

    for enc_name, module in [("m1_enc", model.m1_enc), ("m2_enc", model.m2_enc), ("m3_meta", model.m3_meta)]:
        for layer_name, layer in [("fc1", module.fc1), ("fc2", module.fc2)]:
            save_weight(layer, "weight", f"{enc_name}.{layer_name}.W")
            save_weight(layer, "bias", f"{enc_name}.{layer_name}.b")

    for i in range(4):
        for layer_name, layer in [("fc1", model.fusion.experts[i].fc1), ("fc2", model.fusion.experts[i].fc2)]:
            save_weight(layer, "weight", f"fusion.expert{i}.{layer_name}.W")
            save_weight(layer, "bias", f"fusion.expert{i}.{layer_name}.b")
    save_weight(model.fusion.gate, "weight", "fusion.gate.W")
    save_weight(model.fusion.gate, "bias", "fusion.gate.b")

    save_weight(model.relation[0], "weight", "relation.fc1.W")
    save_weight(model.relation[0], "bias", "relation.fc1.b")
    save_weight(model.relation[2], "weight", "relation.fc2.W")
    save_weight(model.relation[2], "bias", "relation.fc2.b")

    save_weight(model.mlp_pred[0], "weight", "mlp_pred.fc1.W")
    save_weight(model.mlp_pred[0], "bias", "mlp_pred.fc1.b")
    save_weight(model.mlp_pred[2], "weight", "mlp_pred.fc2.W")
    save_weight(model.mlp_pred[2], "bias", "mlp_pred.fc2.b")

    save_weight(model.decode_lm, "weight", "decode_lm.W")
    save_weight(model.decode_lm, "bias", "decode_lm.b")

    for b in range(2):
        for proj in ["WQ", "WK", "WV", "WO"]:
            attn_layer = getattr(model.attn_blocks[b], proj)
            state_dict[f"attn.block{b}.{proj}"] = attn_layer.weight.data.cpu().numpy().T
        state_dict[f"attn.block{b}._t"] = model.attn_blocks[b].temperature.data.cpu().numpy()

    save_weight(model.c_comp, "weight", "c_comp.W")
    save_weight(model.c_comp, "bias", "c_comp.b")

    for buf_name, key in [("H1_prev", "meta.H1_prev"), ("H2_prev", "meta.H2_prev"),
                          ("TAU", "meta.TAU"), ("K_t", "meta.K_t"), ("step", "meta.step")]:
        state_dict[key] = getattr(model, buf_name).data.cpu().numpy()

    with open(path, "wb") as f:
        pickle.dump({
            "_metadata": {"arch": arch, "total_params": model.total_params},
            "state_dict": state_dict,
            "sidecar_log": metrics_log,
            "metrics": metrics_log[-1] if metrics_log else {},
        }, f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [{label}] Saved {path} ({size_mb:.1f} MB)", flush=True)


def save_checkpoint(model, path, metrics_log, epoch):
    save_model(model, path, metrics_log, f"epoch {epoch} checkpoint")
