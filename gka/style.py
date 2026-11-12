"""Display names and plot colours for each model, shared by the analysis scripts."""

ORDER = ["gqa", "gka", "gka_drope", "gva_drope", "gva", "sharedkv", "gqa_g1", "gka_g4"]
LABEL = {
    "gqa": "GQA (baseline)",
    "gka": "GKA (ours, RoPE)",
    "gka_drope": "GKA + DRoPE16 (ours)",
    "gva_drope": "GVA + DRoPE16",
    "gva": "GVA (RoPE, no absorb)",
    "sharedkv": "Shared KV (K=V)",
    "gqa_g1": "MQA (GQA, G=1)",
    "gka_g4": "GKA, G=4 (ours)",
}
# Reference categorical palette (dataviz skill), one fixed slot per model, light mode.
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]))
