"""Display names and plot colours for each model, shared by the analysis scripts.

Model keys are run names at the default seed (gka.runs): seeds of the same model are grouped under it.
"""

ORDER = ["gqa", "gqa_g1", "mla", "gva_drope", "gva", "gka", "gka_xg", "gka_vrope", "gka_xg_vrope", "sharedkv",
         "gka_drope", "gka_g4"]
LABEL = {
    "gqa": "GQA (baseline)",
    "gqa_g1": "MQA (GQA, G=1)",
    "mla": "MLA (dc=128, DRoPE16)",
    "gva_drope": "GVA + DRoPE16",
    "gva": "GVA (RoPE, no absorb)",
    "gka": "GKA (ours, RoPE)",
    "gka_xg": "GKA + cross-group V",
    "gka_vrope": "GKA + rotated V",
    "gka_xg_vrope": "GKA + cross-group + rotated V",
    "sharedkv": "Shared KV (K=V)",
    "gka_drope": "GKA + DRoPE16 (ours)",
    "gka_g4": "GKA, G=4 (ours)",
}
# Reference categorical palette (dataviz skill), one fixed slot per model, light mode.
# Shared KV, the reference floor, is neutral grey.
COLOR = {
    "gqa": "#2a78d6",
    "gqa_g1": "#4a3aa7",
    "mla": "#1baf7a",
    "gva_drope": "#eda100",
    "gva": "#e87ba4",
    "gka": "#eb6834",
    # the GKA follow-up variants: shades of GKA's orange, so the family reads as one in the plots
    "gka_xg": "#a8431b",
    "gka_vrope": "#f4a27f",
    "gka_xg_vrope": "#6b2a10",
    "sharedkv": "#8a8984",
    "gka_drope": "#008300",
    "gka_g4": "#e34948",
}
