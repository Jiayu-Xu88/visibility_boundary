"""Exp1: CONCH zero-shot concept labeling of CRC_MSI_state0 (at scale).

Closes the interpretability gap WITHOUT a pathologist using CONCH's
vision-language alignment. We sample patches assigned to state0 (positives) and
non-state0 patches (background) across CRC slides, extract the actual 512px
images from the WSIs, encode them in CONCH's contrastive image space, and score
each against a curated concept panel (Greenson 2009 classic MSI-H predictors +
neutral controls). We test whether state0 patches are enriched for MSI-associated
concepts vs background.

Refactor: process ONE slide at a time (open -> read -> encode -> close) so we
never hold many SVS handles open; unbuffered progress prints.
"""
import sys, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_utils as A
import openslide

OUT = Path(__file__).resolve().parent
PROJ = A.PROJECT_ROOT
CENTERS = np.load(A.CENTERS)
C2 = 0.5 * (CENTERS * CENTERS).sum(axis=1)
random.seed(0); np.random.seed(0)

TARGET_PER_GROUP = 1000
MAX_SLIDES = 130
POS_PER_SLIDE = 20
BG_PER_SLIDE = 12
PATCH = 512

CONCEPTS = {
    "tumor infiltrating lymphocytes": "MSI",
    "mucinous adenocarcinoma with mucin pools": "MSI",
    "poorly differentiated tumor cells": "MSI",
    "medullary carcinoma with solid sheets": "MSI",
    "Crohn-like peritumoral lymphoid aggregates": "MSI",
    "signet ring cells": "MSI",
    "well differentiated glandular structures": "control",
    "smooth muscle and fibrous stroma": "control",
    "normal colonic mucosa": "control",
    "necrotic debris": "control",
    "adipose tissue": "control",
    "blood and red cells": "control",
}


def tokenize_compat(tokenizer, texts):
    tok = tokenizer(texts, max_length=127, add_special_tokens=True,
                    return_token_type_ids=False, truncation=True,
                    padding="max_length", return_tensors="pt")
    return F.pad(tok["input_ids"], (0, 1), value=tokenizer.pad_token_id)


def plan_patches():
    """Return slide_path -> list[(x,y,group)] with group in {pos,bg}."""
    man = A.slide_manifest_for_cohort()
    crc = man[man.cohort == "CRC_MSI"].to_dict("records")
    random.shuffle(crc)
    plan = {}
    npos = nbg = 0
    for k, r in enumerate(crc):
        if (npos >= TARGET_PER_GROUP and nbg >= TARGET_PER_GROUP) or len(plan) >= MAX_SLIDES:
            break
        try:
            obj = torch.load(PROJ / r["feature_path"], map_location="cpu")
        except Exception:
            continue
        x = obj["features"].float().numpy(); coords = obj["coords"].numpy()
        sp = obj["metadata"]["slide_path"]
        idx = A.stable_indices(r["file_id"] + ":assign", x.shape[0], A.PATCHES_ASSIGN)
        xs = x[idx]; cs = coords[idx]
        a = (xs @ CENTERS.T - C2[None, :]).argmax(axis=1)
        p_idx = np.where(a == 0)[0]; b_idx = np.where(a != 0)[0]
        items = []
        for j in p_idx[:POS_PER_SLIDE]:
            if npos < TARGET_PER_GROUP:
                items.append((int(cs[j][0]), int(cs[j][1]), "pos")); npos += 1
        if len(b_idx):
            for j in np.random.choice(b_idx, size=min(BG_PER_SLIDE, len(b_idx)), replace=False):
                if nbg < TARGET_PER_GROUP:
                    items.append((int(cs[j][0]), int(cs[j][1]), "bg")); nbg += 1
        if items:
            plan[sp] = items
        if k % 20 == 0:
            print(f"  planned slides={len(plan)} pos={npos} bg={nbg}", flush=True)
    print(f"PLAN done: slides={len(plan)} pos={npos} bg={nbg}", flush=True)
    return plan


def main():
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    device = "cuda"
    plan = plan_patches()

    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
    model.eval().to(device)
    tk = get_tokenizer()
    names = list(CONCEPTS.keys())
    with torch.no_grad():
        txt = F.normalize(model.encode_text(tokenize_compat(tk, names).to(device)), dim=-1)

    embs, groups = [], []
    for n, (sp, items) in enumerate(plan.items(), 1):
        full = str(PROJ / sp) if not sp.startswith("/") else sp
        try:
            slide = openslide.OpenSlide(full)
        except Exception as e:
            print(f"  skip slide {n} open-fail {e}", flush=True); continue
        tiles, grp = [], []
        for x, y, g in items:
            try:
                img = slide.read_region((x, y), 0, (PATCH, PATCH)).convert("RGB")
            except Exception:
                continue
            tiles.append(preprocess(img)); grp.append(g)
        slide.close()
        if tiles:
            with torch.no_grad():
                e = model.encode_image(torch.stack(tiles).to(device),
                                       proj_contrast=True, normalize=True)
            embs.append(e.cpu()); groups.extend(grp)
        if n % 10 == 0:
            print(f"  encoded slides={n}/{len(plan)} patches={len(groups)}", flush=True)

    E = torch.cat(embs, 0); groups = np.array(groups)
    sim = (E.to(device) @ txt.T).cpu().numpy()  # [N, C]
    mp = groups == "pos"; mb = groups == "bg"
    print(f"\nfinal: state0={mp.sum()} background={mb.sum()}", flush=True)

    print(f"\n{'concept':45s} {'grp':4s} {'state0':>8s} {'bg':>8s} {'delta':>7s} {'p':>10s}")
    res = []
    for c, name in enumerate(names):
        a = sim[mp, c]; b = sim[mb, c]
        delta = A.cliffs_delta(a, b); p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        print(f"{name:45s} {CONCEPTS[name]:4s} {a.mean():8.4f} {b.mean():8.4f} {delta:+7.3f} {p:10.2e}")
        res.append(dict(concept=name, group=CONCEPTS[name], state0_mean=float(a.mean()),
                        bg_mean=float(b.mean()), cliffs_delta=float(delta), p_value=float(p)))
    top_p = pd.Series([names[i] for i in sim[mp].argmax(1)]).value_counts(normalize=True)
    top_b = pd.Series([names[i] for i in sim[mb].argmax(1)]).value_counts(normalize=True)
    print("\nstate0 dominant concept (top-1 per patch):"); print(top_p.head(6).round(3).to_string())
    print("\nbackground dominant concept:"); print(top_b.head(6).round(3).to_string())

    out = dict(n_state0=int(mp.sum()), n_bg=int(mb.sum()), concept_tests=res,
               state0_top1_distribution=top_p.round(4).to_dict(),
               bg_top1_distribution=top_b.round(4).to_dict())
    (OUT / "exp1_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT/'exp1_results.json'}", flush=True)


if __name__ == "__main__":
    main()
