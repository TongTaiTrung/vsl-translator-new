"""Chẩn đoán độc lập cho ensemble: open-set, nhiễu, hình học embedding.

Chuyển thể từ scratch audit của Agent 2 (/tmp/vsl_audit/agent2/common.py,
stress_extra.py) thành một module dùng lại được + CLI. Chỉ gọi các hàm/lớp
PUBLIC của ``src.model.training`` (train_one, make_folds, Config, get_device,
set_seed, default_groups, fit_temperature, Normalizer) — KHÔNG sửa training.py.

Lấy embedding (vector trước lớp head) bằng forward-pre-hook trên
``model.head`` thay vì sửa VSLNet, như Agent 2 đã làm trong common.py.

CLI:
    python -m src.model.diagnostics --data data/cache/dataset.npz \
        --out runs/experiments/p1/agent3 --seeds 42 7 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.model import training as tr

__all__ = [
    "load_dataset_npz", "run_cv", "diagnose", "open_set_hold_one_class_out",
    "gaussian_noise_false_acceptance", "distractor_flip_rate",
]


# =============================================================================
# DỮ LIỆU
# =============================================================================
def load_dataset_npz(path):
    """Đọc 1 file .npz kiểu data/cache/dataset.npz; các khoá groups/augmented là tuỳ chọn."""
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    classes = [str(c) for c in d["classes"]] if "classes" in d.files \
        else [str(i) for i in range(int(y.max()) + 1)]
    groups = np.asarray(d["groups"]) if "groups" in d.files else np.arange(len(y))
    validation_mask = ~np.asarray(d["augmented"]) if "augmented" in d.files \
        else np.ones(len(y), dtype=bool)
    return X, y, classes, groups, validation_mask


def _fit_fold(X, trn, va, y_used, n_classes, layout, cfg, device):
    """Normalizer + train_one giống run_training (presence/norm_stats theo cfg).

    getattr: cũng chạy được với module training P1 đóng băng (baseline S0, không có các cờ mới).
    """
    presence_norm = getattr(cfg, "presence_norm", False)
    norm = tr.Normalizer(X[trn], presence_norm) if presence_norm else tr.Normalizer(X[trn])
    kwargs = {}
    if getattr(cfg, "distractor_geometry", False):
        kwargs["hand_presence"] = tr.hand_validity(torch.from_numpy(X[trn])).to(device)
        kwargs["norm_stats"] = (torch.from_numpy(norm.mean).to(device), torch.from_numpy(norm.std).to(device),
                                norm.presence_aware)
    else:
        presence = getattr(tr, "PRESENCE_SLICE", slice(129, 131))   # P1: presence là 129:131
        kwargs["hand_presence"] = torch.from_numpy(X[trn][..., presence].copy()).to(device)
    yt = torch.from_numpy(y_used).to(device)
    m, info = tr.train_one(
        torch.from_numpy(norm(X[trn])).to(device), yt[trn],
        torch.from_numpy(norm(X[va])).to(device), yt[va],
        n_classes, layout, cfg, device, log_every=0, **kwargs,
    )
    return m.eval(), norm, info


def _fold_assignment(y, groups, vmask, k, seed):
    """Gán fold cho MỌI mẫu dựa trên fold của nhóm gốc (augmentation theo origin)."""
    fold_of_originals = tr.make_folds(y[vmask], k, groups[vmask], seed)
    fold_by_group = dict(zip(groups[vmask], fold_of_originals))
    return np.asarray([fold_by_group.get(g, -1) for g in groups], dtype=np.int64)


def _notebook_like_config(seed: int, **overrides) -> tr.Config:
    base = dict(max_epochs=200, head="linear", seed=seed, hard_negative_margin=0.8,
                hard_negative_weight=0.25, mixup_alpha=0.0)
    base.update(overrides)
    return tr.Config(**base)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def _logits_and_embedding(model, Xn: np.ndarray, device):
    """logits + vector trước head, qua forward-pre-hook (không cần sửa VSLNet)."""
    captured = []
    hook = model.head.register_forward_pre_hook(lambda _mod, inp: captured.append(inp[0].detach()))
    model.eval()
    logits = model(torch.from_numpy(Xn).to(device))
    hook.remove()
    return logits.float().cpu().numpy(), captured[0].float().cpu().numpy()


# =============================================================================
# CV CHUẨN (để lấy OOF logits + embedding cho diagnose())
# =============================================================================
def run_cv(X, y, classes, groups, vmask, cfg: tr.Config, device=None, k: int = 5,
           folds_to_run: int | None = None):
    """Lặp lại đúng vòng lặp fold của run_training (không lưu), trả về model/norm/va mỗi fold."""
    device = device or tr.get_device()
    tr.set_seed(cfg.seed)
    layout = tr.default_groups(X.shape[-1])
    fold_of = _fold_assignment(y, groups, vmask, k, cfg.seed)
    out = {"models": [], "norms": [], "va": [], "infos": []}
    for f in range(k)[: folds_to_run or k]:
        va = np.where((fold_of == f) & vmask)[0]
        trn = np.where(fold_of != f)[0]
        m, norm, info = _fit_fold(X, trn, va, y, len(classes), layout, cfg, device)
        out["models"].append(m)
        out["norms"].append(norm)
        out["va"].append(va)
        out["infos"].append({k2: v for k2, v in info.items() if k2 != "val_logits"})
    return out


def diagnose(X, y, classes, vmask, res, device, noise_sigma: float = 1.0, seed: int = 0) -> dict:
    """Chỉ số OOF-trên-mẫu-gốc: logit gap, hình học embedding (cặp lớp gần nhau),
    và tỉ lệ logit-gap trung vị giữ được khi thêm nhiễu Gauss sigma=1 (đơn vị: std toàn cục/cột)."""
    C = len(classes)
    N = len(y)
    global_std = X.reshape(-1, X.shape[-1]).std(0).astype(np.float32)
    rng = np.random.default_rng(seed)

    oof = np.zeros((N, C), np.float32)
    mask = np.zeros(N, bool)
    pair_scores: dict[tuple[int, int], list[float]] = {}
    clean_gap, noisy_gap = [], []

    for m, norm, va in zip(res["models"], res["norms"], res["va"]):
        trn_orig = np.setdiff1d(np.where(vmask)[0], va)
        lo, Ev = _logits_and_embedding(m, norm(X[va]), device)
        oof[va] = lo
        mask[va] = True
        _, Et = _logits_and_embedding(m, norm(X[trn_orig]), device)

        # hình học: khoảng cách cosine đến centroid (chuẩn hoá từ embedding TRAIN)
        nv = Ev / np.linalg.norm(Ev, axis=1, keepdims=True)
        nt = Et / np.linalg.norm(Et, axis=1, keepdims=True)
        centroids = np.stack([nt[y[trn_orig] == c].mean(0) for c in range(C)])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        spread = np.array([np.mean(1 - nt[y[trn_orig] == c] @ centroids[c]) for c in range(C)])
        for a in range(C):
            for b in range(a + 1, C):
                sep = float((1 - centroids[a] @ centroids[b]) / max((spread[a] + spread[b]) / 2, 1e-9))
                pair_scores.setdefault((a, b), []).append(sep)

        # logit gap sạch vs. nhiễu Gauss (trong không gian đặc trưng thô, trước chuẩn hoá)
        true_logit = lo[np.arange(len(va)), y[va]]
        rival = lo.copy()
        rival[np.arange(len(va)), y[va]] = -np.inf
        clean_gap.extend((true_logit - rival.max(1)).tolist())

        # chỉ thêm nhiễu vào cột có tín hiệu trong clip: nhiễu lên cột toàn 0 của tay vắng sẽ "tạo ra"
        # một tay giả, khiến chỉ số không so sánh được giữa bản chuẩn hoá thường và theo presence.
        live = (X[va] != 0).any(axis=1, keepdims=True)
        noisy_X = X[va] + rng.normal(0, 1, X[va].shape).astype(np.float32) * noise_sigma * global_std * live
        lp, _ = _logits_and_embedding(m, norm(noisy_X), device)
        true_logit_n = lp[np.arange(len(va)), y[va]]
        rival_n = lp.copy()
        rival_n[np.arange(len(va)), y[va]] = -np.inf
        noisy_gap.extend((true_logit_n - rival_n.max(1)).tolist())

    yo, lo_oof = y[mask], oof[mask]
    hard_pair_sep = {
        f"{classes[a]}/{classes[b]}": float(np.mean(v))
        for (a, b), v in sorted(pair_scores.items(), key=lambda kv: np.mean(kv[1]))
    }
    clean_gap, noisy_gap = np.array(clean_gap), np.array(noisy_gap)
    return {
        "oof_acc": float((lo_oof.argmax(1) == yo).mean()),
        "oof_macro_f1": float(f1_score(yo, lo_oof.argmax(1), average="macro")),
        "logit_gap_median_clean": float(np.median(clean_gap)),
        "logit_gap_median_noise_sigma1": float(np.median(noisy_gap)),
        "noise_sigma1_gap_retention": float(np.median(noisy_gap) / np.median(clean_gap))
        if np.median(clean_gap) else float("nan"),
        "hard_pair_separation_ratio": hard_pair_sep,
    }


# =============================================================================
# OPEN-SET: HOLD-ONE-CLASS-OUT
# =============================================================================
def open_set_hold_one_class_out(X, y, classes, groups, vmask, device, seed, held_classes=None,
                                 fold: int = 0, k: int = 5, min_confidence: float = 0.65,
                                 min_gap: float = 0.08, cfg: tr.Config | None = None) -> list[dict]:
    """Huấn luyện bỏ hẳn 1 lớp khỏi vocab, đo tỉ lệ bị NHẬN NHẦM khi lớp đó xuất
    hiện lúc suy luận (proxy cho 1 dấu hiệu KHÁC chưa học, không phải cho input
    không phải dấu hiệu nào cả — xem gaussian_noise_false_acceptance)."""
    tr.set_seed(seed)
    cfg = cfg or _notebook_like_config(seed)
    fold_of = _fold_assignment(y, groups, vmask, k, seed)
    layout = tr.default_groups(X.shape[-1])
    held_classes = list(held_classes) if held_classes else list(classes)
    results = []
    for held in held_classes:
        h = classes.index(held)
        keep = y != h
        remaining = sorted({int(v) for v in y[keep].tolist()})
        remap = {old: new for new, old in enumerate(remaining)}
        y_remap = np.array([remap.get(int(v), 0) for v in y], dtype=np.int64)

        va = np.where((fold_of == fold) & vmask & keep)[0]
        trn = np.where((fold_of != fold) & keep)[0]
        m, norm, info = _fit_fold(X, trn, va, y_remap, len(remaining), layout, cfg, device)
        lv, _ = _logits_and_embedding(m, norm(X[va]), device)
        T_fit = tr.fit_temperature(torch.from_numpy(lv), torch.from_numpy(y_remap[va]))

        test_idx = np.where((~keep) & vmask)[0]
        lo, _ = _logits_and_embedding(m, norm(X[test_idx]), device)
        row = {"held_class": held, "n_test": int(len(test_idx)), "fitted_T": T_fit,
               "val_acc": info["val_acc"]}
        for T, label in ((1.0, "T1"), (T_fit, "T_fit")):
            p = _softmax(lo / T)
            srt = np.sort(p, axis=1)
            gap = srt[:, -1] - srt[:, -2]
            accepted = (p.max(1) >= min_confidence) & (gap >= min_gap)
            row[f"false_accept_rate_{label}"] = float(accepted.mean())
            row[f"median_maxprob_{label}"] = float(np.median(p.max(1)))
        results.append(row)
    return results


# =============================================================================
# NHIỄU GAUSS / INPUT KHÔNG PHẢI DẤU HIỆU
# =============================================================================
def gaussian_noise_false_acceptance(res, classes, device, n_samples: int = 50, seed: int = 0,
                                     min_confidence: float = 0.65, min_gap: float = 0.08,
                                     temperatures=(1.0, 0.2)) -> dict:
    """Đưa nhiễu Gauss chuẩn (đã ở thang 'đã chuẩn hoá', không qua Normalizer vì
    không đại diện cho dấu hiệu nào) qua các model fold đã train sẵn trong `res`
    (từ run_cv) -> tỉ lệ bị chấp nhận nhầm theo luật của predictor thật."""
    rng = np.random.default_rng(seed)
    in_dim = res["norms"][0].mean.shape[-1]
    T = res["va"][0]  # chỉ để lấy độ dài chuỗi thực tế từ 1 batch validation
    seq_len = None
    for m in res["models"]:
        seq_len = None
        break
    Z = rng.normal(size=(n_samples, 30, in_dim)).astype(np.float32)
    per_model_logits = []
    for m in res["models"]:
        with torch.no_grad():
            logits = m(torch.from_numpy(Z).to(device)).float().cpu().numpy()
        per_model_logits.append(logits)
    out = {}
    for Temp in temperatures:
        probs = np.mean([_softmax(logits / Temp) for logits in per_model_logits], axis=0)
        srt = np.sort(probs, axis=1)
        gap = srt[:, -1] - srt[:, -2]
        accepted = (probs.max(1) >= min_confidence) & (gap >= min_gap)
        pred_counts = {classes[i]: int(c) for i, c in
                       zip(*np.unique(probs.argmax(1), return_counts=True))}
        out[f"T={Temp}"] = {
            "median_maxprob": float(np.median(probs.max(1))),
            "false_accept_rate": float(accepted.mean()),
            "predicted_as": pred_counts,
        }
    return out


# =============================================================================
# TAY THỨ 2 LỌT KHUNG: TỈ LỆ ĐỔI NHÃN
# =============================================================================
def _composite_left_hand(target: np.ndarray, donor: np.ndarray, sl: dict, lowered_y: float | None):
    """Clip thô một tay + tay trái của donor (hai tay); tính lại inter_hand_vec/hand_contact.

    lowered_y: tay phụ đứng yên (frame giữa của donor) hạ tâm lòng bàn tay xuống y này.
    shortcut: body_contact của tay hạ giữ nguyên của donor, và cặp gần nhất chỉ xét 3x3 điểm
    (palm/cái/trỏ) vì không lưu landmark; cần landmark gốc nếu muốn mô phỏng tuyệt đối.
    """
    T = len(target)
    left = np.concatenate([np.arange(sl[n].start, sl[n].start + (sl[n].stop - sl[n].start) // 2)
                           for n in ("hand_pos", "body_contact", "handshape", "orientation", "presence")])
    src = donor[np.round(np.linspace(0, len(donor) - 1, T)).astype(int)]
    if lowered_y is not None:
        valid = np.flatnonzero(np.isfinite(donor[:, sl["hand_pos"].start]))
        src = np.repeat(donor[valid[len(valid) // 2]][None], T, axis=0)
        hp = sl["hand_pos"].start
        src[:, [hp + 1, hp + 4, hp + 7]] += lowered_y - src[:, [hp + 1]]
    out = target.copy()
    out[:, left] = src[:, left]
    hp = out[:, sl["hand_pos"]]
    lp, rp = hp[:, :9].reshape(T, 3, 3), hp[:, 9:].reshape(T, 3, 3)
    d = np.concatenate([np.linalg.norm(lp - rp, axis=-1),
                        np.linalg.norm(lp[:, :, None] - rp[:, None], axis=-1).reshape(T, -1).min(1, keepdims=True)], 1)
    out[:, sl["inter_hand_vec"]] = lp[:, 0] - rp[:, 0]
    out[:, sl["hand_contact"]] = np.exp(-8.0 * d ** 2)
    return out


def distractor_flip_rate(raw, y, classes, vmask, res, device, fx, lowered_y: float = 1.1,
                         min_confidence: float = 0.65, min_gap: float = 0.08) -> dict:
    """Chèn tay trái của clip hai tay (val cùng fold) vào clip một tay (val), xử lý qua đúng
    fx.process_sequence, đo % đổi nhãn và % chấp nhận một nhãn sai. Chỉ dùng mẫu gốc.

    raw: mảng object các clip thô (T, F) khớp chỉ số với X; fx: module extract (v2 hoặc v3).
    """
    sl, hp = fx._SL, fx._SL["hand_pos"].start
    left_frac = np.array([np.isfinite(r[:, hp]).mean() if v else np.nan for r, v in zip(raw, vmask)])
    out = {}
    for name, low in (("same_height", None), ("lowered_static", lowered_y)):
        flips, wrong_acc, to_twohand, n = 0, 0, 0, 0
        for m, norm, va in zip(res["models"], res["norms"], res["va"]):
            targets = va[left_frac[va] < 0.1]
            donors = va[left_frac[va] >= 0.5]
            if not len(donors) or not len(targets):
                continue
            twohand = set(y[donors].tolist())
            clean = np.stack([fx.process_sequence(raw[i], target_len=X_LEN) for i in targets])
            comp = np.stack([fx.process_sequence(_composite_left_hand(raw[i], raw[donors[j % len(donors)]], sl, low),
                                                 target_len=X_LEN) for j, i in enumerate(targets)])
            pc = _logits_and_embedding(m, norm(clean.astype(np.float32)), device)[0].argmax(1)
            p = _softmax(_logits_and_embedding(m, norm(comp.astype(np.float32)), device)[0])
            pred, srt = p.argmax(1), np.sort(p, axis=1)
            accepted = (srt[:, -1] >= min_confidence) & (srt[:, -1] - srt[:, -2] >= min_gap)
            flips += int((pred != pc).sum())
            wrong_acc += int((accepted & (pred != y[targets])).sum())
            to_twohand += int(np.isin(pred, list(twohand)).sum())
            n += len(targets)
        out[name] = {"n": n, "flip_rate": flips / max(n, 1), "wrong_accept_rate": wrong_acc / max(n, 1),
                     "to_two_hand_class_rate": to_twohand / max(n, 1)}
    return out


X_LEN = 30   # độ dài chuỗi sau process_sequence (EXPECTED_FRAMES)


# =============================================================================
# CLI
# =============================================================================
def _run_one_seed(X, y, classes, groups, vmask, seed, k, held_classes, out_dir, device):
    cfg = _notebook_like_config(seed)
    res = run_cv(X, y, classes, groups, vmask, cfg, device=device, k=k)
    report = diagnose(X, y, classes, vmask, res, device, seed=seed)
    report["open_set_hold_one_class_out"] = open_set_hold_one_class_out(
        X, y, classes, groups, vmask, device, seed, held_classes=held_classes, k=k,
    )
    report["gaussian_noise"] = gaussian_noise_false_acceptance(res, classes, device, seed=seed)
    report["seed"] = seed
    path = Path(out_dir) / f"diagnostics_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Chẩn đoán open-set / nhiễu / hình học embedding")
    parser.add_argument("--data", required=True, help="File .npz (X, y, classes[, groups, augmented])")
    parser.add_argument("--out", required=True, help="Thư mục ghi kết quả JSON")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--k", type=int, default=5, help="Số fold")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Chỉ chạy hold-one-class-out trên các lớp này (mặc định: tất cả)")
    args = parser.parse_args()

    X, y, classes, groups, vmask = load_dataset_npz(args.data)
    device = tr.get_device()
    held_classes = args.classes if args.classes else classes
    for seed in args.seeds:
        report = _run_one_seed(X, y, classes, groups, vmask, seed, args.k, held_classes, args.out, device)
        print(f"seed={seed} oof_acc={report['oof_acc']:.4f} "
              f"gap_retention_sigma1={report['noise_sigma1_gap_retention']:.3f}")


if __name__ == "__main__":
    main()
