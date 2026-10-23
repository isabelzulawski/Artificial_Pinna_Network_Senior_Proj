"""Figures for the notebook (matplotlib). Each function draws one figure and returns it.

Series colours are fixed per model, never by rank: this run blue, the reference MLP runs aqua (dashed), the
previous team's CNN orange (dotted). Line style and markers repeat the identity for colour-blind readers.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter, NullFormatter

from .evaluate import GROUP_INFO

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {  # key -> (label, colour, line style, marker)
    "run": ("this run", "#2a78d6", "-", "o"),
    "mlp": ("MLP, reference runs", "#1baf7a", "--", "s"),
    "prev_original": ("previous team's CNN", "#eb6834", ":", "^"),
}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 10, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 5})


_PLAIN = FuncFormatter(lambda v, _: f"{v:g}")
_HZ = FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")
_ERR_CMAP = LinearSegmentedColormap.from_list("err", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))


def _plain_log(ax, x_hz=False):
    """Log axes with plain tick labels (1, 10, 100; 2k, 4k) instead of powers of ten."""
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(_PLAIN)
        ax.yaxis.set_minor_formatter(NullFormatter())
    if ax.get_xscale() == "log":
        ax.xaxis.set_major_formatter(_HZ if x_hz else _PLAIN)
        ax.xaxis.set_minor_formatter(NullFormatter())


def grid(az, el, train_idx, test_idx):
    """The measured directions and the split."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    tr, te = set(train_idx.tolist()), set(test_idx.tolist())
    both = np.array([i in tr and i in te for i in range(len(az))])
    only_tr = np.array([i in tr and i not in te for i in range(len(az))])
    only_te = np.array([i in te and i not in tr for i in range(len(az))])
    for m, lab, col, mk in ((both, "training and test (new sounds)", "#2a78d6", "o"),
                            (only_tr, "training only", "#2a78d6", "o"), (only_te, "test only (held out)", "#eb6834", "x")):
        if m.any():
            ax.scatter(az[m], el[m], s=14, c=col, marker=mk, label=f"{lab}: {int(m.sum())}",
                       alpha=1.0 if mk == "x" or m is both else 0.5)
    ax.set(xlabel="azimuth (deg)", ylabel="elevation (deg)", xticks=range(0, 361, 40), yticks=range(-90, 91, 30),
           title=f"{len(az)} measured directions")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    return fig


def responses(ir, az, el, sr, picks):
    """Magnitude responses of a few directions; the band the features use is shaded."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    n = 4096
    f = np.fft.rfftfreq(n, 1 / sr)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for k, i in enumerate(picks):
        H = 20 * np.log10(np.abs(np.fft.rfft(ir[i], n)) + 1e-9)
        ax.semilogx(f[1:], H[1:], color=cols[k % 4], lw=1.5, label=f"az {round(az[i])}, el {round(el[i])}")
    ax.axvspan(2000, 20000, color=GRID, alpha=0.6, lw=0, label="feature band 2-20 kHz")
    ax.set(xlabel="frequency (Hz)", ylabel="magnitude (dB)", xlim=(100, sr / 2), title="Measured pinna responses")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def feature_steps(steps, freqs, labels):
    """Welch spectrum -> level removed -> differenced, for a few example sounds."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for k, lab in enumerate(labels):
        axs[0].plot(freqs / 1000, steps["welch_db"][k].cpu(), color=cols[k], lw=1.2, label=lab)
        axs[1].plot(freqs / 1000, steps["level_removed_db"][k].cpu(), color=cols[k], lw=1.2)
        axs[2].plot((freqs[1:] + freqs[:-1]) / 2000, steps["features"][k].cpu(), color=cols[k], lw=1.0)
    axs[0].set(title="1. Welch spectrum (dB)", xlabel="kHz")
    axs[1].set(title="2. level removed (dB re mean)", xlabel="kHz")
    axs[2].set(title="3. differenced: the 191 input numbers", xlabel="kHz")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def training(history, info):
    """Training vs validation loss, and validation error; the kept (best) step is marked."""
    s = [h["step"] for h in history]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    axs[0].plot(s, [h["train_loss"] for h in history], color="#2a78d6", marker="o", label="training loss")
    axs[0].plot(s, [h["val_loss"] for h in history], color="#eb6834", ls="--", marker="s", label="validation loss")
    axs[0].set(xlabel="step", ylabel="cross-entropy (soft labels)", title="Loss")
    axs[1].plot(s, [h["val_err_mean"] for h in history], color="#eb6834", ls="--", marker="s", label="validation, mean")
    axs[1].plot(s, [h["val_err_median"] for h in history], color="#eb6834", ls=":", marker="^", label="validation, median")
    if "train_err_mean" in history[0]:
        axs[1].plot(s, [h["train_err_mean"] for h in history], color="#2a78d6", marker="o", label="training set, mean")
    axs[1].set(xlabel="step", ylabel="great-circle error (deg)", title="Error", yscale="log")
    _plain_log(axs[1])
    for ax in axs:
        ax.axvline(info["best_step"], color=INK2, lw=1, ls="-")
        ax.legend(fontsize=8)
    axs[0].annotate(f"kept: step {info['best_step']}", (info["best_step"], axs[0].get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK2)
    fig.tight_layout()
    return fig


def _ref_series(reference, protocol):
    fams = reference["protocols"][protocol]["families"] if reference else {}
    return {k: fams[k] for k in ("mlp", "prev_original") if k in fams}


def core_by_source(result, reference, protocol):
    """Mean error per source type in the core test: this run vs the references."""
    by = result["metrics"]["core"]["by_source"]
    srcs = list(by)
    refs = _ref_series(reference, protocol)
    keys = ["run"] + list(refs)
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(srcs))
    for j, k in enumerate(keys):
        lab, col, _, _ = SERIES[k]
        if k == "run":
            vals, err = [by[s]["gc_mean"] for s in srcs], None
        else:
            vals = [refs[k]["core_by_source"][s][0] for s in srcs]
            err = [refs[k]["core_by_source"][s][1] for s in srcs]
        ax.bar(x + (j - (len(keys) - 1) / 2) * w, vals, w * 0.9, yerr=err, color=col, label=lab,
               hatch="//" if k == "prev_original" else None, edgecolor=SURF, ecolor=INK2, capsize=2)
    ax.set_xticks(x, srcs)
    ax.set(ylabel="mean great-circle error (deg)", yscale="log", title=f"Core test by source type ({protocol})")
    _plain_log(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def groups(result, reference, protocol="seen"):
    """One panel per condition group: mean error vs the group's variable."""
    conds = result["conditions"]
    names = [g for g in GROUP_INFO if g not in ("core", "core_train") and any(c["group"] == g for c in conds.values())]
    if not names:
        return None
    refs = _ref_series(reference, protocol)
    ncol = 4
    fig, axs = plt.subplots(int(np.ceil(len(names) / ncol)), ncol, figsize=(14, 3.0 * np.ceil(len(names) / ncol)),
                            squeeze=False)
    for ax, g in zip(axs.flat, names):
        cs = [c for c, v in conds.items() if v["group"] == g]
        xs = [conds[c]["x"] for c in cs]
        o = np.argsort(xs)
        xs = np.asarray(xs)[o]
        cs = [cs[i] for i in o]
        ax.plot(xs, [result["metrics"][c]["gc_mean"] for c in cs], color=SERIES["run"][1], marker="o",
                label=SERIES["run"][0])
        for k, F in refs.items():  # references may predate a group: plot only the conditions they have
            lab, col, ls, mk = SERIES[k]
            have = [i for i, c in enumerate(cs) if c in F["gc_mean"]]
            if have:
                ax.plot(xs[have], [F["gc_mean"][cs[i]][0] for i in have], color=col, ls=ls, marker=mk, label=lab)
        ax.set(title=g, xlabel=GROUP_INFO[g][0], ylabel="mean error (deg)", yscale="log")
        hz = g in ("octave", "lowpass", "highpass", "cliff")
        if hz:
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xlabel(GROUP_INFO[g][0].replace("(Hz)", "(Hz; k = 1,000)"))
        _plain_log(ax, x_hz=hz)
    for ax in axs.flat[len(names):]:
        ax.set_visible(False)
    axs.flat[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def error_boxplots(result, conds=None):
    """Per-example error distributions (box: quartiles, whiskers: 5th-95th percentile)."""
    ex = result["examples"]
    conds = conds or [c for c in ex if result["conditions"][c]["group"] in ("core", "core_train")]
    data, labels = [], []
    for c in conds:
        e, s = ex[c]["err"], ex[c]["source"]
        for src in np.unique(s):
            data.append(np.maximum(e[s == src], 1e-2))
            labels.append(f"{c}\n{src}" if len(conds) > 1 else src)
        if len(np.unique(s)) > 1:
            data.append(np.maximum(e, 1e-2))
            labels.append(f"{c}\nall" if len(conds) > 1 else "all")
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(data)), 3.8))
    ax.boxplot(data, whis=(5, 95), showfliers=False, medianprops={"color": "#eb6834"},
               boxprops={"color": "#2a78d6"}, whiskerprops={"color": "#2a78d6"}, capprops={"color": "#2a78d6"})
    ax.set_xticks(range(1, len(data) + 1), labels, fontsize=8)
    ax.axhline(5, color=INK2, lw=1, ls="--")
    ax.set(yscale="log", ylabel="great-circle error (deg)", title="Per-example error (dashed: 5 deg)")
    _plain_log(ax)
    fig.tight_layout()
    return fig


def skymap(result, az, el, cond="core"):
    """Mean error at every tested direction."""
    e, t = result["examples"][cond]["err"], result["examples"][cond]["true_idx"]
    per = np.bincount(t, e, minlength=len(az)) / np.maximum(np.bincount(t, minlength=len(az)), 1)
    used = np.bincount(t, minlength=len(az)) > 0
    fig, ax = plt.subplots(figsize=(9, 3.8))
    sc = ax.scatter(az[used], el[used], c=np.clip(per[used], 0.3, None), s=28, cmap=_ERR_CMAP,
                    norm=LogNorm(vmin=0.3, vmax=max(30, per[used].max())))
    cb = fig.colorbar(sc, ax=ax, label="mean error (deg)")
    cb.ax.yaxis.set_major_formatter(_PLAIN)
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel="azimuth (deg)", ylabel="elevation (deg)", xticks=range(0, 361, 40), yticks=range(-90, 91, 30),
           title=f"Mean error per direction ({cond}); {int((per[used] > 10).sum())} of {int(used.sum())} above 10 deg")
    ax.grid(False)
    fig.tight_layout()
    return fig
