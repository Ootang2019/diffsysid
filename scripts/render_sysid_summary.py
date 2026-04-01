#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from diffsysid.render import draw_panel, load_font

W = 1480
H = 920
BG = (246, 247, 250)
PANEL = (255, 255, 255)
BORDER = (212, 216, 224)
TEXT = (23, 27, 33)
MUTED = (98, 105, 117)
COLORS = {
    "GT": (46, 204, 113),
    "Init": (230, 126, 34),
    "Fit": (52, 152, 219),
    "Loss": (231, 76, 60),
    "RMSE": (142, 68, 173),
}

FONT_BIG = load_font(30)
FONT = load_font(20)
FONT_SMALL = load_font(16)


def map_point(x: float, y: float, bounds: tuple[float, float, float, float], rect: tuple[float, float, float, float]):
    xmin, xmax, ymin, ymax = bounds
    left, top, right, bottom = rect
    sx = (x - xmin) / max(1e-9, xmax - xmin)
    sy = (y - ymin) / max(1e-9, ymax - ymin)
    return left + sx * (right - left), bottom - sy * (bottom - top)


def draw_axes(draw: ImageDraw.ImageDraw, rect, xlabel: str, ylabel: str):
    l, t, r, b = rect
    plot = (l + 18, t + 68, r - 18, b - 24)
    draw.rectangle(plot, outline=(225, 228, 234), width=1)
    draw.text((plot[0], b - 22), xlabel, fill=MUTED, font=FONT_SMALL)
    draw.text((plot[0], t + 48), ylabel, fill=MUTED, font=FONT_SMALL)
    return plot


def draw_polyline(draw: ImageDraw.ImageDraw, xs, ys, bounds, rect, color, width: int = 4):
    pts = [map_point(float(x), float(y), bounds, rect) for x, y in zip(xs, ys)]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width)


def draw_horizontal(draw: ImageDraw.ImageDraw, value: float, bounds, rect, color, label: str):
    y = map_point(0.0, value, bounds, rect)[1]
    draw.line([(rect[0], y), (rect[2], y)], fill=color, width=3)
    draw.text((rect[0] + 10, y - 22), label, fill=color, font=FONT_SMALL)


def load_result(path: Path) -> dict:
    return json.loads(path.read_text())


def build_summary(data: dict, result_path: Path) -> dict:
    fit_param = data["fit_param"]
    initial = data["initial_guess"]
    final = data["final_fit"]
    best = data.get("best_seen", final)
    gt = data["ground_truth"]
    history = data["history"]
    target = np.asarray(data["target_trajectory"], dtype=float)
    init_pred = np.asarray(data["initial_prediction"], dtype=float)
    final_pred = np.asarray(data["final_prediction"], dtype=float)

    initial_loss = float(initial["loss"])
    final_loss = float(final["loss"])
    initial_rmse = float(initial["rmse"])
    final_rmse = float(final["rmse"])

    summary = {
        "result_json": str(result_path),
        "system": result_path.parent.name,
        "fit_param": fit_param,
        "asset": data["asset"],
        "angle_convention": data.get("angle_convention"),
        "config": data["config"],
        "parameter_values": {
            "ground_truth": gt[fit_param],
            "initial": initial[fit_param],
            "final": final[fit_param],
            "best_seen": best[fit_param],
        },
        "raw_parameter_values": {
            "ground_truth": gt.get(f"{fit_param}_raw"),
            "initial": initial.get(f"{fit_param}_raw"),
            "final": final.get(f"{fit_param}_raw"),
            "best_seen": best.get(f"{fit_param}_raw"),
        },
        "metrics": {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "best_loss": float(best["loss"]),
            "initial_rmse": initial_rmse,
            "final_rmse": final_rmse,
            "best_rmse": float(best["rmse"]),
            "loss_improvement": initial_loss - final_loss,
            "rmse_improvement": initial_rmse - final_rmse,
            "loss_improvement_ratio": None if initial_loss == 0.0 else (initial_loss - final_loss) / initial_loss,
            "rmse_improvement_ratio": None if initial_rmse == 0.0 else (initial_rmse - final_rmse) / initial_rmse,
            "final_grad": float(final["grad"]),
            "selection": final.get("selection", "best_env"),
        },
        "tip_trajectory": {
            "target_final": target[-1].tolist(),
            "initial_final": init_pred[-1].tolist(),
            "fitted_final": final_pred[-1].tolist(),
            "initial_endpoint_error": float(np.linalg.norm(init_pred[-1] - target[-1])),
            "fitted_endpoint_error": float(np.linalg.norm(final_pred[-1] - target[-1])),
        },
        "history": {
            "iterations": len(history),
            "first": None if not history else history[0],
            "last": None if not history else history[-1],
            "min_loss_iteration": None if not history else int(np.argmin([x.get("loss", x.get("best_loss")) for x in history])),
            "min_rmse_iteration": None if not history else int(np.argmin([x.get("rmse", x.get("best_rmse")) for x in history])),
        },
    }
    return summary


def make_summary_png(data: dict, summary: dict, out_path: Path):
    target = np.asarray(data["target_trajectory"], dtype=float)
    init_pred = np.asarray(data["initial_prediction"], dtype=float)
    final_pred = np.asarray(data["final_prediction"], dtype=float)
    hist = data["history"]
    fit_param = data["fit_param"]

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((28, 18), f"{summary['system']} sysID summary", fill=TEXT, font=FONT_BIG)
    subtitle = (
        f"fit param={fit_param} | GT={summary['parameter_values']['ground_truth']:.6f} | "
        f"Init={summary['parameter_values']['initial']:.6f} | "
        f"Fit={summary['parameter_values']['final']:.6f} | "
        f"Final RMSE={summary['metrics']['final_rmse']:.3e}"
    )
    draw.text((28, 58), subtitle, fill=MUTED, font=FONT)

    rect_loss = (28, 110, 720, 470)
    rect_param = (748, 110, 1452, 470)
    rect_tip = (28, 500, 980, 892)
    rect_text = (1008, 500, 1452, 892)

    draw_panel(draw, rect_loss, "Optimization curves", FONT, FONT_SMALL, fill=PANEL, text=TEXT, muted=MUTED, subtitle="Loss and RMSE across training iterations.", outline=BORDER)
    draw_panel(draw, rect_param, "Parameter convergence", FONT, FONT_SMALL, fill=PANEL, text=TEXT, muted=MUTED, subtitle="Ground truth, initial, and fitted parameter values.", outline=BORDER)
    draw_panel(draw, rect_tip, "Tip trajectory overlay", FONT, FONT_SMALL, fill=PANEL, text=TEXT, muted=MUTED, subtitle="World-space x-z tip trajectories for GT, init, and fit.", outline=BORDER)
    draw_panel(draw, rect_text, "Key metrics", FONT, FONT_SMALL, fill=PANEL, text=TEXT, muted=MUTED, subtitle="Compact summary from the result JSON.", outline=BORDER)

    if hist:
        iters = np.arange(len(hist), dtype=float)
        losses = np.array([h.get("loss", h.get("best_loss")) for h in hist], dtype=float)
        rmses = np.array([h.get("rmse", h.get("best_rmse")) for h in hist], dtype=float)
        loss_bounds = (0.0, max(1.0, float(iters[-1])), min(float(losses.min()), float(rmses.min())), max(float(losses.max()), float(rmses.max())))
        plot_loss = draw_axes(draw, rect_loss, "iteration", "metric value")
        draw_polyline(draw, iters, losses, loss_bounds, plot_loss, COLORS["Loss"])
        draw_polyline(draw, iters, rmses, loss_bounds, plot_loss, COLORS["RMSE"])
        draw.text((plot_loss[0] + 10, plot_loss[1] + 10), "Loss", fill=COLORS["Loss"], font=FONT_SMALL)
        draw.text((plot_loss[0] + 70, plot_loss[1] + 10), "RMSE", fill=COLORS["RMSE"], font=FONT_SMALL)

        param_hist = np.array([h.get("param_value", h.get("best_param_value")) for h in hist], dtype=float)
        param_vals = [
            float(summary["parameter_values"]["ground_truth"]),
            float(summary["parameter_values"]["initial"]),
            float(summary["parameter_values"]["final"]),
            float(param_hist.min()),
            float(param_hist.max()),
        ]
        param_bounds = (0.0, max(1.0, float(iters[-1])), min(param_vals) - 0.05, max(param_vals) + 0.05)
        plot_param = draw_axes(draw, rect_param, "iteration", fit_param)
        draw_polyline(draw, iters, param_hist, param_bounds, plot_param, COLORS["Fit"])
        draw_horizontal(draw, float(summary["parameter_values"]["ground_truth"]), param_bounds, plot_param, COLORS["GT"], "GT")
        draw_horizontal(draw, float(summary["parameter_values"]["initial"]), param_bounds, plot_param, COLORS["Init"], "Init")

    all_x = np.concatenate([target[:, 0], init_pred[:, 0], final_pred[:, 0], np.array([0.0])])
    all_z = np.concatenate([target[:, 2], init_pred[:, 2], final_pred[:, 2], np.array([0.0])])
    traj_bounds = (
        float(all_x.min()) - 0.08,
        float(all_x.max()) + 0.08,
        float(all_z.min()) - 0.08,
        float(all_z.max()) + 0.08,
    )
    plot_tip = draw_axes(draw, rect_tip, "world x", "world z")
    draw_polyline(draw, target[:, 0], target[:, 2], traj_bounds, plot_tip, COLORS["GT"])
    draw_polyline(draw, init_pred[:, 0], init_pred[:, 2], traj_bounds, plot_tip, COLORS["Init"])
    draw_polyline(draw, final_pred[:, 0], final_pred[:, 2], traj_bounds, plot_tip, COLORS["Fit"])
    draw.text((plot_tip[0] + 10, plot_tip[1] + 10), "GT", fill=COLORS["GT"], font=FONT_SMALL)
    draw.text((plot_tip[0] + 55, plot_tip[1] + 10), "Init", fill=COLORS["Init"], font=FONT_SMALL)
    draw.text((plot_tip[0] + 110, plot_tip[1] + 10), "Fit", fill=COLORS["Fit"], font=FONT_SMALL)

    text_lines = [
        f"Fit parameter: {fit_param}",
        f"GT: {summary['parameter_values']['ground_truth']:.6f}",
        f"Init: {summary['parameter_values']['initial']:.6f}",
        f"Final fit: {summary['parameter_values']['final']:.6f}",
        f"Best seen: {summary['parameter_values']['best_seen']:.6f}",
        f"Initial loss: {summary['metrics']['initial_loss']:.3e}",
        f"Final loss: {summary['metrics']['final_loss']:.3e}",
        f"Initial RMSE: {summary['metrics']['initial_rmse']:.3e}",
        f"Final RMSE: {summary['metrics']['final_rmse']:.3e}",
        f"Loss improvement: {summary['metrics']['loss_improvement']:.3e}",
        f"Endpoint error init: {summary['tip_trajectory']['initial_endpoint_error']:.3e}",
        f"Endpoint error fit: {summary['tip_trajectory']['fitted_endpoint_error']:.3e}",
        f"Iterations: {summary['history']['iterations']}",
        f"Selection: {summary['metrics']['selection']}",
    ]
    y = rect_text[1] + 78
    for line in text_lines:
        draw.text((rect_text[0] + 18, y), line, fill=TEXT if y < rect_text[1] + 170 else MUTED, font=FONT_SMALL)
        y += 22

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    p = argparse.ArgumentParser(description="Generate compact summary.json and summary.png for a sysID result.json.")
    p.add_argument("result_json", type=Path)
    p.add_argument("--output-json", type=Path)
    p.add_argument("--output-png", type=Path)
    args = p.parse_args()

    data = load_result(args.result_json)
    summary = build_summary(data, args.result_json)

    output_json = args.output_json or args.result_json.with_name("summary.json")
    output_png = args.output_png or args.result_json.with_name("summary.png")
    output_json.write_text(json.dumps(summary, indent=2))
    make_summary_png(data, summary, output_png)

    print(json.dumps({"summary_json": str(output_json), "summary_png": str(output_png)}, indent=2))


if __name__ == "__main__":
    main()
