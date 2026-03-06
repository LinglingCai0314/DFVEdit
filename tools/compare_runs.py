#!/usr/bin/env python
import argparse
import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_video

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dfvedit import ConfigLoader
from dfvedit.core.pipeline_factory import build_pipe
from dfvedit.samplers.dfv_sampler import DFVSampler as NewDFVSampler
from dfvedit.text.t5_embed import encode_prompt as new_encode_prompt
from dfvedit.video.mask import process_mask_video, apply_mask_to_grad


def tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    x = t.detach().float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=lambda o: list(o) if hasattr(o, "__iter__") and not isinstance(o, (str, bytes, dict)) else str(o),
        )


def load_old_module(old_py: Path):
    spec = importlib.util.spec_from_file_location("old_dfvedit_wanx6", str(old_py))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_old(old_py: Path, old_cfg_path: Path, device: str) -> Dict[str, Any]:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    old = load_old_module(old_py)
    cfg = OmegaConf.load(str(old_cfg_path))
    dataset = cfg.dataset_config
    editing = cfg.editing_config

    model_id = "/root/autodl-tmp/models/Wan-AI/Wan2.1-T2V-14B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    pipe = WanVideoToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    pipe.to(device)

    input_video = load_video(dataset.input_path)
    start = int(dataset.starting_frame)
    num = int(dataset.n_sample_frame)
    rate = int(dataset.sampling_rate)
    input_video = input_video[start:(num + start + 1) * rate:rate]

    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
        "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
        "messy background, three legs, many people in the background, walking backwards"
    )

    with torch.no_grad():
        source_prompt_embeds, source_negative_prompt_embeds = old.encode_prompt(
            pipe=pipe,
            prompt=dataset.source_prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=device,
            amplitude=editing.amplitude,
        )
        embedding_source = torch.cat([source_negative_prompt_embeds, source_prompt_embeds], dim=0)

        target_prompt_embeds, target_negative_prompt_embeds = old.encode_prompt(
            pipe=pipe,
            prompt=dataset.target_prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=device,
            amplitude=editing.amplitude,
        )
        embedding_target = torch.cat([target_negative_prompt_embeds, target_prompt_embeds], dim=0)

        video = pipe.video_processor.preprocess_video(
            input_video,
            height=int(dataset.height),
            width=int(dataset.width),
        )
        video = video.to(device=device, dtype=pipe.vae.dtype)
        z_source = pipe.vae.encode(video)["latent_dist"].mean

        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(z_source.device, z_source.dtype)
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(z_source.device, z_source.dtype)
        z_source = (z_source - latents_mean) * latents_std

        mask = None
        if str(dataset.mask_path) != "0":
            input_mask = load_video(dataset.mask_path)
            input_mask = input_mask[start:(num + start + 1) * rate:rate]
            mask_lat = pipe.video_processor.preprocess_video(input_mask, height=int(dataset.height), width=int(dataset.width))
            mask_lat = mask_lat.to(device=device, dtype=pipe.vae.dtype)
            latent_tensor = pipe.vae.encode(mask_lat)["latent_dist"].mean
            mean = latent_tensor.mean()
            std = latent_tensor.std()
            latent_tensor = (latent_tensor - mean) / (std + 1e-6)
            mask_norm = (latent_tensor - torch.min(latent_tensor)) / (torch.max(latent_tensor) - torch.min(latent_tensor))
            mask_bin = torch.where(mask_norm > 0.4, torch.tensor(1.0, device=mask_norm.device), torch.tensor(0.0, device=mask_norm.device))
            mask = mask_bin[0][0].repeat(1, 16, 1, 1, 1).to(device=device, dtype=pipe.vae.dtype)

    num_steps = int(editing.num_inference_steps)
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas

    sampler = old.DFVSampler(device=device, pipe=pipe, config=editing)
    z_target = z_source.clone()

    k_step = max(2, min(num_steps - 2, num_steps // 2))
    tracked = {0, 1, k_step}
    z_stats = {}
    delta_stats = []

    for i in range(num_steps - 1):
        timestep = timesteps[i + 1].unsqueeze(0)
        step_index = i + 1
        grad, _ = sampler.get_cdvf(
            z_source,
            z_target,
            embedding_source,
            embedding_target,
            timestep=timestep,
            next_timestep=None,
            image_rotary_emb=None,
            optimizer=None,
            eps=None,
            step_index=step_index,
            guidance_scale_source=float(editing.guidance_scale_source),
            guidance_scale_target=float(editing.guidance_scale_target),
        )
        mask_applied = bool(i > 100 and mask is not None)
        if mask_applied:
            grad = grad * mask
        z_target = z_target + grad.to(z_target.dtype)

        delta_stats.append({
            "iter": i,
            "step_index": step_index,
            "mask_applied": mask_applied,
            **tensor_stats(grad),
        })

        if i in tracked:
            z_stats[f"iter_{i}"] = {
                "z_source": tensor_stats(z_source),
                "z_target": tensor_stats(z_target),
            }

    src_tokens = pipe.tokenizer.convert_ids_to_tokens(
        pipe.tokenizer([dataset.source_prompt], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt").input_ids[0]
    )
    old_words = ["▁Water", "color", "▁style"]
    hits = [tok for tok in src_tokens if tok in set(old_words)]

    result = {
        "config_resolved": {
            "input": str(dataset.input_path),
            "mask": str(dataset.mask_path),
            "output": str(dataset.output_path),
            "prompt_original": str(dataset.source_prompt),
            "prompt_target": str(dataset.target_prompt),
            "height": int(dataset.height),
            "width": int(dataset.width),
            "num_frames": int(dataset.n_sample_frame),
            "fps": int(dataset.sampling_rate),
            "start_frame": int(dataset.starting_frame),
            "seed": 42,
            "num_inference_steps": num_steps,
            "guidance_scale_source": float(editing.guidance_scale_source),
            "guidance_scale_target": float(editing.guidance_scale_target),
            "amplitude": float(editing.amplitude),
            "scheduler": "FlowMatchEulerDiscreteScheduler(shift=5.0)",
        },
        "scheduler": {
            "timesteps": [int(x.item()) for x in timesteps],
            "sigmas": [float(x.item()) for x in sigmas],
            "timesteps_dtype": str(timesteps.dtype),
            "sigmas_dtype": str(sigmas.dtype),
            "timesteps_device": str(timesteps.device),
            "sigmas_device": str(sigmas.device),
        },
        "z_stats": z_stats,
        "delta_stats": delta_stats,
        "prompt_embedding": {
            "source": tensor_stats(embedding_source),
            "target": tensor_stats(embedding_target),
            "token_amplify_enabled": True,
            "token_amplify_words": old_words,
            "token_hit_tokens": hits,
        },
        "latent_norm": {
            "latents_mean": [float(x) for x in pipe.vae.config.latents_mean],
            "latents_std_raw": [float(x) for x in pipe.vae.config.latents_std],
            "applied_formula": "z = (z - latents_mean) * (1/latents_std)",
            "denorm_formula": "z = z/(1/latents_std) + latents_mean",
        },
    }

    del pipe, vae, z_source, z_target
    torch.cuda.empty_cache()
    return result


def run_new(new_cfg_path: Path, device: str) -> Dict[str, Any]:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    cfg = ConfigLoader.load(new_cfg_path)
    pipe, _ = build_pipe(cfg, torch.device(device))

    input_video = load_video(cfg.input)
    start = int(cfg.video.start_frame)
    num = int(cfg.video.num_frames)
    rate = int(cfg.video.fps)
    input_video = input_video[start:(num + start + 1) * rate:rate]

    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
        "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
        "messy background, three legs, many people in the background, walking backwards"
    )

    with torch.no_grad():
        src_emb, src_neg = new_encode_prompt(
            pipe=pipe,
            prompt=cfg.editing.prompt_original,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            max_sequence_length=512,
            device=torch.device(device),
            token_amplify_config=cfg.editing.token_amplify,
            debug_tokens=False,
        )
        embedding_src = torch.cat([src_neg, src_emb], dim=0)

        tgt_emb, tgt_neg = new_encode_prompt(
            pipe=pipe,
            prompt=cfg.editing.prompt_target,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            max_sequence_length=512,
            device=torch.device(device),
            token_amplify_config=None,
            debug_tokens=False,
        )
        embedding_tgt = torch.cat([tgt_neg, tgt_emb], dim=0)

        video = pipe.video_processor.preprocess_video(input_video, height=cfg.video.height, width=cfg.video.width)
        video = video.to(device=device, dtype=pipe.vae.dtype)
        latents_src = pipe.vae.encode(video)["latent_dist"].mean

        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents_src.device, latents_src.dtype)
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents_src.device, latents_src.dtype)
        latents_src = (latents_src - latents_mean) * latents_std

        mask = None
        if cfg.mask:
            mask_frames = load_video(cfg.mask)
            mask_frames = mask_frames[start:(num + start + 1) * rate:rate]
            mask = process_mask_video(mask_frames, pipe, cfg.video.height, cfg.video.width, torch.device(device), debug_dir=None)

    num_steps = int(cfg.editing.num_inference_steps)
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas

    sampler = NewDFVSampler(pipe=pipe, device=torch.device(device), config=cfg.editing)
    latents_edit = latents_src.clone()

    k_step = max(2, min(num_steps - 2, num_steps // 2))
    tracked = {0, 1, k_step}
    z_stats = {}
    delta_stats = []

    for i in range(num_steps - 1):
        step_index = i + 1
        cdfv, _ = sampler.compute_cdfv(
            latents_src=latents_src,
            latents_edit=latents_edit,
            text_emb_src=embedding_src,
            text_emb_tgt=embedding_tgt,
            step_index=step_index,
            guidance_scale_src=float(cfg.editing.guidance_scale_source),
            guidance_scale_tgt=float(cfg.editing.guidance_scale_target),
        )
        mask_applied = bool(mask is not None and i > cfg.editing.mask_apply_after_step)
        if mask_applied:
            cdfv = apply_mask_to_grad(cdfv, mask)
        latents_edit = latents_edit + cdfv.to(latents_edit.dtype)

        delta_stats.append({
            "iter": i,
            "step_index": step_index,
            "mask_applied": mask_applied,
            **tensor_stats(cdfv),
        })

        if i in tracked:
            z_stats[f"iter_{i}"] = {
                "z_source": tensor_stats(latents_src),
                "z_target": tensor_stats(latents_edit),
            }

    token_amplify = dataclasses.asdict(cfg.editing.token_amplify)
    src_tokens = pipe.tokenizer.convert_ids_to_tokens(
        pipe.tokenizer([cfg.editing.prompt_original], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt").input_ids[0]
    )
    hit_set = set(token_amplify.get("words", []))
    hits = [tok for tok in src_tokens if tok in hit_set]

    result = {
        "config_resolved": dataclasses.asdict(cfg),
        "scheduler": {
            "timesteps": [int(x.item()) for x in timesteps],
            "sigmas": [float(x.item()) for x in sigmas],
            "timesteps_dtype": str(timesteps.dtype),
            "sigmas_dtype": str(sigmas.dtype),
            "timesteps_device": str(timesteps.device),
            "sigmas_device": str(sigmas.device),
        },
        "z_stats": z_stats,
        "delta_stats": delta_stats,
        "prompt_embedding": {
            "source": tensor_stats(embedding_src),
            "target": tensor_stats(embedding_tgt),
            "token_amplify_enabled": bool(cfg.editing.token_amplify.enabled),
            "token_amplify_words": cfg.editing.token_amplify.words,
            "token_hit_tokens": hits,
        },
        "latent_norm": {
            "latents_mean": [float(x) for x in pipe.vae.config.latents_mean],
            "latents_std_raw": [float(x) for x in pipe.vae.config.latents_std],
            "applied_formula": "z = (z - latents_mean) * (1/latents_std)",
            "denorm_formula": "z = z/(1/latents_std) + latents_mean",
        },
    }

    del pipe, latents_src, latents_edit
    torch.cuda.empty_cache()
    return result


def build_report(old_res: Dict[str, Any], new_res: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Alignment Diff Report")

    old_flat = flatten(old_res["config_resolved"])
    new_flat = flatten(new_res["config_resolved"])
    keys = sorted(set(old_flat.keys()) | set(new_flat.keys()))
    config_diff = []
    for k in keys:
        ov = old_flat.get(k, "<missing>")
        nv = new_flat.get(k, "<missing>")
        if ov != nv:
            config_diff.append((k, ov, nv))

    lines.append("\n## Config Diff")
    if not config_diff:
        lines.append("- No config differences")
    else:
        for k, ov, nv in config_diff[:80]:
            lines.append(f"- `{k}`: old=`{ov}` new=`{nv}`")

    old_sig = torch.tensor(old_res["scheduler"]["sigmas"], dtype=torch.float64)
    new_sig = torch.tensor(new_res["scheduler"]["sigmas"], dtype=torch.float64)
    min_len = min(len(old_sig), len(new_sig))
    sig_diff = (old_sig[:min_len] - new_sig[:min_len]).abs().max().item() if min_len else math.nan

    old_ts = torch.tensor(old_res["scheduler"]["timesteps"], dtype=torch.float64)
    new_ts = torch.tensor(new_res["scheduler"]["timesteps"], dtype=torch.float64)
    min_len_t = min(len(old_ts), len(new_ts))
    ts_diff = (old_ts[:min_len_t] - new_ts[:min_len_t]).abs().max().item() if min_len_t else math.nan

    lines.append("\n## Scheduler Diff")
    lines.append(f"- max|timesteps_old-timesteps_new| = {ts_diff:.6f}")
    lines.append(f"- max|sigmas_old-sigmas_new| = {sig_diff:.6e}")

    lines.append("\n## Key Tensor Stats Diff")
    lines.append("|item|old mean|new mean|abs diff|")
    lines.append("|---|---:|---:|---:|")

    candidate_iters = ["iter_0", "iter_1"]
    old_keys = set(old_res["z_stats"].keys())
    new_keys = set(new_res["z_stats"].keys())
    common = sorted(old_keys & new_keys)
    extra = [k for k in common if k not in {"iter_0", "iter_1"}]
    if extra:
        candidate_iters.append(extra[-1])  # K step

    for it in candidate_iters:
        if it in old_res["z_stats"] and it in new_res["z_stats"]:
            for name in ["z_source", "z_target"]:
                om = old_res["z_stats"][it][name]["mean"]
                nm = new_res["z_stats"][it][name]["mean"]
                lines.append(f"|{it}.{name}.mean|{om:.6f}|{nm:.6f}|{abs(om-nm):.6e}|")

    old_delta = old_res["delta_stats"]
    new_delta = new_res["delta_stats"]
    first_div = None
    for i in range(min(len(old_delta), len(new_delta))):
        d = abs(old_delta[i]["mean"] - new_delta[i]["mean"])
        if d > 1e-5:
            first_div = (i, d, old_delta[i]["mean"], new_delta[i]["mean"])
            break

    lines.append("\n## Delta Diff")
    if first_div is None:
        lines.append("- No significant mean diff (>1e-5) in compared steps")
    else:
        i, d, om, nm = first_div
        lines.append(f"- first significant delta mean diff at iter={i}: old={om:.6e}, new={nm:.6e}, abs_diff={d:.6e}")

    lines.append("\n## Prompt/Token Amplify")
    lines.append(
        f"- old enabled={old_res['prompt_embedding']['token_amplify_enabled']} "
        f"words={old_res['prompt_embedding']['token_amplify_words']} "
        f"hits={old_res['prompt_embedding']['token_hit_tokens']}"
    )
    lines.append(
        f"- new enabled={new_res['prompt_embedding']['token_amplify_enabled']} "
        f"words={new_res['prompt_embedding']['token_amplify_words']} "
        f"hits={new_res['prompt_embedding']['token_hit_tokens']}"
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old/new DFVEdit runs with aligned settings")
    parser.add_argument("--old-config", required=True)
    parser.add_argument("--new-config", required=True)
    parser.add_argument("--old-py", default="/root/autodl-tmp/old_dfvedit/dfvedit_wanx6.py")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/DFVEdit/output/debug")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    old_out = out_dir / "old"
    new_out = out_dir / "new"

    old_res = run_old(Path(args.old_py), Path(args.old_config), args.device)
    dump_json(old_out / "compare_stats.json", old_res)

    new_res = run_new(Path(args.new_config), args.device)
    dump_json(new_out / "compare_stats.json", new_res)

    report = build_report(old_res, new_res)
    report_path = out_dir / "compare_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote: {old_out / 'compare_stats.json'}")
    print(f"Wrote: {new_out / 'compare_stats.json'}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
