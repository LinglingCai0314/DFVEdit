"""
Configuration loader for DFVEdit.

Supports:
- New simplified format (task-specific only)
- Legacy format (backward compatible)
- Model registry for defaults
- Environment variable substitution
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Union

from omegaconf import OmegaConf

from dfvedit.config.schema import (
    DFVEditConfig,
    VideoConfig,
    EditingConfig,
    ModelConfig,
    TokenAmplifyConfig,
    SaveConfig,
)


class ConfigLoader:
    """
    Configuration loader with support for multiple formats.

    Features:
    - Environment variable substitution (${VAR_NAME})
    - Model registry for centralized defaults
    - Legacy format conversion
    - Task-specific config merging with defaults
    """

    # Model registry path (relative to package)
    DEFAULT_MODEL_REGISTRY = Path(__file__).resolve().parents[2] / "config" / "video_editing" / "models.yaml"

    # Registry cache
    _model_registry: Optional[Dict[str, Any]] = None

    @classmethod
    def get_env_vars(cls) -> Dict[str, str]:
        """Get environment variables for path substitution."""
        return {
            "CKPT_ROOT": os.environ.get("CKPT_ROOT", "/mnt/workspace/cailingling/models"),
            "DATA_ROOT": os.environ.get("DATA_ROOT", "/mnt/workspace/cailingling/data"),
            "OUTPUT_ROOT": os.environ.get("OUTPUT_ROOT", "./experiments"),
        }

    @classmethod
    def substitute_env_vars(cls, config_str: str) -> str:
        """Substitute environment variables in config string."""
        env_vars = cls.get_env_vars()
        for key, value in env_vars.items():
            config_str = config_str.replace(f"${{{key}}}", value)
            config_str = config_str.replace(f"${key}", value)
        return config_str

    @classmethod
    def load_model_registry(cls, registry_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load model registry with default configurations."""
        if cls._model_registry is not None:
            return cls._model_registry

        registry_path = registry_path or cls.DEFAULT_MODEL_REGISTRY
        if not registry_path.exists():
            print(f"Warning: Model registry not found at {registry_path}, using defaults")
            return {}

        with open(registry_path, "r", encoding="utf-8") as f:
            content = cls.substitute_env_vars(f.read())

        registry = OmegaConf.create(content)
        result = OmegaConf.to_container(registry, resolve=True)
        cls._model_registry = result if isinstance(result, dict) else {}
        return cls._model_registry

    @classmethod
    def load(cls, config_path: Union[str, Path], model_name: Optional[str] = None) -> DFVEditConfig:
        """
        Load configuration from file.

        Args:
            config_path: Path to config file (YAML)
            model_name: Optional model name override (cogvideox, wanx)

        Returns:
            DFVEditConfig object
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load and substitute env vars
        with open(config_path, "r", encoding="utf-8") as f:
            content = cls.substitute_env_vars(f.read())

        raw_config = OmegaConf.create(content)

        # Detect and handle legacy format
        if cls._is_legacy_format(raw_config):
            from dfvedit.utils.logging import get_logger

            get_logger().info("Detected legacy config format, converting...")
            return cls._convert_legacy_config(raw_config, model_name=model_name)

        # New format: merge with model defaults
        return cls._merge_with_defaults(raw_config, model_name)

    @classmethod
    def _is_legacy_format(cls, config: Any) -> bool:
        """Check if config uses legacy format."""
        return "dataset_config" in config or "editing_config" in config

    @classmethod
    def _convert_legacy_config(cls, raw_config: Any, model_name: Optional[str] = None) -> DFVEditConfig:
        """Convert legacy config format to new format."""
        dataset = raw_config.get("dataset_config", {})
        editing = raw_config.get("editing_config", {})

        registry = cls.load_model_registry()
        defaults = registry.get("defaults", {})

        resolved_model_name = model_name or raw_config.get("model", "wanx")
        model_defaults = registry.get("models", {}).get(resolved_model_name, {})
        model_path = raw_config.get("pretrained_model_path", "") or model_defaults.get("path", "")

        return DFVEditConfig(
            input=dataset.get("input_path", ""),
            output=dataset.get("output_path", ""),
            mask=dataset.get("mask_path", ""),
            model=ModelConfig(
                name=resolved_model_name,
                path=model_path,
                dtype=raw_config.get("dtype", defaults.get("dtype", "bfloat16")),
            ),
            video=VideoConfig(
                height=dataset.get("height", 512),
                width=dataset.get("width", 512),
                num_frames=dataset.get("n_sample_frame", 28),
                start_frame=dataset.get("starting_frame", 0),
                fps=dataset.get("sampling_rate", 8),
            ),
            editing=EditingConfig(
                prompt_original=dataset.get("source_prompt", ""),
                prompt_target=dataset.get("target_prompt", ""),
                num_inference_steps=editing.get("num_inference_steps", 50),
                guidance_scale_source=editing.get("guidance_scale_source", 5.0),
                guidance_scale_target=editing.get("guidance_scale_target", 15.0),
                amplitude=editing.get("amplitude", 1.0),
            ),
            seed=raw_config.get("seed", defaults.get("seed", 42)),
            device=raw_config.get("device", defaults.get("device", "cuda")),
            debug=raw_config.get("debug", False),
            debug_tokens=raw_config.get("debug_tokens", False),
        )

    @classmethod
    def _merge_with_defaults(cls, raw_config: Any, model_name: Optional[str] = None) -> DFVEditConfig:
        """Merge task config with model defaults."""
        registry = cls.load_model_registry()
        defaults = registry.get("defaults", {})

        # Determine model name
        model_name = model_name or raw_config.get("model", "wanx")
        model_defaults = registry.get("models", {}).get(model_name, {})

        # Build video config
        video_defaults = model_defaults.get("video", {})
        video_config = VideoConfig(
            height=raw_config.get("height", video_defaults.get("height", 512)),
            width=raw_config.get("width", video_defaults.get("width", 864)),
            num_frames=raw_config.get("num_frames", video_defaults.get("num_frames", 40)),
            start_frame=raw_config.get("start_frame", 0),
            fps=raw_config.get("fps", video_defaults.get("fps", 10)),
        )

        # Build token amplify config
        token_amplify_raw = raw_config.get("token_amplify", {})
        token_amplify_config = TokenAmplifyConfig(
            words=token_amplify_raw.get("words", []),
            amplitude=token_amplify_raw.get("amplitude", raw_config.get("amplitude", 1.0)),
            enabled=token_amplify_raw.get("enabled", True),
        )

        # Build save config
        save_raw = raw_config.get("save", {})
        save_config = SaveConfig(
            every=save_raw.get("every", 5),
            steps=save_raw.get("steps"),
        )

        # Build editing config
        editing_config = EditingConfig(
            prompt_original=raw_config.get("prompt_original", ""),
            prompt_target=raw_config.get("prompt_target", ""),
            num_inference_steps=raw_config.get("num_inference_steps", defaults.get("num_inference_steps", 50)),
            guidance_scale_source=raw_config.get("guidance_scale_source", defaults.get("guidance_scale_source", 5.0)),
            guidance_scale_target=raw_config.get("guidance_scale_target", defaults.get("guidance_scale_target", 15.0)),
            amplitude=raw_config.get("amplitude", 1.0),
            token_amplify=token_amplify_config,
            save=save_config,
        )

        # Build model config
        model_config = ModelConfig(
            name=model_name,
            path=model_defaults.get("path", ""),
            dtype=raw_config.get("dtype", defaults.get("dtype", "bfloat16")),
        )

        return DFVEditConfig(
            input=raw_config.get("input", ""),
            output=raw_config.get("output", ""),
            mask=raw_config.get("mask", ""),
            model=model_config,
            video=video_config,
            editing=editing_config,
            seed=raw_config.get("seed", defaults.get("seed", 42)),
            device=raw_config.get("device", defaults.get("device", "cuda")),
            debug=raw_config.get("debug", False),
            debug_tokens=raw_config.get("debug_tokens", False),
        )
