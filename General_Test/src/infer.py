from __future__ import annotations

try:
    from .extract_latents import main
except ImportError:
    from extract_latents import main


if __name__ == "__main__":
    main()
