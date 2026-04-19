import os

from src.dashboard_app import main as legacy_main
from src.dashboard_v3 import main as v3_main


def main() -> None:
    use_legacy = os.getenv("VOC_DASHBOARD_LEGACY", "").strip().lower() in {"1", "true", "yes", "on"}
    if use_legacy:
        legacy_main()
        return
    v3_main()


if __name__ == "__main__":
    main()
