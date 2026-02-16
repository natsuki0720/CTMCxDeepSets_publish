#!/usr/bin/env python3
"""後方互換エントリーポイント。

新しい配置先は `scripts/data_generation/entrypoint_gen_with_MLE.py`。
既存オペレーション互換のため、この薄いラッパーを残す。
"""

from scripts.data_generation.entrypoint_gen_with_MLE import main


if __name__ == "__main__":
    main()
