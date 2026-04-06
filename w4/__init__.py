"""S&OP 决策中枢 v4.0 — Tool Calling Agent + 多轮记忆。

将仓库中的 `w3/` 置于 sys.path 最前，使 `import sop_hub` 始终使用 w3 内置包（与作业「w4 基于 w3」一致）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_W3_ROOT = Path(__file__).resolve().parent.parent / "w3"
_s = str(_W3_ROOT.resolve())
if _s not in sys.path:
    sys.path.insert(0, _s)
