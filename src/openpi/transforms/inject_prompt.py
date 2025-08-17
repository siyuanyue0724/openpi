# openpi/transforms/inject_prompt.py

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional
import json
import pathlib
import numpy as np


@dataclass
class InjectPromptFromTaskIndex:
    """
    根据样本中的 task_index 注入自然语言 prompt。

    用法（两选一）：
      1) 指定 JSONL 路径：tasks_jsonl="/path/to/meta/tasks.jsonl"
         - JSONL 每行需包含 "task_index" 和 "task"（若无 "task"，会回退到 "instruction"/"prompt"）
         - 也接受别名参数 path="/path/to/meta/tasks.jsonl"
      2) 直接传映射：idx2task={0: "do X", 1: "do Y"}

    其他参数：
      - task_index_key: 数据样本里保存 task 索引的键名（默认 "task_index"）
      - dst_key       : 写入 prompt 的键名（默认 "prompt"）
      - strict        : 缺键或找不到映射时是否抛错（默认 True）
    """

    # 数据来源（二选一）：预载入映射 或 JSONL 路径
    idx2task: Optional[Mapping[int, str]] = None
    tasks_jsonl: Optional[str] = None
    # 别名，等价于 tasks_jsonl，方便有人更习惯用 path
    path: Optional[str] = None

    # 键名配置
    task_index_key: str = "task_index"
    dst_key: str = "prompt"

    # 行为：严格模式会在缺失/找不到映射时抛错
    strict: bool = True

    # 运行时缓存（构造时填充）
    _map: Dict[int, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # 只允许提供一种来源
        has_map = self.idx2task is not None
        has_file = (self.tasks_jsonl is not None) or (self.path is not None)
        if has_map and has_file:
            raise ValueError("InjectPromptFromTaskIndex: 请仅提供 idx2task 或 tasks_jsonl/path 其中之一。")
        if not has_map and not has_file:
            raise ValueError("InjectPromptFromTaskIndex: 需要提供 idx2task 或 tasks_jsonl/path。")

        if has_map:
            # 规范化：key 转 int，value 转 str
            self._map = {int(k): str(v) for k, v in dict(self.idx2task).items()}  # type: ignore[arg-type]
            return

        # 从 JSONL 读取
        p = pathlib.Path(self.tasks_jsonl or self.path)  # type: ignore[arg-type]
        if not p.exists():
            raise FileNotFoundError(f"tasks_jsonl 不存在: {p}")

        mapping: Dict[int, str] = {}
        with p.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception as e:
                    raise ValueError(f"解析 JSON 失败 {p}:{lineno}: {e}") from e

                # 允许多种字段名
                idx = row.get("task_index", row.get("id"))
                text = row.get("task", row.get("instruction", row.get("prompt")))
                if idx is None or text is None:
                    continue
                try:
                    mapping[int(idx)] = str(text)
                except Exception:
                    # 忽略无法转型的行
                    continue

        if not mapping and self.strict:
            raise ValueError(f"未在 {p} 解析到任何 (task_index, task) 映射。")
        self._map = mapping

    def __call__(self, sample: dict) -> dict:
        # 取出 task_index
        if self.task_index_key not in sample:
            if self.strict:
                raise KeyError(f"样本缺少键：{self.task_index_key}")
            sample.setdefault(self.dst_key, "")
            return sample

        v = sample[self.task_index_key]

        # 兼容标量 / numpy / torch.Tensor / [1] 等
        try:
            if hasattr(v, "item"):
                v = v.item()
            elif isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
                v = v[0]
            v = int(v)
        except Exception:
            if self.strict:
                raise
            sample.setdefault(self.dst_key, "")
            return sample

        # 查映射
        if v not in self._map:
            if self.strict:
                raise KeyError(f"未找到 task_index={v} 的映射")
            sample.setdefault(self.dst_key, "")
            return sample

        sample[self.dst_key] = self._map[v]
        return sample
