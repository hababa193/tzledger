# -*- coding: utf-8 -*-
r"""載入層的設定物件。

原型版本把語料根目錄硬編在模組頂端(`ROOT = Path(r'C:\Users\...')`),
所以那個版本**只能在作者那台機器上跑**。這裡改成顯式設定物件:
路徑、判定表、黑名單、語料樣式都由呼叫端提供,模組本身不知道任何具體路徑。

判定表(manifest)的角色
----------------------
判定表是「這個檔的原始時間戳位於哪個時區」的**外部事實紀錄**,
由一次性的鑑定作業產出(逐檔比對已知無歧義的來源),不是猜的。
載入層只負責**查表 + 拒絕查不到的檔**,不負責判定 —— 這兩件事分開,
是因為判定要人看證據,而載入要能無人值守地跑。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

# verdict → 該檔「原始時間戳」所處的時區
TZ_OF: Dict[str, str] = {
    'TPE': 'Asia/Taipei',
    'UTC': 'UTC',
    'UTC_EXPLICIT': 'UTC',   # 檔案自述帶 +00:00 / Z
    'EPOCH': 'UTC',          # 數值時間戳,本質無時區歧義
}

# 這些 verdict 代表「沒判出來」→ fail-closed
BAD_VERDICTS = frozenset({'UNRESOLVED', 'TOO_SHORT', None})

# 檔名帶這些片段者視為「時間序列語料」(相容層只對這些檔生效)
DEFAULT_CORPUS_HINTS: tuple = ('_1min', '_1m', '_5m', '_1h', '_4h', '_8h', '_1d',
                               '_lsr', '_oi_', '_funding', '_ohlcv')


def norm_path(p) -> str:
    r"""路徑正規化 —— 大小寫、相對/絕對、分隔符全部拉平後才比對。

    Windows 上 ``C:\Data\x.csv`` 與 ``c:/data/X.CSV`` 是同一個檔,但字串不相等。
    判定表用字串當 key,不正規化就會查不到而誤觸 fail-closed ——
    也就是「明明判定過了卻讀不進來」。這種假警報會逼使用者去開放行開關,
    等於用一個安全機制的失誤磨掉另一個安全機制。
    """
    return os.path.normcase(os.path.abspath(str(p)))


def basename_any(raw) -> str:
    r"""取 basename,不管字串裡用的是哪種分隔符。

    判定表可能在 Windows 產出、在 Linux 讀(反之亦然),而
    ``Path('a\b.csv').name`` 在 Linux 上會整串回傳 ``a\b.csv``。
    先統一成 POSIX 分隔符再取名字,兩邊行為才一致。
    """
    return Path(str(raw).replace('\\', '/')).name


@dataclass
class LoaderConfig:
    """載入層的全部外部依賴。

    root
        語料根目錄。判定表裡的相對路徑以此為基準展開。
    manifests
        判定表 JSON 的路徑清單。格式 ``{"records": [{"path": ..., "verdict": ...}]}``。
        可以是空的 —— 沒有判定表時,只有「自述格式無歧義」的檔載得進來,
        其餘一律 fail-closed。這是刻意的:**空的判定表不等於全部放行。**
    blacklist
        檔名黑名單,永不載入。用於「已鑑定為內容與檔名不符」的檔。
    corpus_hints
        判定何謂「時間序列語料」的檔名片段。相容層只對命中的檔動手。
    tz_of / bad_verdicts
        通常不需要改;開放出來是為了讓測試能注入。
    """
    root: Path
    manifests: Sequence[Path] = field(default_factory=tuple)
    blacklist: frozenset = field(default_factory=frozenset)
    corpus_hints: tuple = DEFAULT_CORPUS_HINTS
    tz_of: Dict[str, str] = field(default_factory=lambda: dict(TZ_OF))
    bad_verdicts: frozenset = BAD_VERDICTS

    def __post_init__(self):
        self.root = Path(self.root)
        self.manifests = tuple(Path(m) for m in self.manifests)
        self.blacklist = frozenset(self.blacklist)

    # ── 判定表 ────────────────────────────────────────────────
    def load_manifests(self):
        """讀入全部判定表,回傳 ``(by_path, by_name)`` 兩層索引。

        兩層是刻意的:完整路徑優先,查不到才退到 basename。
        basename 命中會在 ledger 留下 ``INHERITED_BY_BASENAME`` 標記 ——
        因為「同名不同檔」是真實存在的風險,稽核時要看得出這筆是怎麼查到的。
        """
        by_path, by_name = {}, {}
        for m in self.manifests:
            if not m.exists():
                continue
            with open(m, encoding='utf-8') as fh:
                payload = json.load(fh)
            for rec in payload.get('records', []):
                raw = rec['path']
                full = raw if os.path.isabs(raw) else str(self.root / raw.replace('/', os.sep))
                by_path[norm_path(full)] = rec['verdict']
                by_name.setdefault(basename_any(raw), rec['verdict'])
        return by_path, by_name

    @classmethod
    def from_env(cls, **overrides) -> 'LoaderConfig':
        """從環境變數建構,方便 CI 與他人 clone 後直接跑。

        ``TZLEDGER_ROOT``       語料根目錄(預設 = 當前工作目錄)
        ``TZLEDGER_MANIFESTS``  以 ``os.pathsep`` 分隔的判定表路徑
        """
        root = Path(os.environ.get('TZLEDGER_ROOT', os.getcwd()))
        raw = os.environ.get('TZLEDGER_MANIFESTS', '')
        manifests = tuple(Path(p) for p in raw.split(os.pathsep) if p.strip())
        return cls(root=root, manifests=manifests, **overrides)
