# -*- coding: utf-8 -*-
r"""結論身分標記 —— 這次分析的結果,可不可以拿來用?

問題
----
時區污染修好之後,一個很自然的想法是「把之前的結論重跑一遍就好了」。
但這裡有個更前面的問題:**之前那些結論,有哪些是真的被污染影響到的?**

如果全部作廢,成本高到不會有人真的去做,實際結果是繼續引用舊結論;
如果全部保留重跑,又等於假設污染沒有影響 —— 那修它幹嘛。

實測出來的答案
--------------
把同一份資料的「未修正」與「已修正」兩版拿來逐點比對:

* **依位置比對(第 n 列對第 n 列)**:完全相同,相關係數 1.0000。
  這很合理 —— 修正只改了時間戳的標籤,沒有動任何一列的數值,也沒有重排順序。
* **依時間戳聚合成日線後比報酬**:相關係數只有 **0.535 – 0.612**。

同一份資料,兩種讀法,一個毫髮無傷,一個面目全非。

差別在於**有沒有用到時間戳去切邊界**。位移 8 小時不會改變任何一根 bar 的內容,
但會改變「哪些 bar 屬於同一天」—— 於是日線的開高低收全部重組。

三條傳導路徑
------------
從上面推出來:缺陷只透過這三種操作傳導。

1. **日/週界線** —— ``resample`` / ``1D`` / ``1W`` / ``.dt.date``
2. **跨來源或跨頻率對齊** —— ``merge`` / ``join`` / ``reindex`` / ``concat(axis=1)``
3. **日曆與時段判斷** —— ``.dt.hour`` / ``.dt.dayofweek`` / 用日期當切點

沒走這三條路的分析(純單一來源、順序式、只看相對位置)**不受影響**,結論可以留。

所以身分不是資料的屬性,是「資料 × 用法」的屬性
------------------------------------------------
本模組知道你**讀了什麼**(從 ledger),但無法知道你**怎麼用**。
所以 ``declare_derived_path`` 必須由呼叫端顯式傳入,沒有預設值 ——
給預設值就等於替使用者猜,而猜錯的方向是「以為結論還有效」。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional, TextIO

from .core import Ledger

DERIVED_PATHS = (
    '日/週界線(resample / 1D / 1W / W-FRI / .dt.date)',
    '跨來源或跨頻率對齊(merge / join / reindex / concat axis=1)',
    '日曆·時段判斷(.dt.hour / .dt.dayofweek / 日期切點)',
)

# ── 身分標籤 ────────────────────────────────────────────────
CLEAN_NO_CORRECTED_INPUT = 'CLEAN_NO_CORRECTED_INPUT'
"""本次沒有讀到任何需要時區修正的檔 → 結論不受影響。"""

CLEAN_PATH_NOT_AFFECTED = 'CLEAN_PATH_NOT_AFFECTED'
"""有讀到修正過的檔,但呼叫端聲明未走三條傳導路徑 → 結論不受影響。"""

UNTESTED = 'UNTESTED'
"""有讀到修正過的檔 **且** 走了傳導路徑 → **原結論不具檢定力,退回未測狀態。**"""

UNVERIFIED_INPUT = 'UNVERIFIED_INPUT'
"""曾用 ``allow_unresolved=True`` 放行未判定檔 → 身分不明,**比未測更差**。"""

UNKNOWN_NO_LEDGER = 'UNKNOWN_NO_LEDGER'
"""ledger 是空的 —— 沒有任何檔經過本層,代表資料是繞過去讀的。**不等於乾淨。**"""


@dataclass
class Provenance:
    """一次分析的資料身分。"""
    tag: str
    n_corrected: int
    n_clean: int
    n_assumed: int
    declared_derived_path: bool
    corrected_files: List[str] = field(default_factory=list)
    assumed_files: List[str] = field(default_factory=list)

    @property
    def citable(self) -> bool:
        """這次的結論可不可以引用。"""
        return self.tag in (CLEAN_NO_CORRECTED_INPUT, CLEAN_PATH_NOT_AFFECTED)


def classify(ledger: Ledger, declare_derived_path: bool) -> Provenance:
    """依 ledger 與呼叫端的宣告,判定本次分析的身分。

    參數
    ----
    declare_derived_path
        本次分析是否用到了 ``DERIVED_PATHS`` 列出的任一種操作。
        **必須顯式傳入** —— 本模組看得到你讀了什麼,看不到你怎麼用。
    """
    corrected = ledger.corrected()
    assumed = ledger.assumed()
    clean = [e for e in ledger
             if e.verdict in ('UTC', 'UTC_EXPLICIT', 'EPOCH', 'DATE_ONLY')
             and not e.assumed]

    if not len(ledger):
        # ★這一條是自己踩到之後補的。
        #   原版在 ledger 為空時回傳 CLEAN_… —— 但空的 ledger 代表
        #   **本次沒有任何檔案經過本層**(被 pd.read_parquet、裸 read_csv 讀走了),
        #   那是「不知道」,不是「乾淨」。
        #   這是 fail-open,而且與這整個模組要防的缺陷完全同型:
        #   一個安靜的、看起來像好消息的預設值。
        tag = UNKNOWN_NO_LEDGER
    elif assumed:
        tag = UNVERIFIED_INPUT
    elif not corrected:
        tag = CLEAN_NO_CORRECTED_INPUT
    elif not declare_derived_path:
        tag = CLEAN_PATH_NOT_AFFECTED
    else:
        tag = UNTESTED

    return Provenance(
        tag=tag,
        n_corrected=len(corrected),
        n_clean=len(clean),
        n_assumed=len(assumed),
        declared_derived_path=bool(declare_derived_path),
        corrected_files=[e.file for e in corrected],
        assumed_files=[e.file for e in assumed],
    )


def report(prov: Provenance, out: Optional[TextIO] = None) -> None:
    """把身分判定印成人看得懂的段落,貼進報告用。"""
    stream = out if out is not None else sys.stdout
    line = '=' * 78

    print(line, file=stream)
    print('  資料身分標記', file=stream)
    print(line, file=stream)
    print('  本次讀入 {} 檔 —— 需修正 {} / 本來就乾淨 {} / 放行未判定 {}'.format(
        prov.n_corrected + prov.n_clean + prov.n_assumed,
        prov.n_corrected, prov.n_clean, prov.n_assumed), file=stream)
    print('  呼叫端聲明「走了傳導路徑」= {}'.format(prov.declared_derived_path), file=stream)
    for i, path in enumerate(DERIVED_PATHS, 1):
        print('     {}. {}'.format(i, path), file=stream)
    print('  ★身分 = {}'.format(prov.tag), file=stream)

    if prov.tag == UNTESTED:
        print('  ⚠️ 讀到了修正過的檔,而且走了會傳導的路徑 → **原結論不具檢定力**。', file=stream)
        print('     處置:退回未測狀態,與全新的假設同等對待,重跑完整流程。', file=stream)
        print('     ⚠️ 重跑是**追加一次檢定**,不是回復原狀 ——', file=stream)
        print('        多重比較的負擔不會因為資料修好了就消失。', file=stream)
        for f in prov.corrected_files[:5]:
            print('     - {}'.format(f), file=stream)
    elif prov.tag == UNVERIFIED_INPUT:
        print('  ★★ 曾用 allow_unresolved=True 放行未判定檔 —— **身分不明,比未測更差**。', file=stream)
        print('     結論不得引用;先把這些檔判定完成再說。', file=stream)
        for f in prov.assumed_files[:5]:
            print('     - {}'.format(f), file=stream)
    elif prov.tag == UNKNOWN_NO_LEDGER:
        print('  ★★ 本次沒有任何檔案經過本層 —— 資料是**繞過去**讀的', file=stream)
        print('     (pd.read_parquet / 裸 read_csv 等)。**空的 ledger 不等於乾淨。**', file=stream)
        print('     處置:改用 Loader.read_ts,或在報告中自行說明', file=stream)
        print('     該資料源為何無時區歧義(例:epoch 毫秒)。', file=stream)

    print(line, file=stream)
