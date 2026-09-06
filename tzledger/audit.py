# -*- coding: utf-8 -*-
r"""稽核工具 —— 回答「這批資料是怎麼進來的」與「哪裡還沒遷移」。

載入層擋得住單一檔案的時區問題,但擋不住兩類更難發現的缺陷:

* **繞過。** 有腳本根本沒用這一層,直接 ``pd.read_csv``。
  載入層對它一無所知,ledger 是空的 —— 而空的 ledger 看起來像「乾淨」。
* **漏檔。** 每一個載進來的檔都判定正確、都沒報錯,但**少載了一半**。
  每一行檢查都通過,結論卻建立在四分之一的資料上。

這個模組處理這兩類。
"""
from __future__ import annotations

import glob
import ntpath
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

from .core import Loader, sniff_format

# ══════════════════════════════════════════════════════════════
#   一、判定覆蓋率 —— 全樹盤點
# ══════════════════════════════════════════════════════════════
def audit_corpus(loader: Loader, patterns: Sequence[str] = ('**/*.csv',),
                 root: Optional[Path] = None) -> pd.DataFrame:
    """盤點語料樹的判定覆蓋率。

    回傳每個檔的 ``(verdict, how, fmt)``。重點在於**看 how 欄的分布**,
    而不只是看有幾個檔沒判定:
    大量 ``INHERITED_BY_BASENAME`` 代表判定表的路徑與實際佈局已經對不上,
    那是一個即將失效的狀態,即使現在還沒有任何檔被擋下來。
    """
    base = Path(root) if root is not None else loader.config.root
    files = sorted({
        f for pattern in patterns
        for f in glob.glob(str(base / pattern), recursive=True)
        if os.path.isfile(f)
    })
    rows = []
    for f in files:
        verdict, how = loader.resolve_tz(f)
        rows.append(dict(
            file=ntpath.basename(f),
            verdict=verdict,
            how=how,
            fmt=sniff_format(f),
            rel_dir=os.path.relpath(os.path.dirname(f), str(base)),
            path=f,
        ))
    return pd.DataFrame(rows, columns=['file', 'verdict', 'how', 'fmt', 'rel_dir', 'path'])


def blocked_files(audit: pd.DataFrame) -> pd.DataFrame:
    """會觸發 fail-closed 的檔 = 待判定清單。"""
    if not len(audit):
        return audit
    bad = audit['verdict'].isna() | audit['verdict'].isin(['UNRESOLVED', 'TOO_SHORT'])
    return audit[bad]


# ══════════════════════════════════════════════════════════════
#   二、遷移清單 —— 誰還在繞過這一層
# ══════════════════════════════════════════════════════════════
_READ_CALL = re.compile(r'(?:pd|pandas)\.read_csv\s*\(\s*([^)]{0,160})')


def find_direct_reads(root, corpus_hints: Sequence[str],
                      skip_names: Iterable[str] = ()) -> pd.DataFrame:
    """掃出仍直接 ``read_csv`` 讀語料的腳本 = 遷移清單。

    這是靜態的字串比對,會漏(組出來的路徑、包一層函式的),
    所以它給的是**下界**:「至少這些地方還沒遷移」。
    刻意不做得更聰明 —— 一個會漏的清單只要誠實標示自己會漏,就仍然有用;
    一個聲稱完整但其實會漏的清單則會製造虛假的安全感。

    ``imports_loader`` 欄特別值得看:已經 import 了本層卻還在直讀的腳本,
    代表遷移做了一半,那通常比完全沒遷移更危險 —— 同一支程式裡
    有些序列修正過、有些沒有,而它們最後會被 join 在一起。
    """
    skip = set(skip_names)
    rows = []
    for py in glob.glob(str(Path(root) / '**' / '*.py'), recursive=True):
        name = ntpath.basename(py)
        if name in skip:
            continue
        try:
            with open(py, encoding='utf-8', errors='replace') as fh:
                src = fh.read()
        except OSError:
            continue
        uses_loader = 'tzledger' in src
        for match in _READ_CALL.finditer(src):
            arg = match.group(1)
            if any(h in arg for h in corpus_hints):
                rows.append(dict(
                    script=os.path.relpath(py, str(root)),
                    line=src[:match.start()].count('\n') + 1,
                    arg=arg.strip()[:70],
                    imports_loader=uses_loader,
                ))
    return pd.DataFrame(rows, columns=['script', 'line', 'arg', 'imports_loader'])


# ══════════════════════════════════════════════════════════════
#   三、跨序列覆蓋率 —— 抓「每個檔都對,但少載了一半」
# ══════════════════════════════════════════════════════════════
def coverage(frames: Dict[str, pd.DataFrame], min_density: float = 0.8,
             ts_col: str = 'ts') -> pd.DataFrame:
    r"""比較多個序列的**資料密度**,標記離群者。

    立法案例
    --------
    一組回測腳本繞過載入層,用精確檔名樣式 ``{symbol}_1min_{year}.csv`` 逐年讀取。
    但其中一個標的在磁碟上是**半年一檔**(``..._2023_01010630.csv``),
    於是那個樣式**只命中 6 年裡的 2 年**。

    結果:該標的聚合成 1 小時 bar 之後只有 **12,341** 根,其餘標的是 39,000–45,000 根。
    差了將近四倍。**那個數字一直印在畫面上,但沒有人拿它跟別的標的比。**

    每一項檢查都通過了 —— 檔案讀得到、時間戳解析正常、沒有 NaT、沒有例外。
    缺的不是檢查,是**比較的對象**。

    為什麼用密度而不是列數比值
    --------------------------
    第一版用「列數 ÷ 全體中位列數」判離群,結果誤報:
    有的標的上市較晚,只有 2.93 年的歷史,列數比值 0.579 —— 但那是**合法的**。

    改看**密度 = 列數 ÷ 年數**再與中位比:
      * 漏檔案例:12,341 根 ÷ 5.41 年 = 2,281/年,而 1 小時 bar 應為 8,760/年 → 密度 **0.26**
      * 上市較晚:526,648 列/年 vs 1 分 bar 應為 525,600/年 → 密度 **1.002**

    密度抓得到前者,又不會誤判後者。

    參數
    ----
    frames
        ``{序列名稱: DataFrame}``,每個 DataFrame 需有 tz-aware 的 ``ts`` 欄。
    min_density
        密度低於中位數的這個比例 → 標記離群。

    回傳
    ----
    每個序列一列的 DataFrame,含 ``density`` 與 ``outlier`` 欄。
    """
    rows = []
    for name, df in frames.items():
        if df is None or not len(df):
            rows.append(dict(series=name, rows=0, start=None, end=None,
                             years=0.0, per_year=float('nan')))
            continue
        span_years = ((df[ts_col].max() - df[ts_col].min()).total_seconds()
                      / (365.25 * 86400))
        # 只在跨度為零(單列、或全部時間戳相同)時放棄計算。
        # 這裡原本卡了一個 `> 0.1 年` 的門檻,結果是**跨度不足五週的序列
        # 全部算不出密度,於是全部被標成離群** —— 一個為了防除以零而設的保險,
        # 反而讓正常的短序列觸發警報。密度是相對中位數的比值,
        # 跨度大小不影響比較的有效性,只有跨度為零才真的無法計算。
        rows.append(dict(
            series=name,
            rows=len(df),
            start=str(df[ts_col].min())[:10],
            end=str(df[ts_col].max())[:10],
            years=round(span_years, 3),
            per_year=round(len(df) / span_years, 0) if span_years > 0 else float('nan'),
        ))

    cov = pd.DataFrame(rows, columns=['series', 'rows', 'start', 'end', 'years', 'per_year'])
    usable = cov['per_year'].notna() & (cov['per_year'] > 0)
    if usable.any():
        median = float(cov.loc[usable, 'per_year'].median())
        cov['density'] = (cov['per_year'] / median).round(3)
    else:
        cov['density'] = float('nan')
    # 密度算不出來(空序列)也算離群 —— 「沒有資料」不該安靜通過
    cov['outlier'] = ~(cov['density'] >= min_density)
    return cov


def assert_coverage(cov: pd.DataFrame) -> None:
    """有離群就丟出 —— 正式回測建議呼叫這個,而不是只印出來。

    印出來的警告與沒有警告在自動化流程裡是同一件事:都不會擋住下一步。
    """
    bad = cov[cov['outlier']]
    if not len(bad):
        return
    detail = ', '.join(
        f"{r['series']}(密度 {r['density']:.0%} of 中位;"
        f"{int(r['rows']):,} 列 / {r['years']:.3f} 年)"
        for _, r in bad.iterrows())
    raise ValueError(
        f'[tzledger] 跨序列覆蓋率離群:{detail}\n'
        f'  處置:先確認是「該序列真的比較短」還是「載入漏檔」。\n'
        f'  前者調低 min_density,後者必須修載入 —— 不要調參數繞過。')


def format_coverage(cov: pd.DataFrame) -> str:
    """把覆蓋率表排成固定寬度,方便貼進報告或 log。"""
    lines = ['  {:<12}{:>11}{:>12}{:>12}{:>9}{:>12}{:>8}'.format(
        '序列', '列數', '起', '訖', '年', '列/年', '密度')]
    for _, r in cov.iterrows():
        mark = '  ★離群' if r['outlier'] else ''
        per_year = r['per_year'] if pd.notna(r['per_year']) else 0
        density = r['density'] if pd.notna(r['density']) else float('nan')
        lines.append('  {:<12}{:>11,}{:>12}{:>12}{:>9.3f}{:>12,.0f}{:>8.3f}{}'.format(
            str(r['series']), int(r['rows']), str(r['start']), str(r['end']),
            r['years'], per_year, density, mark))
    return '\n'.join(lines)
