# -*- coding: utf-8 -*-
r"""相容層 —— 讓既有腳本一行遷移。

這個模組是**過渡工具**,不是目標狀態。新程式請直接呼叫 ``Loader.read_ts``。
之所以還是要有它,是因為現實情況通常是:發現污染的當下,已經有幾十支腳本
散落各處直接 ``pd.read_csv`` 讀語料,而且它們的邏輯與時區無關、
只是碰巧吃到了壞掉的輸入。要求全部改寫才能重跑,等於要求先做完最大的一件事
才能開始做第二大的 —— 實務上的結果就是兩件都不做。

與「舊 shim」的三個差別
-----------------------
這一層的前身是一支 monkeypatch ``pd.read_csv`` 的過渡腳本。它會失效,原因是:

1. **預設關閉。** 沒設環境變數 ``TZ_FIX=1`` 就靜默拿到未修正的資料。
   於是「忘記設」與「這批資料本來就乾淨」在輸出上完全無法區分。
   → 本層無開關,``install()`` 之後永遠生效。

2. **輸出仍是 tz-naive,而且用減法。** 舊版做 ``t - Timedelta(hours=8)``,
   結果還是 naive。這代表**它自己無法判斷一份資料修過了沒有** ——
   於是與別的工具疊加、或不小心跑兩次,就會累加成 16 小時位移。
   實測疊加後與正確資料的逐點相等率掉到 2–8%,價格差到四位數,
   而且**沒有任何錯誤訊息**。
   → 本層輸出 tz-aware UTC,``tz_convert('UTC')`` 對已 aware 的資料是 no-op,
   所以**重複套用安全**。

3. **逃生門是全域環境變數。** ``TZ_ALLOW_UNRESOLVED=1`` 一開,全部放行。
   → 本層的放行是 ``install(allow_unresolved=True)`` 顯式傳入,而且一定寫進 ledger。

冪等為什麼是關鍵性質
--------------------
一個修正操作如果不冪等,那麼「這份資料修過了嗎」就變成必須靠人記得的事。
靠人記得的事在多人、多腳本、多次重跑的環境下必然會出錯,
而且出錯的方向恰好是**看不出來**的那一種(值變了、型別沒變、沒有例外)。

冪等把這個問題消掉:修過的再修一次沒事,所以不需要有人記得。
"""
from __future__ import annotations

import ntpath
from typing import Optional, Sequence

import pandas as pd

from .config import BAD_VERDICTS
from .core import Loader, LedgerEntry, UnresolvedTimezone, _raw_read

_ORIGINAL_READ_CSV = None
_INSTALLED = False


def is_installed() -> bool:
    return _INSTALLED


def install(loader: Loader, allow_unresolved: bool = False,
            ts_cols: Sequence[str] = ('timestamp',), naive: bool = False) -> None:
    r"""把 ``pd.read_csv`` 導向本層 —— **僅對語料檔生效,其餘原樣通過**。

    參數
    ----
    loader
        提供設定與 ledger 的載入器實例。
    allow_unresolved
        放行未判定的檔(當作 UTC)。預設 fail-closed。
    ts_cols
        要嘗試的時間欄名。第一個在 DataFrame 裡找得到的就用它。
    naive
        修正時區後**把 tz 標記拿掉**(值 = 真 UTC,但 dtype 是 tz-naive)。

        這個選項存在的理由很具體:舊腳本裡大量存在
        ``df[df.timestamp > '2023-01-01']`` 這種與裸字串的比較,
        對 tz-aware 欄位會直接丟
        ``TypeError: Invalid comparison between dtype=datetime64[ns, UTC] and str``。
        如果遷移必須同時改邏輯,就沒有人會遷移。
        ``naive=True`` 讓**值被修正**而**介面不變**。

        代價是輸出看不出時區,所以**只用於既有腳本的回推重跑**;
        新程式一律用 tz-aware。

    注意
    ----
    monkeypatch 是有代價的手段:它讓「這行 ``read_csv`` 到底做了什麼」
    取決於別處有沒有呼叫過 ``install()``。這裡接受這個代價,
    是因為替代方案(不遷移、或先改寫幾十支腳本)更糟。
    但正因如此,本層只碰檔名命中 ``corpus_hints`` 的檔,
    其餘一律原樣通過 —— 讀設定檔、讀報表的 ``read_csv`` 不該受影響。
    """
    global _ORIGINAL_READ_CSV, _INSTALLED

    # ★防自我遞迴:重複安裝會讓 _ORIGINAL 指向已經 patch 過的函式,
    #   於是解除安裝時「還原」成 patch 版本,而且從此無法再還原。
    if _INSTALLED or getattr(pd.read_csv, '_tzledger_patched', False):
        return
    _ORIGINAL_READ_CSV = pd.read_csv

    hints = tuple(loader.config.corpus_hints)

    def _patched(filepath_or_buffer, *args, **kwargs):
        frame = _raw_read(filepath_or_buffer, *args, **kwargs)

        try:
            raw_path = str(filepath_or_buffer)
            name = ntpath.basename(raw_path)
        except Exception:
            return frame                                  # 不是路徑(buffer 等)→ 原樣

        if not any(h in name for h in hints):
            return frame                                  # 非語料 → 原樣

        col = next((c for c in ts_cols if c in frame.columns), None)
        if col is None:
            return frame

        stamps = pd.to_datetime(frame[col], errors='coerce')

        # ★冪等分支:已經是 tz-aware → 只確保它是 UTC,不做任何位移
        if getattr(stamps.dt, 'tz', None) is not None:
            converted = stamps.dt.tz_convert('UTC')
            frame = frame.copy()
            frame[col] = converted.dt.tz_localize(None) if naive else converted
            loader.ledger.append(LedgerEntry(
                file=name, path=raw_path, verdict='ALREADY_AWARE',
                how='idempotent_noop', rows=len(frame)))
            return frame

        verdict, how = loader.resolve_tz(raw_path, col)
        if verdict in BAD_VERDICTS:
            loader.ledger.append(LedgerEntry(
                file=name, path=raw_path, verdict=verdict,
                how=how + '|BLOCKED', rows=len(frame)))
            if not allow_unresolved:
                raise UnresolvedTimezone(
                    f'[tzledger.install] fail-closed:「{name}」無時區判定'
                    f'(verdict={verdict})。\n'
                    f'  路徑:{raw_path}\n'
                    f'  處置:先判定該檔,或 install(..., allow_unresolved=True) 承擔後果。')
            verdict = 'UTC'
            how += '|ASSUMED_UTC_UNVERIFIED'

        if verdict == 'EPOCH':
            numeric = pd.to_numeric(frame[col], errors='coerce')
            valid = numeric.dropna()
            unit = 'ms' if len(valid) and float(valid.iloc[0]) > 1e11 else 's'
            stamps = pd.to_datetime(numeric, unit=unit, utc=True)
            how += f'|unit={unit}'
        else:
            # ★宣告式,不做算術
            stamps = stamps.dt.tz_localize(loader.config.tz_of[verdict])

        converted = stamps.dt.tz_convert('UTC')
        frame = frame.copy()
        frame[col] = converted.dt.tz_localize(None) if naive else converted
        loader.ledger.append(LedgerEntry(
            file=name, path=raw_path, verdict=verdict,
            how=how + ('|naive_out' if naive else ''), rows=len(frame)))
        return frame

    _patched._tzledger_patched = True
    pd.read_csv = _patched
    _INSTALLED = True


def uninstall() -> None:
    """還原 ``pd.read_csv``。主要供測試使用,讓測試之間不互相影響。"""
    global _ORIGINAL_READ_CSV, _INSTALLED
    if _ORIGINAL_READ_CSV is not None:
        pd.read_csv = _ORIGINAL_READ_CSV
    _ORIGINAL_READ_CSV = None
    _INSTALLED = False
