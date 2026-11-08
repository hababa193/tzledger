# -*- coding: utf-8 -*-
r"""fail-closed 的時間序列載入層 —— 讀進來的 ``ts`` 欄保證是 tz-aware UTC。

設計立場
--------
這一層唯一的職責是:**讓「時區不明的資料被靜默地當成正確資料使用」變成不可能。**

不是「盡量正確」,是「不確定就不給」。因為時區污染的特性是:
它不會讓程式壞掉,只會讓結果**安靜地偏移**,而偏移過的結果與正確結果長得一模一樣。
沒有例外、沒有 traceback、沒有紅字 —— 你只會得到一個看起來很合理的錯誤答案。

三個設計決定
------------
1. **沒有全域開關。** 修正永遠生效。上一版工具的預設是關的(要設 ``TZ_FIX=1``),
   結果是「忘了設」與「這批資料本來就乾淨」在輸出上完全無法區分。
2. **不做時間算術。** ``TPE → UTC`` 用 ``tz_localize`` + ``tz_convert``,不用 ``- 8h``。
   在台灣兩者數值相同,但前者是**意圖的宣告**(這個時間戳原本是台北時間),
   後者是**對數字的操作**(把它減 8)。前者可稽核、對有日光節約的時區才正確,
   而且冪等 —— 重複套用是 no-op,重複減 8h 會累加成 16h。
3. **每次載入都留紀錄。** ledger 記下每個檔的判定、判定是怎麼查到的、列數、時間跨度。
   結論可不可信,取決於它建立在什麼資料上;那就必須有辦法事後查。

為什麼判定與載入要分開
----------------------
判定(這個檔的原始時間戳在哪個時區)需要人看證據 —— 比對已知無歧義的來源、
逐點檢查、看跨相關的峰值落在哪。載入需要能無人值守地跑。
把兩件事塞進同一個函式,結果一定是載入去猜判定,而猜錯不會有人知道。
所以這裡只查表,查不到就拒絕。
"""
from __future__ import annotations

import ntpath
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .config import BAD_VERDICTS, LoaderConfig, norm_path

# ★永遠指向未被 patch 的原版,在模組載入時就擷取。
#
# 這一行是整個模組最容易被刪掉、也最不能刪的一行。相容層(compat.install)會把
# ``pd.read_csv`` 導向本層;如果本層內部也用 ``pd.read_csv`` 讀檔,就會變成
#   sniff_format → patched read_csv → resolve_tz → sniff_format → ...
# 無限遞迴。而遞迴丟出的 RecursionError 會被 sniff_format 的 except 吞掉,
# 於是函式回傳 None,判定退化成「查無」,再被當成 UTC 處理 ——
# **一個該做的時區修正被靜默跳過,沒有任何錯誤訊息。**
_RAW_READ_CSV = pd.read_csv


def _raw_read(*args, **kwargs):
    """本層內部讀檔的唯一管道 —— 絕不能走 patch 後的 ``pd.read_csv``。理由見上。"""
    return _RAW_READ_CSV(*args, **kwargs)


class UnresolvedTimezone(RuntimeError):
    """時區判定缺失或不可信 —— fail-closed 時丟出。

    這是刻意設計成 exception 而不是 warning 的:warning 會被 log 淹掉,
    而這一層的全部價值就在於「不確定的資料不准通過」。
    """


# ══════════════════════════════════════════════════════════════
#   稽核紀錄
# ══════════════════════════════════════════════════════════════
@dataclass
class LedgerEntry:
    """一次載入的稽核紀錄。

    ``how`` 欄記的是「判定是怎麼查到的」,不是「判定是什麼」。
    這兩者要分開,因為 ``verdict='UTC'`` 從完整路徑查到、從同名檔繼承、
    或是被 ``allow_unresolved`` 放行後假設的,可信度天差地遠,
    但在 ``verdict`` 欄裡長得一模一樣。
    """
    file: str
    path: str
    verdict: Optional[str]
    how: str
    rows: int
    span: Tuple[Optional[str], Optional[str]] = (None, None)

    @property
    def assumed(self) -> bool:
        """這筆是否為「放行未判定檔」—— 身分不明,結論不得引用。"""
        return 'ASSUMED_UTC_UNVERIFIED' in self.how

    @property
    def corrected(self) -> bool:
        """這筆是否**實際被移動過時間**(原始不是 UTC)。"""
        return self.verdict not in (None, 'UTC', 'UTC_EXPLICIT', 'EPOCH', 'DATE_ONLY')


class Ledger:
    """本次 session 全部載入紀錄。

    刻意做成物件而非模組層的 list:原型版用全域 list,測試之間會互相污染
    (每個測試都得記得先 ``clear()``,忘了就是假的通過),
    而且同一支程式無法同時開兩個獨立的載入情境。
    """

    def __init__(self) -> None:
        self._entries: List[LedgerEntry] = []

    def append(self, entry: LedgerEntry) -> None:
        self._entries.append(entry)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, i):
        return self._entries[i]

    @property
    def entries(self) -> List[LedgerEntry]:
        return list(self._entries)

    def corrected(self) -> List[LedgerEntry]:
        """實際被移動過時間的檔。"""
        return [e for e in self._entries if e.corrected]

    def assumed(self) -> List[LedgerEntry]:
        """被 ``allow_unresolved`` 放行、身分不明的檔。"""
        return [e for e in self._entries if e.assumed]

    def to_frame(self) -> pd.DataFrame:
        """整份 ledger 攤成 DataFrame,方便丟進報告。"""
        if not self._entries:
            return pd.DataFrame(columns=['file', 'path', 'verdict', 'how', 'rows',
                                         'start', 'end'])
        return pd.DataFrame([
            dict(file=e.file, path=e.path, verdict=e.verdict, how=e.how,
                 rows=e.rows, start=e.span[0], end=e.span[1])
            for e in self._entries
        ])

    def summary(self) -> pd.DataFrame:
        """依 ``(verdict, how 的第一段)`` 彙總 —— 一眼看出這次資料是怎麼進來的。"""
        import collections
        counter = collections.Counter(
            (e.verdict, e.how.split('|')[0]) for e in self._entries)
        rows = [dict(verdict=k[0], how=k[1], n=v) for k, v in sorted(
            counter.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))]
        return pd.DataFrame(rows, columns=['verdict', 'how', 'n'])


# ══════════════════════════════════════════════════════════════
#   格式偵測
# ══════════════════════════════════════════════════════════════
def sniff_format(path, ts_col: str = 'timestamp') -> Optional[str]:
    """先看**檔案自己說了什麼** —— 有些格式根本沒有時區歧義,不該要求判定表。

    回傳
        ``'INLINE_TZ'``  時間戳自帶 ``+00:00`` / ``Z`` → 無歧義
        ``'EPOCH'``      數值(毫秒或秒)→ 本質就是 UTC,無歧義
        ``'NAIVE'``      裸的本地時間字串 → **有歧義,必須查判定表**
        ``None``         讀不到 / 找不到時間欄

    為什麼需要這一步
    ----------------
    原型的第一版把「未登錄判定表」與「有時區風險」混為一談,
    於是 305 個語料檔裡有 100 個觸發 fail-closed —— 但其中 84 個
    (自帶時區 35 個 + epoch 49 個)本來就沒有歧義。

    這種假警報比漏警報更難處理:使用者被擋了 84 次之後,
    唯一理性的反應就是去找放行開關,然後那個開關會一直開著。
    **安全機制的可用性是它的安全性的一部分。**
    """
    try:
        head = _raw_read(path, nrows=1)
    except Exception:
        return None
    if not len(head.columns):
        return None
    col = ts_col if ts_col in head.columns else next(
        (c for c in head.columns
         if 'time' in str(c).lower() or str(c).lower() in ('ts', 'date')), None)
    if col is None or not len(head):
        return None

    value = str(head[col].iloc[0])
    if value.replace('.', '', 1).isdigit():
        return 'EPOCH'
    # 自帶時區的三種寫法:結尾 Z、日期後出現 +、或結尾是 -HH:MM 偏移
    if value.endswith('Z') or ('+' in value[10:]) or ('-' in value[11:] and ':' in value[-6:]):
        return 'INLINE_TZ'
    return 'NAIVE'


# ══════════════════════════════════════════════════════════════
#   載入器
# ══════════════════════════════════════════════════════════════
class Loader:
    """時間序列載入器 —— 一個實例綁一份設定與一本 ledger。"""

    def __init__(self, config: LoaderConfig, ledger: Optional[Ledger] = None) -> None:
        self.config = config
        self.ledger = ledger if ledger is not None else Ledger()
        self.by_path, self.by_name = config.load_manifests()

    # ── 判定 ──────────────────────────────────────────────────
    def resolve_tz(self, path, ts_col: str = 'timestamp') -> Tuple[Optional[str], str]:
        """回傳 ``(verdict, how)``。優先序:檔案自述格式 > 完整路徑 > basename。

        basename 命中會標記 ``INHERITED_BY_BASENAME`` —— 留痕是刻意的。
        兩個不同目錄下的同名檔可能來自不同來源、有不同時區;
        用 basename 查到的判定「大概率對」,但稽核時必須看得出它不是精確命中。
        """
        fmt = sniff_format(path, ts_col)
        if fmt == 'INLINE_TZ':
            return 'UTC_EXPLICIT', 'INLINE_TZ_無歧義'
        if fmt == 'EPOCH':
            return 'EPOCH', 'EPOCH_無歧義'

        verdict = self.by_path.get(norm_path(path))
        if verdict is not None:
            return verdict, 'by_path'
        verdict = self.by_name.get(ntpath.basename(str(path)))
        if verdict is not None:
            return verdict, 'INHERITED_BY_BASENAME'
        return None, 'NOT_FOUND'

    # ── 載入 ──────────────────────────────────────────────────
    def read_ts(self, path, ts_col: str = 'timestamp',
                usecols: Optional[Sequence[str]] = None,
                allow_unresolved: bool = False) -> pd.DataFrame:
        """時間序列檔的唯一入口。回傳的 ``ts`` 欄**保證是 tz-aware UTC**。

        參數
        ----
        allow_unresolved
            放行未判定的檔,當作 UTC 處理。**逐次呼叫顯式傳入**,
            不提供全域開關 —— 全域開關一旦打開就不會有人記得關掉,
            而且從輸出上看不出它開著。無論放不放行都會寫進 ledger。

        丟出
        ----
        UnresolvedTimezone
            檔在黑名單、或判定缺失且未顯式放行。
        ValueError
            找不到時間欄、或有無法解析的時間戳(NaT)。
        """
        name = ntpath.basename(str(path))
        if name in self.config.blacklist:
            raise UnresolvedTimezone(
                f'[tzledger]「{name}」在黑名單(已鑑定為內容與檔名不符),拒絕載入。')

        verdict, how = self.resolve_tz(path, ts_col)
        if verdict in BAD_VERDICTS:
            message = (
                f'[tzledger] fail-closed:「{name}」無時區判定'
                f'(verdict={verdict}, how={how})。\n'
                f'  路徑:{path}\n'
                f'  放行會產生「一半修正過、一半沒有」的資料 —— 那比全錯更難發現,\n'
                f'  因為錯誤只出現在接縫處,而接縫在圖上看起來像正常的波動。\n'
                f'  處置:先判定該檔並登錄判定表,'
                f'或在本次呼叫顯式傳 allow_unresolved=True 並承擔後果。')
            if not allow_unresolved:
                raise UnresolvedTimezone(message)
            verdict = 'UTC'                       # 承擔後果:當作 UTC
            how += '|ASSUMED_UTC_UNVERIFIED'      # ★但一定要標記,身分不明比未測更差

        cols = list(usecols) if usecols is not None else None
        if cols is not None and ts_col not in cols:
            cols = [ts_col] + cols

        frame = _raw_read(path, usecols=cols)
        if ts_col not in frame.columns:
            raise ValueError(
                f'[tzledger]「{name}」沒有時間欄「{ts_col}」;'
                f'欄位={list(frame.columns)}')

        raw = frame[ts_col]
        stamps, how = self._parse_stamps(raw, verdict, how, name)

        # ★時區:宣告式,不做算術
        #   tz-naive → tz_localize(來源時區):宣告「這串數字原本是哪裡的時間」
        #   tz-aware → tz_convert('UTC') :換算表示法,值不變
        #   兩者都冪等 —— 重複套用是 no-op。這是它與 `- 8h` 的根本差別。
        target_tz = self.config.tz_of[verdict]
        if stamps.dt.tz is None:
            stamps = self._localize(stamps, target_tz, name)
        stamps = stamps.dt.tz_convert('UTC')

        frame['ts'] = stamps
        if ts_col != 'ts':
            frame = frame.drop(columns=[ts_col])

        self.ledger.append(LedgerEntry(
            file=name, path=str(path), verdict=verdict, how=how, rows=len(frame),
            span=(str(frame['ts'].min()), str(frame['ts'].max())) if len(frame)
            else (None, None)))
        return frame

    # ── 內部 ──────────────────────────────────────────────────
    @staticmethod
    def _localize(stamps: pd.Series, target_tz: str, name: str) -> pd.Series:
        r"""把 tz-naive 的時間戳宣告成 ``target_tz`` 的本地時間。

        這裡是「宣告式 vs 算術」差別真正顯現的地方。在沒有日光節約的時區
        (台北、東京)兩者數值相同,所以那些時區看不出差別;
        **有 DST 的時區則完全不同**:偏移量隨日期改變,固定加減必然錯一小時,
        而且只錯在轉換日之後的那一段 —— 又是一個只在接縫處出現的錯誤。

        DST 轉換還會產生兩種本地時間根本不存在或有歧義的情況:

        * **不存在**:春季跳過的那一小時(美東 2023-03-12 02:00–02:59)
        * **有歧義**:秋季重複的那一小時(美東 2023-11-05 01:00–01:59 出現兩次)

        pandas 對這兩種情況預設丟例外。這裡把它轉成本層的錯誤型別並說明成因 ——
        直接冒出 ``NonExistentTimeError`` 的話,使用者最可能的反應是
        加 ``errors='coerce'`` 把它變成 NaT,那就從 fail-closed 掉回 fail-open 了。
        """
        try:
            return stamps.dt.tz_localize(target_tz)
        except Exception as exc:
            # 例外型別不統一:pandas 2 丟 pytz 的 NonExistentTimeError / AmbiguousTimeError,
            # pandas 3 不再依賴 pytz,改丟一般 ValueError —— 只能連訊息一起認
            kind = type(exc).__name__
            text = f'{kind} {exc}'.lower()
            if 'nonexistent' in text or 'ambiguous' in text:
                raise UnresolvedTimezone(
                    f'[tzledger]「{name}」含在 {target_tz} 不存在或有歧義的本地時間'
                    f'({kind})。\n'
                    f'  成因:日光節約轉換 —— 春季跳過一小時、秋季重複一小時。\n'
                    f'  處置:確認來源匯出時用的是本地時間還是 UTC。\n'
                    f'  ⚠️ 不要用 errors=\'coerce\' 把它變成 NaT —— '
                    f'那是把「時間不明確」改寫成「時間不存在」,問題沒解決,只是不再報錯。'
                ) from exc
            raise

    def _parse_stamps(self, raw: pd.Series, verdict: str, how: str,
                      name: str) -> Tuple[pd.Series, str]:
        """把原始欄位解析成 datetime,並強制 NaT = 0。

        NaT 不容許存在,是因為 pandas 對 NaT 的預設行為是**安靜地略過**:
        ``.min()``、``.resample()``、``.groupby()`` 都會跳過它,
        於是「這個檔有 3% 的時間戳壞掉」會表現成「結果好像有點怪」,
        而不是「程式報錯」。
        """
        if verdict == 'EPOCH':
            numeric = pd.to_numeric(raw, errors='coerce')
            valid = numeric.dropna()
            if not len(valid):
                raise ValueError(f'[tzledger]「{name}」EPOCH 欄全數無法轉成數值。')
            # 依量級推 ms/s —— 2001-09-09 之後的秒級時間戳約 1e9,毫秒級約 1e12
            unit = 'ms' if float(valid.iloc[0]) > 1e11 else 's'
            stamps = pd.to_datetime(numeric, unit=unit, utc=True)
            how += f'|unit={unit}'
        else:
            stamps = pd.to_datetime(raw, format='ISO8601', errors='coerce')
            if stamps.isna().all():
                # 非 ISO8601 → 退一步讓 pandas 自己猜,但要在 ledger 留痕
                stamps = pd.to_datetime(raw, errors='coerce')
                how += '|NON_ISO8601'

        n_nat = int(stamps.isna().sum())
        if n_nat:
            samples = raw[stamps.isna()].head(3).tolist()
            raise ValueError(
                f'[tzledger]「{name}」有 {n_nat} 筆時間戳無法解析(NaT)。\n'
                f'  本層要求 NaT = 0:pandas 對 NaT 的預設是安靜略過,\n'
                f'  放行等於讓「部分資料壞掉」偽裝成「結果有點怪」。\n'
                f'  樣本:{samples}')
        return stamps, how


# ══════════════════════════════════════════════════════════════
#   模組層便利介面
# ══════════════════════════════════════════════════════════════
_DEFAULT: Optional[Loader] = None


def configure(config: LoaderConfig) -> Loader:
    """設定模組層的預設載入器,回傳它。"""
    global _DEFAULT
    _DEFAULT = Loader(config)
    return _DEFAULT


def default_loader() -> Loader:
    """取得預設載入器;未設定過就從環境變數建一個。"""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Loader(LoaderConfig.from_env())
    return _DEFAULT


def read_ts(path, ts_col: str = 'timestamp',
            usecols: Optional[Sequence[str]] = None,
            allow_unresolved: bool = False) -> pd.DataFrame:
    """``Loader.read_ts`` 的模組層捷徑,用預設載入器。"""
    return default_loader().read_ts(path, ts_col=ts_col, usecols=usecols,
                                    allow_unresolved=allow_unresolved)


def ledger() -> Ledger:
    """取得預設載入器的 ledger。"""
    return default_loader().ledger
