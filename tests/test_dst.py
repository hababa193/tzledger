# -*- coding: utf-8 -*-
r"""日光節約 —— 「宣告式而非算術」這個主張的實際證據。

為什麼需要這一組
----------------
這個 repo 主張時區轉換要用 ``tz_localize`` + ``tz_convert``,不要用 ``- 8h``。
但在**台北**,這兩種寫法的數值完全相同 —— 台灣自 1980 年起沒有日光節約。

也就是說:只用台北的資料,寫不出任何一個能區分兩種作法的測試。
把「宣告式比較好」當成不必證明的常識,正是這一層想防的那種習慣。

所以這裡改用美東。2023-03-12 當地 02:00 EST 直接跳到 03:00 EDT,
偏移量從 UTC-5 變成 UTC-4。固定加減必然在轉換點之後錯一小時,
而且**只錯後半段** —— 又是一個只出現在接縫處的錯誤。
"""
from __future__ import annotations

import pandas as pd
import pytest

from tzledger import UnresolvedTimezone


def test_declarative_conversion_handles_dst(dst_loader, dst_corpus, dst_expected_utc):
    """★宣告式轉換橫跨 DST 轉換點仍然得到連續的時間軸。"""
    df = dst_loader.read_ts(dst_corpus / 'NYSE_1min_2023.csv', usecols=['close'])
    # freq='infer':讀進來的索引沒有 freq,預期值有 <Minute>;pandas 4 起會比對 freq。
    # 不用 check_freq=False —— pandas 2 的 assert_index_equal 沒有這個參數
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(df['ts'], freq='infer'), dst_expected_utc, check_names=False)

    gaps = pd.Series(df['ts']).diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(minutes=1)], '時間軸出現缺口'


def test_fixed_offset_would_be_wrong(dst_corpus, dst_expected_utc):
    """★把算術作法寫出來,證明它在同一份資料上會錯。

    這不是測本專案的程式,是把「為什麼不用更簡單的作法」做成可執行的反例。
    口頭解釋擋不住下一個人把它改回去,一個會紅的測試可以。
    """
    raw = pd.read_csv(dst_corpus / 'NYSE_1min_2023.csv')
    naive = pd.to_datetime(raw['timestamp'])

    # 「美東 = UTC-5,所以加 5 小時就好」—— 在轉換點之後就錯了
    arithmetic = (naive + pd.Timedelta(hours=5)).dt.tz_localize('UTC')

    assert not arithmetic.equals(pd.Series(dst_expected_utc))

    diff = (arithmetic - pd.Series(dst_expected_utc))
    assert diff.iloc[0] == pd.Timedelta(0)                    # 轉換前:對的
    assert diff.iloc[-1] == pd.Timedelta(hours=1)             # 轉換後:整整差一小時
    assert set(diff.unique()) == {pd.Timedelta(0), pd.Timedelta(hours=1)}


def test_arithmetic_creates_a_seam_not_a_uniform_shift(dst_corpus):
    """★這個錯誤最惡劣的性質:它不是整段偏移,是**中間裂開**。

    整段偏移還有機會在對照另一個來源時被發現;
    只有一段偏移的資料,在圖上看起來就是一次跳空 —— 而金融資料本來就有跳空。
    """
    raw = pd.read_csv(dst_corpus / 'NYSE_1min_2023.csv')
    naive = pd.to_datetime(raw['timestamp'])
    arithmetic = naive + pd.Timedelta(hours=5)

    gaps = arithmetic.diff().dropna()
    assert (gaps == pd.Timedelta(hours=1, minutes=1)).sum() == 1   # 一道接縫
    assert (gaps == pd.Timedelta(minutes=1)).sum() == len(gaps) - 1


# ══════════════════════════════════════════════════════════════
#   不存在 / 有歧義的本地時間
# ══════════════════════════════════════════════════════════════
def test_nonexistent_local_time_is_rejected(dst_loader, dst_corpus):
    """★轉換日不存在的本地時間 → 拒絕載入,不是靜靜地變成 NaT。"""
    with pytest.raises(UnresolvedTimezone) as excinfo:
        dst_loader.read_ts(dst_corpus / 'NONEXIST_1min_2023.csv', usecols=['close'])
    assert '日光節約' in str(excinfo.value)


def test_nonexistent_error_warns_against_coerce(dst_loader, dst_corpus):
    """錯誤訊息要主動擋掉最可能的錯誤修法。

    使用者看到 ``NonExistentTimeError`` 的第一反應通常是加 ``errors='coerce'``,
    那會把「時間不明確」改寫成「時間不存在」—— 問題沒解決,只是不再報錯,
    而且下一關的 NaT 檢查會把它擋在更遠的地方,更難追。
    """
    with pytest.raises(UnresolvedTimezone) as excinfo:
        dst_loader.read_ts(dst_corpus / 'NONEXIST_1min_2023.csv', usecols=['close'])
    assert 'coerce' in str(excinfo.value)


def test_dst_file_still_lands_in_utc(dst_loader, dst_corpus):
    """不管來源時區多複雜,輸出契約不變:tz-aware UTC。"""
    df = dst_loader.read_ts(dst_corpus / 'NYSE_1min_2023.csv', usecols=['close'])
    assert str(df['ts'].dt.tz) == 'UTC'
    assert dst_loader.ledger[0].verdict == 'NYC'
    assert dst_loader.ledger[0].corrected is True
