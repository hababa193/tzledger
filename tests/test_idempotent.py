# -*- coding: utf-8 -*-
r"""冪等性 —— 這個 repo 的核心論點。

主張
----
時區修正必須是**冪等**的:修過的資料再修一次,結果不變。

因為只要不冪等,「這份資料修過了嗎」就變成必須靠人記得的事。
在多支腳本、多次重跑、多個工具疊加的環境裡,靠人記得的事一定會出錯,
而且出錯的方向恰好看不出來 —— 值變了、型別沒變、沒有例外、沒有紅字。

``tz_localize`` + ``tz_convert`` 冪等;``- Timedelta(hours=8)`` 不冪等。
下面兩組測試把這個差別做成可執行的證據。
"""
from __future__ import annotations

import pandas as pd
import pytest

from tzledger.compat import install, is_installed, uninstall


# ══════════════════════════════════════════════════════════════
#   一、兩種寫法的本質差別
# ══════════════════════════════════════════════════════════════
def test_declarative_conversion_is_idempotent():
    """★宣告式:``tz_localize`` 之後再 ``tz_convert('UTC')`` 幾次都一樣。"""
    naive = pd.Series(pd.date_range('2023-03-01 20:00', periods=10, freq='1min'))

    once = naive.dt.tz_localize('Asia/Taipei').dt.tz_convert('UTC')
    twice = once.dt.tz_convert('UTC')
    thrice = twice.dt.tz_convert('UTC')

    pd.testing.assert_series_equal(once, twice)
    pd.testing.assert_series_equal(once, thrice)


def test_arithmetic_shift_accumulates():
    """★算術式:每套用一次就再位移 8 小時,而且不會有任何錯誤訊息。

    這正是前一版工具的失效模式。它做 ``t - Timedelta(hours=8)`` 且輸出仍是
    tz-naive,所以資料本身**帶不出「我修過了」這個資訊** ——
    第二次套用看起來跟第一次一模一樣地合理。
    """
    naive = pd.Series(pd.date_range('2023-03-01 20:00', periods=10, freq='1min'))

    once = naive - pd.Timedelta(hours=8)
    twice = once - pd.Timedelta(hours=8)

    assert (twice - once).iloc[0] == pd.Timedelta(hours=-8)
    assert (twice - naive).iloc[0] == pd.Timedelta(hours=-16)   # ★雙重位移
    assert twice.dt.tz is None                                  # ★而且看不出來


def test_arithmetic_result_is_indistinguishable_from_correct():
    """★被污染的資料與正確資料在型別、長度、欄位上完全相同。

    這是整件事最麻煩的地方:沒有任何自動化檢查能從**一份**資料看出它錯了。
    只能靠「與另一個來源比對」或「一開始就不讓它發生」。
    """
    naive = pd.Series(pd.date_range('2023-03-01 20:00', periods=10, freq='1min'))
    correct = naive.dt.tz_localize('Asia/Taipei').dt.tz_convert('UTC').dt.tz_localize(None)
    corrupted = correct - pd.Timedelta(hours=8)

    assert correct.dtype == corrupted.dtype
    assert len(correct) == len(corrupted)
    assert correct.is_monotonic_increasing == corrupted.is_monotonic_increasing
    assert not correct.equals(corrupted)          # 只有逐值比對才看得出來


# ══════════════════════════════════════════════════════════════
#   二、相容層的冪等
# ══════════════════════════════════════════════════════════════
def test_repeated_reads_give_identical_result(loader, corpus):
    """★同一個檔透過相容層讀兩次,結果必須相同。"""
    install(loader)
    first = pd.read_csv(corpus / 'AAA_1min_2023.csv')
    second = pd.read_csv(corpus / 'AAA_1min_2023.csv')
    pd.testing.assert_frame_equal(first, second)


def test_already_aware_input_is_not_shifted(loader, corpus):
    """★已經是 tz-aware 的資料再經過一次,只換算表示法,不位移。"""
    install(loader)
    df = pd.read_csv(corpus / 'CCC_1min_2023.csv')       # 自帶 +00:00
    assert df['timestamp'].dt.tz is not None
    assert df['timestamp'].iloc[0] == pd.Timestamp('2023-03-01 20:00:00', tz='UTC')
    assert loader.ledger[0].verdict == 'ALREADY_AWARE'
    assert loader.ledger[0].how == 'idempotent_noop'


def test_correction_matches_direct_read(loader, corpus, expected_utc):
    """相容層與直接呼叫 ``read_ts`` 必須給出同一個時間軸。

    兩條路徑各自實作了一次轉換邏輯,這是真實的重複 ——
    所以需要一個測試把它們釘在一起,否則哪天只改了一邊不會有人發現。
    """
    direct = loader.read_ts(corpus / 'AAA_1min_2023.csv', usecols=['close'])
    install(loader)
    patched = pd.read_csv(corpus / 'AAA_1min_2023.csv')

    pd.testing.assert_index_equal(
        pd.DatetimeIndex(direct['ts']), expected_utc, check_names=False)
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(patched['timestamp']), expected_utc, check_names=False)


# ══════════════════════════════════════════════════════════════
#   三、安裝／解除安裝
# ══════════════════════════════════════════════════════════════
def test_double_install_does_not_break_uninstall(loader, corpus):
    """★重複 install 不能讓「原版 read_csv」的備份指向已 patch 的函式。

    否則 uninstall 會「還原」成 patch 版本,而且從此再也還原不回去 ——
    一個安靜的、只有在測試互相干擾時才會現形的錯誤。
    """
    original = pd.read_csv
    install(loader)
    install(loader)                      # 第二次應該是 no-op
    uninstall()
    assert pd.read_csv is original


def test_uninstall_restores_plain_behaviour(loader, corpus):
    install(loader)
    assert is_installed()
    uninstall()
    assert not is_installed()
    raw = pd.read_csv(corpus / 'AAA_1min_2023.csv')
    assert raw['timestamp'].dtype == object      # 還原後就是普通的字串欄


def test_non_corpus_files_pass_through(loader, corpus):
    """★相容層只碰語料檔。讀設定檔、讀報表的 ``read_csv`` 不該被動到。

    monkeypatch 的代價是「這行做了什麼取決於別處有沒有 install」,
    範圍越小,這個代價越可控。
    """
    other = corpus / 'some_report.csv'
    pd.DataFrame({'timestamp': ['2023-03-01 20:00:00'], 'value': [1]}).to_csv(
        other, index=False)

    install(loader)
    df = pd.read_csv(other)
    assert df['timestamp'].dtype == object       # 原樣通過,沒被解析
    assert len(loader.ledger) == 0               # 也沒進 ledger


def test_naive_mode_fixes_values_but_keeps_interface(loader, corpus, expected_utc):
    """``naive=True``:值被修正,但 dtype 仍是 tz-naive。

    存在的理由是舊腳本裡大量的 ``df.timestamp > '2023-01-01'``,
    對 tz-aware 欄位會直接 TypeError。如果遷移必須同時改邏輯,就不會有人遷移。
    """
    install(loader, naive=True)
    df = pd.read_csv(corpus / 'AAA_1min_2023.csv')

    assert df['timestamp'].dt.tz is None                       # 介面不變
    assert (df['timestamp'] > '2023-03-01').all()              # 舊寫法還能用
    pd.testing.assert_index_equal(                             # 值已修正
        pd.DatetimeIndex(df['timestamp']),
        pd.DatetimeIndex(expected_utc.tz_localize(None)),
        check_names=False)


def test_unresolved_still_blocked_through_compat(loader, corpus):
    """相容層不是後門 —— 未判定的檔透過它讀一樣會被擋。"""
    from tzledger import UnresolvedTimezone
    install(loader)
    with pytest.raises(UnresolvedTimezone):
        pd.read_csv(corpus / 'ZZZ_1min_2023.csv')
