# -*- coding: utf-8 -*-
r"""跨序列覆蓋率 —— 抓「每個檔都對,但少載了一半」。

這一組的兩個案例是對立的,而且第一版的判準只能過其中一個:

* **漏檔**:歷史一樣長,但列數只有別人的四分之一 → 必須報警
* **上市較晚**:列數只有別人的一半,但那是**合法的** → 不能報警

用「列數比值」判離群會把第二個誤報成第一個。
用「密度 = 列數 ÷ 年數」兩個都判對。
"""
from __future__ import annotations

import pandas as pd
import pytest

from tzledger.audit import assert_coverage, coverage, format_coverage


def _series(start: str, years: float, per_year: int) -> pd.DataFrame:
    """造一段時間軸:從 ``start`` 起、長 ``years`` 年、每年 ``per_year`` 列。"""
    n = max(int(years * per_year), 2)
    span = pd.Timedelta(days=365.25 * years)
    stamps = pd.date_range(pd.Timestamp(start, tz='UTC'), periods=n,
                           freq=span / (n - 1))
    return pd.DataFrame({'ts': stamps, 'close': range(n)})


# ══════════════════════════════════════════════════════════════
#   案例一:漏檔 —— 必須抓到
# ══════════════════════════════════════════════════════════════
def test_missing_files_are_flagged():
    r"""★立法案例。

    一組腳本用精確檔名樣式 ``{symbol}_1min_{year}.csv`` 逐年讀取,
    但其中一個標的在磁碟上是**半年一檔**,於是只命中 6 年裡的 2 年。
    聚合成小時 bar 之後只有 12,341 根,其餘標的 39,000–45,000 根。

    每一項檢查都通過了。缺的不是檢查,是**比較的對象**。
    """
    frames = {
        'AAA': _series('2021-01-01', 5.4, 8760),      # 正常
        'BBB': _series('2021-01-01', 5.4, 8760),      # 正常
        'CCC': _series('2021-01-01', 5.4, 2281),      # ★只讀到 2/6 年
    }
    cov = coverage(frames)
    flagged = set(cov.loc[cov['outlier'], 'series'])
    assert flagged == {'CCC'}


def test_missing_files_raise_when_asserted():
    frames = {
        'AAA': _series('2021-01-01', 5.4, 8760),
        'CCC': _series('2021-01-01', 5.4, 2281),
    }
    with pytest.raises(ValueError, match='覆蓋率離群'):
        assert_coverage(coverage(frames))


# ══════════════════════════════════════════════════════════════
#   案例二:上市較晚 —— 不能誤報
# ══════════════════════════════════════════════════════════════
def test_short_history_is_not_flagged():
    r"""★第一版判準在這裡誤報。

    這個序列的**總列數**只有別人的一半多一點(比值 0.579),
    因為它的歷史只有 2.93 年。但它的**密度**是 1.002 —— 資料是完整的。

    「資料比較少」與「資料漏了」是兩件事。用總量判,兩者無法區分。
    """
    frames = {
        'AAA': _series('2021-01-01', 5.4, 525600),
        'BBB': _series('2021-01-01', 5.4, 525600),
        'SHORT': _series('2023-05-01', 2.93, 525600),   # ★歷史短但完整
    }
    cov = coverage(frames)
    assert not cov['outlier'].any()

    short = cov.loc[cov['series'] == 'SHORT'].iloc[0]
    assert short['rows'] < cov['rows'].max() * 0.7      # 總量確實少很多
    assert short['density'] == pytest.approx(1.0, abs=0.05)   # 但密度是滿的


def test_row_count_ratio_would_have_false_positived():
    """★把第一版的判準寫成測試,證明它為什麼不行。

    留著這個測試是因為「為什麼不用更簡單的作法」這個問題會反覆出現,
    而口頭解釋不如一段跑得起來的反例。
    """
    frames = {
        'AAA': _series('2021-01-01', 5.4, 525600),
        'BBB': _series('2021-01-01', 5.4, 525600),
        'SHORT': _series('2023-05-01', 2.93, 525600),
    }
    cov = coverage(frames)
    median_rows = cov['rows'].median()
    row_ratio = cov.loc[cov['series'] == 'SHORT', 'rows'].iloc[0] / median_rows

    assert row_ratio < 0.8          # 舊判準會報警
    assert not cov.loc[cov['series'] == 'SHORT', 'outlier'].iloc[0]   # 新判準不會


# ══════════════════════════════════════════════════════════════
#   邊界
# ══════════════════════════════════════════════════════════════
def test_empty_series_is_flagged():
    """★空的序列算離群 —— 「完全沒有資料」不該安靜通過。

    這是 fail-open 最常見的入口:``if not len(df): return``
    看起來像在處理邊界,實際上是把最嚴重的情況變成沒有情況。
    """
    frames = {
        'AAA': _series('2021-01-01', 5.4, 8760),
        'EMPTY': pd.DataFrame({'ts': pd.to_datetime([], utc=True), 'close': []}),
    }
    cov = coverage(frames)
    assert bool(cov.loc[cov['series'] == 'EMPTY', 'outlier'].iloc[0])


def test_none_series_is_flagged():
    frames = {'AAA': _series('2021-01-01', 5.4, 8760), 'NONE': None}
    cov = coverage(frames)
    assert bool(cov.loc[cov['series'] == 'NONE', 'outlier'].iloc[0])


def test_short_span_series_are_not_all_flagged():
    r"""★迴歸測試 —— 這個 bug 是跑 examples/demo.py 時掉出來的。

    原本為了防除以零,在算密度前卡了一個「跨度要大於 0.1 年」的門檻。
    結果是**跨度不足五週的序列全部算不出密度,於是全部被標成離群** ——
    一個為了安全而設的保險,自己變成了假警報的來源。

    密度是相對中位數的比值,跨度大小不影響比較的有效性。
    只有跨度為零(單列)才真的無法計算。
    """
    frames = {name: _series('2023-03-01', 1 / 365.25, 525600)   # 各一天
              for name in ('AAA', 'BBB', 'CCC')}
    cov = coverage(frames)
    assert not cov['outlier'].any(), '同樣長度的短序列不該被標成離群'
    assert cov['density'].notna().all()
    assert_coverage(cov)                       # 不該丟


def test_short_span_still_detects_real_gaps():
    """短跨度下仍然抓得到真的漏檔 —— 修 bug 不能把功能一起修掉。"""
    frames = {
        'AAA': _series('2023-03-01', 1 / 365.25, 525600),
        'BBB': _series('2023-03-01', 1 / 365.25, 525600),
        'GAPPY': _series('2023-03-01', 1 / 365.25, 100000),   # 密度只有 1/5
    }
    cov = coverage(frames)
    assert set(cov.loc[cov['outlier'], 'series']) == {'GAPPY'}


def test_single_row_series_is_flagged():
    """跨度為零 → 密度無從計算 → 標記離群,不是安靜通過。"""
    single = pd.DataFrame({'ts': pd.to_datetime(['2023-03-01'], utc=True), 'close': [1]})
    frames = {'AAA': _series('2021-01-01', 5.4, 8760), 'ONE': single}
    cov = coverage(frames)
    assert bool(cov.loc[cov['series'] == 'ONE', 'outlier'].iloc[0])


def test_assert_coverage_passes_when_all_uniform():
    frames = {n: _series('2021-01-01', 5.4, 8760) for n in ('AAA', 'BBB', 'CCC')}
    assert_coverage(coverage(frames))          # 不該丟


def test_format_coverage_marks_outliers():
    frames = {
        'AAA': _series('2021-01-01', 5.4, 8760),
        'CCC': _series('2021-01-01', 5.4, 2281),
    }
    text = format_coverage(coverage(frames))
    assert '離群' in text
    assert 'CCC' in text
