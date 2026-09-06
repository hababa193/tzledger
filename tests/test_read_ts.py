# -*- coding: utf-8 -*-
"""核心載入行為 —— 五種輸入格式都要落在同一個 UTC 時間軸上。"""
from __future__ import annotations

import pandas as pd
import pytest

from tzledger import sniff_format


# ══════════════════════════════════════════════════════════════
#   格式偵測
# ══════════════════════════════════════════════════════════════
@pytest.mark.parametrize('filename, expected', [
    ('AAA_1min_2023.csv', 'NAIVE'),        # 裸字串 → 有歧義
    ('CCC_1min_2023.csv', 'INLINE_TZ'),    # 自帶 +00:00
    ('DDD_1min_2023.csv', 'EPOCH'),        # 毫秒
    ('EEE_1min_2023.csv', 'EPOCH'),        # 秒
])
def test_sniff_format(corpus, filename, expected):
    assert sniff_format(corpus / filename) == expected


def test_sniff_returns_none_for_missing_file(tmp_path):
    """讀不到就回 None,不是丟例外 —— 呼叫端要能區分「讀不到」與「格式不明」。"""
    assert sniff_format(tmp_path / 'does_not_exist.csv') is None


# ══════════════════════════════════════════════════════════════
#   載入
# ══════════════════════════════════════════════════════════════
def test_output_is_always_tz_aware_utc(loader, corpus):
    """五種格式,一種輸出。這是這一層存在的全部理由。"""
    for name in ['AAA_1min_2023.csv', 'BBB_1min_2023.csv', 'CCC_1min_2023.csv',
                 'DDD_1min_2023.csv', 'EEE_1min_2023.csv']:
        df = loader.read_ts(corpus / name, usecols=['close'])
        assert df['ts'].dt.tz is not None, f'{name} 輸出是 tz-naive'
        assert str(df['ts'].dt.tz) == 'UTC', f'{name} 不是 UTC'


def test_tpe_file_is_shifted_to_utc(loader, corpus, expected_utc):
    """台北時間的檔要被移到正確的 UTC 位置。"""
    df = loader.read_ts(corpus / 'AAA_1min_2023.csv', usecols=['close'])
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(df['ts']), expected_utc, check_names=False)


def test_utc_file_is_not_shifted(loader, corpus):
    """判定為 UTC 的檔不該被動到 —— 修正只在需要時發生。"""
    df = loader.read_ts(corpus / 'BBB_1min_2023.csv', usecols=['close'])
    assert str(df['ts'].iloc[0]) == '2023-03-01 20:00:00+00:00'


def test_epoch_unit_inferred(loader, corpus):
    """毫秒與秒兩種 epoch 要落在同一個時間點上。"""
    ms = loader.read_ts(corpus / 'DDD_1min_2023.csv', usecols=['close'])
    sec = loader.read_ts(corpus / 'EEE_1min_2023.csv', usecols=['close'])
    pd.testing.assert_series_equal(ms['ts'], sec['ts'], check_names=False)
    assert ms['ts'].iloc[0] == pd.Timestamp('2023-03-01 20:00:00', tz='UTC')


def test_all_unambiguous_formats_agree(loader, corpus):
    """★自帶時區、epoch 毫秒、epoch 秒 —— 三種寫法描述同一件事,就該得到同一個答案。"""
    frames = [loader.read_ts(corpus / n, usecols=['close'])
              for n in ('CCC_1min_2023.csv', 'DDD_1min_2023.csv', 'EEE_1min_2023.csv')]
    for other in frames[1:]:
        pd.testing.assert_series_equal(frames[0]['ts'], other['ts'], check_names=False)


def test_ts_column_replaces_source_column(loader, corpus):
    """輸出一律叫 ``ts``,原欄位移除 —— 下游不必知道來源欄叫什麼。"""
    df = loader.read_ts(corpus / 'BBB_1min_2023.csv', usecols=['close'])
    assert 'ts' in df.columns
    assert 'timestamp' not in df.columns


def test_usecols_always_includes_timestamp(loader, corpus):
    """呼叫端沒把時間欄列進 usecols 也要能work —— 忘記列是常態,不該變成錯誤。"""
    df = loader.read_ts(corpus / 'BBB_1min_2023.csv', usecols=['close'])
    assert list(df.columns) == ['close', 'ts']


def test_missing_time_column_raises(loader, corpus):
    with pytest.raises(ValueError, match='沒有時間欄'):
        loader.read_ts(corpus / 'BBB_1min_2023.csv', ts_col='no_such_column')


# ══════════════════════════════════════════════════════════════
#   ledger
# ══════════════════════════════════════════════════════════════
def test_ledger_records_every_load(loader, corpus):
    loader.read_ts(corpus / 'AAA_1min_2023.csv', usecols=['close'])
    loader.read_ts(corpus / 'BBB_1min_2023.csv', usecols=['close'])
    assert len(loader.ledger) == 2
    assert [e.verdict for e in loader.ledger] == ['TPE', 'UTC']


def test_ledger_distinguishes_corrected_from_clean(loader, corpus):
    """★「被移動過」與「本來就對」必須分得開 —— 身分判定整個建立在這上面。"""
    loader.read_ts(corpus / 'AAA_1min_2023.csv', usecols=['close'])   # TPE,要移
    loader.read_ts(corpus / 'BBB_1min_2023.csv', usecols=['close'])   # UTC,不動
    loader.read_ts(corpus / 'DDD_1min_2023.csv', usecols=['close'])   # epoch,不動
    assert [e.file for e in loader.ledger.corrected()] == ['AAA_1min_2023.csv']


def test_ledger_how_records_lookup_route(loader, corpus):
    """``how`` 記的是「怎麼查到的」,不是「查到什麼」。"""
    loader.read_ts(corpus / 'AAA_1min_2023.csv', usecols=['close'])   # 查判定表
    loader.read_ts(corpus / 'CCC_1min_2023.csv', usecols=['close'])   # 檔案自述
    routes = [e.how for e in loader.ledger]
    assert routes[0] == 'by_path'
    assert routes[1] == 'INLINE_TZ_無歧義'


def test_ledger_summary_is_a_frame(loader, corpus):
    loader.read_ts(corpus / 'AAA_1min_2023.csv', usecols=['close'])
    summary = loader.ledger.summary()
    assert list(summary.columns) == ['verdict', 'how', 'n']
    assert int(summary['n'].sum()) == 1


def test_ledgers_are_independent_between_loaders(config, corpus):
    """★兩個載入器不共用 ledger —— 原型版用全域 list,測試之間會互相污染。"""
    from tzledger import Loader
    a, b = Loader(config), Loader(config)
    a.read_ts(corpus / 'AAA_1min_2023.csv', usecols=['close'])
    assert len(a.ledger) == 1
    assert len(b.ledger) == 0
