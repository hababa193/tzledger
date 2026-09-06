# -*- coding: utf-8 -*-
r"""結論身分 —— 同一份資料,身分隨「怎麼用」而不同。

這一組要釘住的性質有點反直覺:``classify`` 對同一份 ledger 會給出不同答案,
取決於呼叫端聲明有沒有走傳導路徑。這是刻意的,不是 bug ——
時區位移不改變任何一列的內容,只改變「哪些列屬於同一天」,
所以沒有用到日界線的分析真的不受影響。
"""
from __future__ import annotations

import io

import pytest

from tzledger import Loader, LoaderConfig
from tzledger.provenance import (
    CLEAN_NO_CORRECTED_INPUT,
    CLEAN_PATH_NOT_AFFECTED,
    UNKNOWN_NO_LEDGER,
    UNTESTED,
    UNVERIFIED_INPUT,
    classify,
    report,
)


# ══════════════════════════════════════════════════════════════
#   四種身分
# ══════════════════════════════════════════════════════════════
def test_corrected_input_on_derived_path_is_untested(loader, corpus):
    """讀到修正過的檔 + 走了日界線 → 原結論不具檢定力。"""
    loader.read_ts(corpus / 'AAA_1min_2023.csv', usecols=['close'])
    prov = classify(loader.ledger, declare_derived_path=True)
    assert prov.tag == UNTESTED
    assert prov.n_corrected == 1
    assert not prov.citable


def test_same_data_off_derived_path_is_clean(loader, corpus):
    """★同一份資料,聲明沒走傳導路徑 → 結論可以留。"""
    loader.read_ts(corpus / 'AAA_1min_2023.csv', usecols=['close'])
    prov = classify(loader.ledger, declare_derived_path=False)
    assert prov.tag == CLEAN_PATH_NOT_AFFECTED
    assert prov.citable


def test_identity_depends_on_usage_not_only_on_data(loader, corpus):
    """★把上面兩條的對比明確寫成一個測試 —— 這是整個模組的論點。"""
    loader.read_ts(corpus / 'AAA_1min_2023.csv', usecols=['close'])
    on_path = classify(loader.ledger, declare_derived_path=True)
    off_path = classify(loader.ledger, declare_derived_path=False)
    assert on_path.tag != off_path.tag


def test_clean_input_is_clean_even_on_derived_path(loader, corpus):
    """本來就是 UTC 的檔,就算走日界線也沒事 —— 沒有東西被移動過。"""
    loader.read_ts(corpus / 'BBB_1min_2023.csv', usecols=['close'])
    loader.read_ts(corpus / 'DDD_1min_2023.csv', usecols=['close'])
    prov = classify(loader.ledger, declare_derived_path=True)
    assert prov.tag == CLEAN_NO_CORRECTED_INPUT
    assert prov.citable


def test_assumed_input_is_worse_than_untested(loader, corpus):
    """★放行過未判定檔 → 身分不明,連「沒走傳導路徑」都救不了。

    未測是「不知道結果對不對」;身分不明是「不知道輸入是什麼」。
    後者更前面,所以更嚴重。
    """
    loader.read_ts(corpus / 'ZZZ_1min_2023.csv', usecols=['close'],
                   allow_unresolved=True)
    prov = classify(loader.ledger, declare_derived_path=False)
    assert prov.tag == UNVERIFIED_INPUT
    assert not prov.citable


def test_assumed_beats_clean_when_mixed(loader, corpus):
    """混合輸入時取最差的身分 —— 一個不明的輸入就足以污染整個結論。"""
    loader.read_ts(corpus / 'BBB_1min_2023.csv', usecols=['close'])
    loader.read_ts(corpus / 'ZZZ_1min_2023.csv', usecols=['close'],
                   allow_unresolved=True)
    prov = classify(loader.ledger, declare_derived_path=False)
    assert prov.tag == UNVERIFIED_INPUT


# ══════════════════════════════════════════════════════════════
#   ★空 ledger 的 fail-open
# ══════════════════════════════════════════════════════════════
def test_empty_ledger_is_unknown_not_clean(config):
    r"""★這一條是自己踩到之後補的,而且與整個模組要防的缺陷完全同型。

    原版在 ledger 為空時回傳 ``CLEAN_…``。但空的 ledger 代表
    **本次沒有任何檔案經過這一層** —— 資料是被 ``pd.read_parquet`` 或
    裸 ``read_csv`` 讀走的。那是「不知道」,不是「乾淨」。

    一個安靜的、看起來像好消息的預設值。
    """
    loader = Loader(config)
    prov = classify(loader.ledger, declare_derived_path=True)
    assert prov.tag == UNKNOWN_NO_LEDGER
    assert not prov.citable


def test_empty_ledger_is_unknown_regardless_of_declaration(config):
    """不管怎麼聲明都是 UNKNOWN —— 因為問題出在更前面。"""
    loader = Loader(config)
    for declared in (True, False):
        assert classify(loader.ledger, declare_derived_path=declared).tag \
            == UNKNOWN_NO_LEDGER


# ══════════════════════════════════════════════════════════════
#   報告輸出
# ══════════════════════════════════════════════════════════════
def test_report_names_the_offending_files(loader, corpus):
    loader.read_ts(corpus / 'AAA_1min_2023.csv', usecols=['close'])
    buf = io.StringIO()
    report(classify(loader.ledger, declare_derived_path=True), out=buf)
    text = buf.getvalue()
    assert UNTESTED in text
    assert 'AAA_1min_2023.csv' in text


def test_report_explains_what_to_do(loader, corpus):
    """報告要說「所以現在該怎麼辦」,不能只丟一個標籤。"""
    loader.read_ts(corpus / 'ZZZ_1min_2023.csv', usecols=['close'],
                   allow_unresolved=True)
    buf = io.StringIO()
    report(classify(loader.ledger, declare_derived_path=False), out=buf)
    assert '結論不得引用' in buf.getvalue()


def test_report_warns_about_repeated_looks(loader, corpus):
    """★重跑是追加一次檢定,不是回復原狀 —— 這句話必須出現在報告裡。

    資料修好了,多重比較的負擔不會跟著消失。
    """
    loader.read_ts(corpus / 'AAA_1min_2023.csv', usecols=['close'])
    buf = io.StringIO()
    report(classify(loader.ledger, declare_derived_path=True), out=buf)
    assert '追加一次檢定' in buf.getvalue()
