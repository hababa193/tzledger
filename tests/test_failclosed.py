# -*- coding: utf-8 -*-
"""fail-closed 行為 —— 不確定的資料不准通過。

這一組測的不是「功能會不會壞」,是「安全機制會不會被繞過」。
每一條都對應一種實際發生過、或差一點就發生的放行方式。
"""
from __future__ import annotations

import pytest

from tzledger import UnresolvedTimezone


# ══════════════════════════════════════════════════════════════
#   一、判定缺失 → 拒絕
# ══════════════════════════════════════════════════════════════
def test_unresolved_file_is_rejected(loader, corpus):
    """裸字串 + 不在判定表 = 時區不明 → 拒絕載入,不是猜一個。"""
    with pytest.raises(UnresolvedTimezone, match='fail-closed'):
        loader.read_ts(corpus / 'ZZZ_1min_2023.csv', usecols=['close'])


def test_rejection_message_names_the_file_and_the_fix(loader, corpus):
    """錯誤訊息要能讓人**當場知道怎麼辦**,否則使用者只會去找開關。"""
    with pytest.raises(UnresolvedTimezone) as excinfo:
        loader.read_ts(corpus / 'ZZZ_1min_2023.csv', usecols=['close'])
    message = str(excinfo.value)
    assert 'ZZZ_1min_2023.csv' in message
    assert 'allow_unresolved' in message


def test_rejected_load_is_not_in_ledger(loader, corpus):
    """被擋下來的檔沒有進 ledger —— 因為它根本沒被載入。"""
    with pytest.raises(UnresolvedTimezone):
        loader.read_ts(corpus / 'ZZZ_1min_2023.csv', usecols=['close'])
    assert len(loader.ledger) == 0


# ══════════════════════════════════════════════════════════════
#   二、放行 → 一定留痕
# ══════════════════════════════════════════════════════════════
def test_allow_unresolved_lets_it_through(loader, corpus):
    df = loader.read_ts(corpus / 'ZZZ_1min_2023.csv', usecols=['close'],
                        allow_unresolved=True)
    assert str(df['ts'].dt.tz) == 'UTC'


def test_allow_unresolved_is_marked_in_ledger(loader, corpus):
    """★放行可以,但**不能安靜地放行**。這個標記是後面身分判定的唯一依據。"""
    loader.read_ts(corpus / 'ZZZ_1min_2023.csv', usecols=['close'],
                   allow_unresolved=True)
    entry = loader.ledger[0]
    assert entry.assumed is True
    assert 'ASSUMED_UTC_UNVERIFIED' in entry.how


def test_allow_unresolved_is_per_call_not_global(loader, corpus):
    """★放行是逐次的 —— 放行過一次不會讓下一次也通過。

    全域開關的問題不是它不安全,是它**不會被關掉**:
    開的人解決了當下的問題就走了,而從輸出上看不出它還開著。
    """
    loader.read_ts(corpus / 'ZZZ_1min_2023.csv', usecols=['close'],
                   allow_unresolved=True)
    with pytest.raises(UnresolvedTimezone):
        loader.read_ts(corpus / 'ZZZ_1min_2023.csv', usecols=['close'])


# ══════════════════════════════════════════════════════════════
#   三、黑名單 → 永不載入
# ══════════════════════════════════════════════════════════════
def test_blacklisted_file_is_rejected(loader, corpus):
    """已鑑定為「內容與檔名不符」的檔,連判定表說它是 UTC 都不能救它。"""
    with pytest.raises(UnresolvedTimezone, match='黑名單'):
        loader.read_ts(corpus / 'WRONG_1min_2023.csv', usecols=['close'])


def test_blacklist_beats_allow_unresolved(loader, corpus):
    """★黑名單不受放行開關影響 —— 兩者防的是不同的東西。

    ``allow_unresolved`` 說的是「我承擔時區不明的風險」;
    黑名單說的是「這個檔的內容不是它宣稱的東西」。
    後者不是風險承擔的問題,是資料根本不對。
    """
    with pytest.raises(UnresolvedTimezone, match='黑名單'):
        loader.read_ts(corpus / 'WRONG_1min_2023.csv', usecols=['close'],
                       allow_unresolved=True)


# ══════════════════════════════════════════════════════════════
#   四、NaT → 拒絕
# ══════════════════════════════════════════════════════════════
def test_unparseable_timestamps_are_rejected(loader, corpus):
    """★NaT = 0 是硬要求。

    pandas 對 NaT 的預設行為是安靜略過:``.min()``、``.resample()``、
    ``.groupby()`` 都會跳過它。所以「這個檔有 3% 的時間戳壞掉」
    會表現成「結果好像有點怪」,而不是「程式報錯」。
    """
    with pytest.raises(ValueError, match='NaT'):
        loader.read_ts(corpus / 'NAT_1min_2023.csv', usecols=['close'])


def test_nat_error_shows_samples(loader, corpus):
    """報錯要附上實際壞掉的值,否則使用者無從判斷是資料問題還是解析問題。"""
    with pytest.raises(ValueError) as excinfo:
        loader.read_ts(corpus / 'NAT_1min_2023.csv', usecols=['close'])
    assert 'not-a-timestamp' in str(excinfo.value)


# ══════════════════════════════════════════════════════════════
#   五、空判定表 ≠ 全部放行
# ══════════════════════════════════════════════════════════════
def test_empty_manifest_does_not_open_the_gate(corpus):
    """★沒有判定表時,有歧義的檔一律擋下 —— 而不是「沒表就都放行」。"""
    from tzledger import Loader, LoaderConfig
    bare = Loader(LoaderConfig(root=corpus))          # 不給 manifests

    # 自述無歧義的仍然讀得到
    assert len(bare.read_ts(corpus / 'CCC_1min_2023.csv', usecols=['close'])) > 0

    # 有歧義的一律擋下,即使它其實是乾淨的 UTC
    with pytest.raises(UnresolvedTimezone):
        bare.read_ts(corpus / 'BBB_1min_2023.csv', usecols=['close'])
