# -*- coding: utf-8 -*-
"""測試用的合成語料。

刻意不依賴任何真實資料檔:原型版本的測試綁死在作者機器上的絕對路徑,
於是那些測試**只有一台機器跑得動**,等同沒有測試。
這裡全部在 ``tmp_path`` 現生,clone 下來就能跑。

合成資料同時解決另一個問題:真實語料裡沒有「壞掉的時間戳」這種樣本,
所以用真實資料寫不出 NaT 的測試 —— 而那正是最需要被測的分支。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tzledger import Loader, LoaderConfig
from tzledger.compat import uninstall

# 全部 fixture 共用的一段時間軸:2023-03-01 起,每分鐘一根,共 240 根。
# 跨了台北的午夜(UTC 16:00),所以 TPE→UTC 的修正一定會改變「哪些列屬於同一天」。
BASE = pd.Timestamp('2023-03-01 20:00:00')
N_ROWS = 240


def _prices(n: int = N_ROWS) -> list:
    """決定性的假價格 —— 不用亂數,測試失敗時才能重現。"""
    return [round(100.0 + i * 0.25, 4) for i in range(n)]


def _write(path: Path, stamps, prices=None, ts_col: str = 'timestamp') -> Path:
    frame = pd.DataFrame({
        ts_col: stamps,
        'close': prices if prices is not None else _prices(len(stamps)),
    })
    frame.to_csv(path, index=False)
    return path


# ══════════════════════════════════════════════════════════════
#   語料檔
# ══════════════════════════════════════════════════════════════
@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """建一棵小語料樹,回傳根目錄。"""
    root = tmp_path / 'corpus'
    root.mkdir()

    naive = pd.date_range(BASE, periods=N_ROWS, freq='1min')

    # ① 原始時間戳是台北時間,但檔案裡是裸字串 → 有歧義,必須查判定表
    _write(root / 'AAA_1min_2023.csv', naive.strftime('%Y-%m-%d %H:%M:%S'))

    # ② 原始時間戳就是 UTC,裸字串 → 一樣有歧義,一樣要查表
    _write(root / 'BBB_1min_2023.csv', naive.strftime('%Y-%m-%d %H:%M:%S'))

    # ③ 自帶時區 → 無歧義,不需要判定表
    aware = naive.tz_localize('UTC')
    _write(root / 'CCC_1min_2023.csv', aware.strftime('%Y-%m-%dT%H:%M:%S+00:00'))

    # ④ epoch 毫秒 → 無歧義
    #   先固定成 ns 再換算:pandas 3 的 date_range 預設是 us 精度,
    #   直接 astype('int64') 會差三個數量級
    epoch_ns = aware.as_unit('ns').astype('int64')
    _write(root / 'DDD_1min_2023.csv', epoch_ns // 10 ** 6)

    # ⑤ epoch 秒 → 無歧義
    _write(root / 'EEE_1min_2023.csv', epoch_ns // 10 ** 9)

    # ⑥ 裸字串且**不在判定表裡** → 應該被 fail-closed 擋下
    _write(root / 'ZZZ_1min_2023.csv', naive.strftime('%Y-%m-%d %H:%M:%S'))

    # ⑦ 含壞掉的時間戳 → 應該因 NaT 被擋下
    broken = list(naive.strftime('%Y-%m-%d %H:%M:%S'))
    broken[7] = 'not-a-timestamp'
    broken[19] = ''
    _write(root / 'NAT_1min_2023.csv', broken)

    # ⑧ 已鑑定為「內容與檔名不符」→ 進黑名單
    _write(root / 'WRONG_1min_2023.csv', naive.strftime('%Y-%m-%d %H:%M:%S'))

    # 判定表:只登錄前兩個 + 黑名單那個
    manifest = {
        'records': [
            {'path': 'AAA_1min_2023.csv', 'verdict': 'TPE'},
            {'path': 'BBB_1min_2023.csv', 'verdict': 'UTC'},
            {'path': 'NAT_1min_2023.csv', 'verdict': 'UTC'},
            {'path': 'WRONG_1min_2023.csv', 'verdict': 'UTC'},
        ]
    }
    (root / 'tz_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    return root


@pytest.fixture
def config(corpus: Path) -> LoaderConfig:
    return LoaderConfig(
        root=corpus,
        manifests=[corpus / 'tz_manifest.json'],
        blacklist={'WRONG_1min_2023.csv'},
    )


@pytest.fixture
def loader(config: LoaderConfig) -> Loader:
    return Loader(config)


@pytest.fixture(autouse=True)
def _no_leaked_patch():
    """每個測試結束都還原 ``pd.read_csv``。

    相容層是 monkeypatch,漏掉還原會讓後面的測試在被 patch 的環境下跑,
    產生「單獨跑會過、整批跑會失敗」這種最難查的測試。
    """
    yield
    uninstall()


# ══════════════════════════════════════════════════════════════
#   工具
# ══════════════════════════════════════════════════════════════
@pytest.fixture
def expected_utc():
    """AAA(台北時間)修正後應有的 UTC 時間軸。"""
    return (pd.date_range(BASE, periods=N_ROWS, freq='1min')
            .tz_localize('Asia/Taipei').tz_convert('UTC'))


# ══════════════════════════════════════════════════════════════
#   有日光節約的時區 —— 宣告式與算術真正分出高下的地方
# ══════════════════════════════════════════════════════════════
#
# 台北沒有 DST,所以 `tz_localize('Asia/Taipei')` 與 `- 8h` 數值完全相同,
# 用台北的資料**測不出**兩者的差別。要證明宣告式為什麼比較好,
# 必須用一個偏移量會變的時區。
#
# 美東 2023-03-12:當地 02:00 EST 直接跳到 03:00 EDT,偏移從 -5 變成 -4。
# 固定加減必然在轉換點之後錯一小時。

NY_TRANSITION_UTC = pd.Timestamp('2023-03-12 06:30:00', tz='UTC')
NY_ROWS = 120          # 橫跨轉換點前後各一小時


@pytest.fixture
def dst_corpus(tmp_path: Path) -> Path:
    """一段橫跨美東 DST 轉換的本地時間資料。"""
    root = tmp_path / 'dst'
    root.mkdir()

    # 先在 UTC 上造連續的一分鐘網格,再換算成紐約當地時間寫出去。
    # 這樣產生的本地時間會**自然跳過不存在的那一小時**,與真實匯出檔一致。
    utc_grid = pd.date_range(NY_TRANSITION_UTC, periods=NY_ROWS, freq='1min')
    local = utc_grid.tz_convert('America/New_York').tz_localize(None)
    _write(root / 'NYSE_1min_2023.csv', local.strftime('%Y-%m-%d %H:%M:%S'))

    # 另一份:含**不存在的本地時間**(02:00–02:59 EST 在轉換日並不存在)
    bad = pd.date_range('2023-03-12 01:50:00', periods=20, freq='1min')
    _write(root / 'NONEXIST_1min_2023.csv', bad.strftime('%Y-%m-%d %H:%M:%S'))

    manifest = {'records': [
        {'path': 'NYSE_1min_2023.csv', 'verdict': 'NYC'},
        {'path': 'NONEXIST_1min_2023.csv', 'verdict': 'NYC'},
    ]}
    (root / 'tz_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    return root


@pytest.fixture
def dst_loader(dst_corpus: Path) -> Loader:
    """判定表多一個 ``NYC`` verdict,對應到有 DST 的時區。"""
    from tzledger.config import TZ_OF
    tz_of = dict(TZ_OF)
    tz_of['NYC'] = 'America/New_York'
    return Loader(LoaderConfig(
        root=dst_corpus,
        manifests=[dst_corpus / 'tz_manifest.json'],
        tz_of=tz_of,
    ))


@pytest.fixture
def dst_expected_utc():
    """NYSE 檔修正後應有的 UTC 時間軸 —— 連續的一分鐘網格,沒有缺口。"""
    return pd.date_range(NY_TRANSITION_UTC, periods=NY_ROWS, freq='1min')
