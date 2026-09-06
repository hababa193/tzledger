# -*- coding: utf-8 -*-
"""tzledger —— fail-closed 的時間序列載入層。

讀進來的 ``ts`` 欄保證是 tz-aware UTC;判不出時區的檔直接拒絕載入;
每一次載入都在 ledger 留下可稽核的紀錄。

最小用法
--------
    from tzledger import LoaderConfig, Loader

    loader = Loader(LoaderConfig(root='data', manifests=['data/tz_manifest.json']))
    df = loader.read_ts('data/AAA_1min_2023.csv', usecols=['close'])
    assert str(df['ts'].dt.tz) == 'UTC'

    print(loader.ledger.summary())
"""
from .config import (
    BAD_VERDICTS,
    DEFAULT_CORPUS_HINTS,
    TZ_OF,
    LoaderConfig,
)
from .core import (
    Ledger,
    LedgerEntry,
    Loader,
    UnresolvedTimezone,
    configure,
    default_loader,
    ledger,
    read_ts,
    sniff_format,
)
from .provenance import Provenance, classify, report

__version__ = '0.1.0'

__all__ = [
    # 設定
    'LoaderConfig', 'TZ_OF', 'BAD_VERDICTS', 'DEFAULT_CORPUS_HINTS',
    # 核心
    'Loader', 'Ledger', 'LedgerEntry', 'UnresolvedTimezone', 'sniff_format',
    'read_ts', 'configure', 'default_loader', 'ledger',
    # 身分
    'Provenance', 'classify', 'report',
    '__version__',
]
