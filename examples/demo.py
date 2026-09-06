# -*- coding: utf-8 -*-
"""跑一遍完整流程 —— 現生一小份語料,不需要任何外部資料。

    python examples/demo.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tzledger import Loader, LoaderConfig, UnresolvedTimezone   # noqa: E402
from tzledger.audit import assert_coverage, coverage, format_coverage   # noqa: E402
from tzledger.provenance import classify, report                # noqa: E402

RULE = '─' * 74


def build_corpus(root: Path) -> None:
    """造四個檔:兩個有時區歧義、一個自帶時區、一個沒登錄判定表。"""
    naive = pd.date_range('2023-03-01 20:00', periods=1440, freq='1min')
    price = [round(100 + i * 0.01, 4) for i in range(1440)]

    def write(name, stamps):
        pd.DataFrame({'timestamp': stamps, 'close': price}).to_csv(
            root / name, index=False)

    write('AAA_1min_2023.csv', naive.strftime('%Y-%m-%d %H:%M:%S'))   # 其實是台北時間
    write('BBB_1min_2023.csv', naive.strftime('%Y-%m-%d %H:%M:%S'))   # 其實是 UTC
    write('CCC_1min_2023.csv',
          naive.tz_localize('UTC').strftime('%Y-%m-%dT%H:%M:%S+00:00'))  # 自帶時區
    write('ZZZ_1min_2023.csv', naive.strftime('%Y-%m-%d %H:%M:%S'))   # 沒人判定過

    (root / 'tz_manifest.json').write_text(json.dumps({'records': [
        {'path': 'AAA_1min_2023.csv', 'verdict': 'TPE'},
        {'path': 'BBB_1min_2023.csv', 'verdict': 'UTC'},
    ]}, ensure_ascii=False, indent=1), encoding='utf-8')


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_corpus(root)
        loader = Loader(LoaderConfig(root=root, manifests=[root / 'tz_manifest.json']))

        # ① 三個判定得出來的檔:全部落在 tz-aware UTC
        print(RULE)
        print('① 三種來源格式 → 一種輸出')
        print(RULE)
        frames = {}
        for name in ('AAA_1min_2023.csv', 'BBB_1min_2023.csv', 'CCC_1min_2023.csv'):
            df = loader.read_ts(root / name, usecols=['close'])
            frames[name.split('_')[0]] = df
            print(f'  {name:<22} 首列 ts = {df["ts"].iloc[0]}  tz={df["ts"].dt.tz}')
        print('\n  ★ AAA 與 BBB 的原始字串完全相同,但判定不同 → 落在不同的 UTC 位置。')
        print('    這就是「時區判定」要當成外部事實記錄的理由:光看檔案內容分不出來。')

        # ② 沒判定過的檔:擋下來
        print()
        print(RULE)
        print('② 判不出時區的檔 → fail-closed')
        print(RULE)
        try:
            loader.read_ts(root / 'ZZZ_1min_2023.csv', usecols=['close'])
        except UnresolvedTimezone as exc:
            print('  ' + str(exc).replace('\n', '\n  '))

        # ③ ledger
        print()
        print(RULE)
        print('③ 這批資料是怎麼進來的')
        print(RULE)
        print(loader.ledger.summary().to_string(index=False))
        print()
        print('  ★ how 欄記的是「判定怎麼查到的」,不是「判定是什麼」——')
        print('    by_path 與 INHERITED_BY_BASENAME 的可信度不同,但 verdict 欄看不出來。')

        # ④ 跨序列覆蓋率
        print()
        print(RULE)
        print('④ 跨序列覆蓋率')
        print(RULE)
        cov = coverage(frames)
        print(format_coverage(cov))
        try:
            assert_coverage(cov)
            print('\n  三個序列密度一致,通過。')
        except ValueError as exc:
            print('\n  ' + str(exc).replace('\n', '\n  '))

        # ⑤ 結論身分
        print()
        print(RULE)
        print('⑤ 結論身分 —— 同一份資料,兩種用法')
        print(RULE)
        print('\n[A] 這次分析有用到日界線(resample 成日線):')
        report(classify(loader.ledger, declare_derived_path=True))
        print('\n[B] 同一份 ledger,但這次分析只做單一序列的順序式計算:')
        report(classify(loader.ledger, declare_derived_path=False))
        print('\n  ★ 時區位移不改變任何一列的內容,只改變「哪些列屬於同一天」。')
        print('    所以沒用到日界線的分析真的不受影響 —— 身分是「資料 × 用法」的屬性。')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
