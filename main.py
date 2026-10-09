"""项目统一命令行入口。

用法::

    python main.py fetch   --symbol MSFT
    python main.py analyze --symbol MSFT
    python main.py run     --symbol MSFT
    python main.py review

原有接口 ``python fetch_data.py --tickers MSFT --period 3mo`` 依然可用。
"""

import sys

from stock_analysis.cli import main

if __name__ == "__main__":
    sys.exit(main())
