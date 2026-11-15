# data-qd-118 · 需求预测滚动回测

Leakage-free rolling-origin demand forecast evaluation。业务输入、计算和当前边界详见 [数据契约](docs/contract.md)。

## 安装与演示

Python >=3.11，运行和测试仅用标准库；先确认 `python3 --version`。
每仓建立独立环境，构建工具锁版本。首次安装无公共缓存时可能联网；业务运行不访问网络。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-build.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps .
.venv/bin/python -m pip check
.venv/bin/demand-backtest --source fixtures/input.json --output run/report.json
.venv/bin/demand-backtest --vintage-source fixtures/vintage.json --output run/vintage.json
.venv/bin/demand-backtest --source fixtures/reconcile_calendar_input.json --output run/rec_calendar.json
.venv/bin/demand-backtest --vintage-source fixtures/reconcile_vintage_input.json --output run/rec_vintage.json
.venv/bin/demand-reconcile --report run/rec_calendar.json --report run/rec_vintage.json \
    --hierarchy fixtures/hierarchy.json --output run/reconciled.json
cat > run/batch.json <<'JSON'
{"txid": "seed-1", "expected_generation": 0,
 "puts": [{"name": "calendar", "path": "run/rec_calendar.json"},
          {"name": "vintage", "path": "run/rec_vintage.json"}],
 "drops": []}
JSON
.venv/bin/demand-store apply --db run/facts.db --manifest run/batch.json
.venv/bin/demand-store query --db run/facts.db > run/facts.json
.venv/bin/python -m unittest discover -s tests -v
```

本地已有wheel时，安装构建工具可加 `--no-index --find-links <公共wheel缓存目录>`。
修改源码后重新安装。本仓不共享其他仓环境，不依赖Docker、外部账号或持续服务。

输入JSON最多8MB，拒绝重复键、非有限数值和未定义字段；源字节SHA256随结果保留。
输出是可独立读取的JSON；成功后原子替换，输入/计算失败退出2并保留旧报告。
输入和输出不得相同，输出符号链接拒绝。单操作者批处理，不承诺并发发布或断电恢复。
同输入重跑不累加。测试使用临时目录，运行产物在忽略的run/。

src/demand_backtest/model.py 为业务模型，相关算法模块独立组织；common.py 提供受限JSON及报告发布，
cli.py 衔接数据链路，vintage.py 处理修订版回测。reconcile.py/reconcile_cli.py 把两份既有报告
协调到层级上，tree_qp.py 是精确树二次规划求解器。store.py/store_cli.py 把既有回测报告持久化为
SQLite预测事实仓（`demand-store`的`apply`/`query`），不重跑模型。fixtures 为明确自制样例，
代码与样例采用MIT许可。

## 业务与样例结果

日历回测：2个滚动origin共12条预测事实；naive MAE=3，moving_mean和seasonal_naive MAE均为4。
forecast.py只接收历史前缀，model.py建立逐origin事实与训练期尺度；无外部模型/下载。

修订回测（`--vintage-source`）：7个origin请求中5个可评估，共30条事实；bolts的naive MAE=25/8，
moving_mean和seasonal_naive MAE均为4；steady为平值序列，逐origin尺度为0，MASE=null。
每个origin只用发布日<=该origin的版本训练，目标取发布日<=score_as_of的最新版本；
lineage、actual_published_on与skipped原因便于审计，后发修订不改变早期预测。

层级协调（`demand-reconcile`）：消费上述两份报告，按`(origin,target_date,method)`分组，
每组精确求解`Σ weight×(x−forecast)²`在“父值=直接子值之和、叶值≥0”下的唯一最小值。
样例层级5节点两层，12组共60条事实；目标函数与叶梯度KKT证书均保留为Fraction，
父值严格等于子值之和，零叶为精确0（如steady）。

预测事实仓（`demand-store`）：`apply`消费上述既有报告，按`(series,origin,target_date,method)`
存事实并以报告字节SHA绑定归属；`query`在同一快照输出代号、报告清单、排序事实与归属。
代号从0起，每次成功事务原子递增；同txid同内容幂等返回原回执，异内容拒绝。详见
[数据契约](docs/contract.md)的Forecast fact store v1。

