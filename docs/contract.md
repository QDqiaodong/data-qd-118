# Rolling-origin demand evaluation v1

为需求分析人员比较三个简单预测基线：最近值、固定窗口移动均值、季节朴素。
observations每行series/date/quantity，日期为ISO日历日，quantity非负整数<=10^9。
各series必须连续每日、无重复；缺日是未知数据而非0，拒绝。输入最多20000行。
参数initial_train>=2、horizon=1..100、mean_window和season均为1..initial_train。
每序列必须至少initial_train+horizon天。日期可乱序，解析后按序列和日期排序。

从initial_train个历史点开始，每天扩展一次训练前缀，仅评估实际值完整覆盖的horizon。
每次只把训练前缀传给forecast函数；多步预测不把预测窗口中的已知实际值反喂模型。
naive使用最后一值，moving_mean用最后mean_window个历史点均值，seasonal_naive循环最后season个值。
即使horizon>season，也只重复最后一个历史季节块。没有调参、测试集选择或未来特征。

输出每个origin的训练终点/数量/MASE尺度，所有origin-target-method预测事实与逐序列逐模型指标。
误差=预测-实际；MAE、MSE、bias按预测事实等权。WAPE=sum|误差|/sum实际，分母0为null。
MASE尺度每个origin单独用训练前缀lag1平均绝对差，不使用未来或全序列尺度；
尺度0的事实不计MASE并显式给mase_count，全部为0时MASE=null。
结果为精确Fraction文本，如3、7/2，不做四舍五入。没有置信区间、百分比格式或加权总体指标。

相邻origin的目标日期可重叠，这是不同预测决策，不去重；不把它们称独立实验样本。
最多200000预测事实，超出拒绝。样例2个origin、12个预测事实；naive MAE=3、moving_mean=4、seasonal_naive=4。
当前无训练型机器学习、库存补货决策、实时服务、数据抓取或分布漂移检测。

## Revision vintage v1（`--vintage-source`）

`--vintage-source`与`--source`互斥；前者按修订版本回测，后者为上述日历回测，旧入口不变。
输入沿用四个预测参数initial_train/horizon/mean_window/season，以revisions替换observations，
另加origins和score_as_of。

revisions每行series/date/published_on/quantity：date是该业务日，published_on是该版本发布日，
均须ISO日历日且date<=published_on，否则拒绝。同一series/date/published_on只允许一个quantity；
相同重复折叠为一个版本，quantity不同则冲突并拒绝。origins是series/date行数组，同键重复折叠。
一行修订只表示当天的一个版本，不要求每序列每日连续；训练前缀要求从序列最早业务日起连续日历日，
不足initial_train、缺日或该起点前尚无可用版本时，该origin跳过并给出reason/detail。

每个origin只使用发布日<=origin的最新版本：从序列最早业务日逐日取该日最新已发布值组成训练前缀，
同一规则不允许后发修订进入早期训练。目标实际值取发布日<=score_as_of的最新版本；
若某horizon目标日没有<=score_as_of的版本，该origin整条跳过，不做部分展开。
origins可乱序，输出按(series,date)排序且同键去重，因此顺序确定。

每个origin独立训练前缀尺度与Fraction指标，规则与日历回测一致：三个基线、MAE/MSE/bias/WAPE/MASE、
逐序列逐模型汇总、最多200000预测事实。lineage列出每series/date的所有版本及其latest/scored标记，
forecasts每条带actual_published_on，skipped列出跳过origin及reason=unknown_series、
insufficient_history、missing_calendar_date、missing_training_vintage、missing_target、
missing_target_vintage之一。

样例fixtures/vintage.json：2序列、7个origin请求、5个被评估（2个跳过），30条预测事实。
bolts的naive MAE=25/8、moving_mean=4、seasonal_naive=4；steady整段平值，尺度为0，MASE=null。
把后发修订追加进输入不会改变其发布前的任何预测事实。

## Hierarchy reconciliation v1（`demand-reconcile`）

`demand-reconcile`消费两份既有回测报告，输出可加总的非负层级预测；不回写、不修改两份输入。
`--report`必须出现两次（一份日历回测报告、一份修订回测报告），`--hierarchy`指向层级文件，
`--output`为唯一写目标。主入口`demand-backtest`保持不变。

hierarchy是`{"nodes":[{"series","parent","weight"}]}`，节点数<=200。`series`唯一且非空；
`parent`为`null`（根）或另一节点`series`，必须形成**单根树**：恰好一个根、每个父引用有效、
从根可达全部节点（因此也拒绝与根脱节的环）。`weight`为正Fraction字符串，如`1`、`3/2`、`50`；
非正、不可解析或分量超过18位数字均拒绝。每个节点对应一条序列，层级内不得有同名序列。

两份报告均为既有`schema_version=1`输出：日历回测报告与`mode=revision_vintage`报告，
各取`result.forecasts`。把两份事实合并后按`(origin,target_date,method)`分组：
同一组必须包含层级里全部且仅有的序列各一条预测事实，且`horizon`一致；
重复事实（同组同序列出现两次，无论来自哪份报告）拒绝。
不同`origin`即便`target_date`重叠也绝不合并；`actual`及其附带字段不参与协调。
空组合法（零个事实）：报告为`group_count=0`，不发明预测。
两份报告任一份超出200000条预测事实，或合并后总数超过该上限，均拒绝。

每组精确求解
`minimize Σ_i weight_i × (x_i − forecast_i)²`
`s.t. x_parent = Σ_{直接子} x_child，x_leaf ≥ 0`。
根值由直接子值之和决定，父值不独立；叶值非负。最优解唯一（严格凸且约束等价于仿射消除）。
算法在树上做等式消元＋原-对偶活跃集，只处理当前工作集，**不枚举全部活跃集**，
因此节点数200、总事实20万的输入仍可用（20万事实由1000个200节点组构成）。
禁止的做法（不采用）：先按0截断负叶再仅按截断值重算父级——那只满足可行性却非最优。

输出`result`：
- `groups`：按`(origin,target_date,method)`排序的确定序列，每组含`inputs`（来源报告种类）、
  `horizon`、`nodes`（按`series`排序，含`parent`、`weight`、`forecast`原值、`reconciled`协调值、
  `leaf`、`zero`，均为规范Fraction文本）、`objective`目标函数值、`zero_leaves`零叶序列、
  `leaf_kkt`叶梯度证书（`series`/`value`/`gradient`；非零叶梯度必为0，零叶梯度≥0）。
- `hierarchy`：节点数、根、叶序列、层级文件SHA256。
- `reports`：两份输入的`kind`、报告自身`source_sha256`、报告字节SHA256。
- `input_sha256`：三份输入字节的SHA256。
- `fact_count`、`group_count`、`aggregation`、`evaluation_unit`说明。

绑定：输出记录两份报告与层级文件的字节SHA256；输入顺序不影响输出（按`kind`归一）。
两份报告是既有发布产物，200000条事实约合55MB，故报告逐个上限128MB、层级文件仍为8MB；
两者都沿用既有严格JSON解析（拒绝重复键、非有限数值）。报告由既有`demand-backtest`生成，
每份报告本身已受200000事实上限约束。
坏输入（缺字段、多余字段、非规范Fraction、非树层级、组不齐、horizon不一致、重复事实、
超限、报告类型不符等）退出码2，且不触碰已有输出文件。两个输入文件绝不覆盖：
`--output`等于任一输入路径时退出2。成功后以原子替换写出。
样例`fixtures/reconcile_calendar_input.json`与`fixtures/reconcile_vintage_input.json`
经`demand-backtest`生成两份报告，再用`fixtures/hierarchy.json`（5节点、两层、含正权重Fraction）
协调，得到12组、60条事实；必须出现最优解触零的叶（如`steady`、部分组`nuts`）。
