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

## Forecast fact store v1（`demand-store`）

`demand-store`把既有回测报告持久化为SQLite预测事实仓，提供`apply`与`query`两个子命令；
**不重跑模型**，只消费既有`schema_version=1`报告。主入口`demand-backtest`与
`demand-reconcile`保持不变。事实键为`(series, origin, target_date, method)`，
同键事实以`horizon`、`forecast`、`actual`、`scale`四个精确值判定是否相同，全部用规范
Fraction文本读写，不四舍五入。

### apply

```
demand-store apply --db facts.db --manifest batch.json
```

清单为严格JSON对象`{"txid","expected_generation","puts":[{"name","path"}],"drops":[{"name"}]}`，
拒绝重复键、非有限数值和未定义字段。`txid`为1..128字符标识符；`expected_generation`为
0..2^63-2整数；`puts`与`drops`各至多1000项。**名称在批内互斥**：同一`name`不得在
`puts`或`drops`中出现两次。`name`是报告在仓内的归属名（非路径），`path`只用于当次读取。

每个`puts.path`必须是一份既有报告：日历回测报告或`mode=revision_vintage`报告，字段形状与
`demand-reconcile`一致（`schema_version=1`、`source_sha256`为小写SHA256、`result.forecasts`）。
单份报告上限200000条事实、128MB；清单上限8MB。报告内同键重复即拒绝。
两条记录可共享同一键当且仅当四个值完全相同；完全相同则共享一条事实并保留**所有**报告归属；
任一值不同则**整批拒绝**，不写入任何内容。`actual`非负、`scale`非负，均为规范Fraction
或非负整数；`actual_published_on`、`error_forecast_minus_actual`等报告附带字段忽略。

替换/撤回语义：先删除`drops`中名称的全部归属，再以**批后状态**判定冲突，最后由
`puts`建立归属。因此：
- 只撤回一个共享报告：事实仍有其他归属，事实保留、归属减少；
- 一次批内同时撤回全部旧报告并放入更正报告（即使数值改变）：允许，因为冲突按批后状态判定；
- 只替换共享事实的一方而另一方仍持有旧值：冲突，整批拒绝；
- 撤回仓内不存在的名称：拒绝；
- `puts`名称已在仓内：拒绝。名称批内互斥意味着同一名称不能既`drops`又`puts`，因此更正
  须在同一批用新名称放入更正报告并撤回旧名称（见上一条）。`reports`目录按名称唯一；
  报告字节SHA相同不等于名称相同，两条同字节报告各自保留归属。

只有某条事实的**最后一个**归属消失时才删除该事实。代号（`generation`）初始为0，
每次成功`apply`递增1；写入的事实、归属、报告目录、报告字节SHA、账本与回执在同一事务内
原子提交，仅当`expected_generation`等于当前已提交代号时才执行，否则退出2且逻辑库不变。
回执包含`txid`、新代号、批内容SHA256、`facts_added`/`facts_removed`/`facts_shared`、
当前`fact_count`、完整报告目录与`dropped`。

幂等重试：同`txid`且**批内容相同**时，返回原始回执（即使当前代号已前进，即提交后崩溃/
丢失回执后重试仍安全）；同`txid`但内容不同则拒绝。内容指纹绑定**报告字节SHA256**、
名称与`drops`，**不绑定路径**——同一字节的报告换路径重试仍视为同内容。

### query

```
demand-store query --db facts.db
```

在**同一读快照**中输出`schema_version=1`、`mode=forecast_store_query`、`generation`、
按`name`排序的报告目录（`name`/`kind`/
`source_sha256`/`report_sha256`/`fact_count`）、按
`(series, origin, target_date, method)`排序的事实（含四个精确值与按名称排序的`reports`
归属）与`fact_count`。不会混代：读到的是一个已提交代号及其完整状态，绝无半提交批次。

### 失败与文件约束

校验/冲突/代号不匹配/未知撤回/类型不符等失败一律退出2，且逻辑库（事实、归属、报告目录、
账本与代号）完全不变；校验失败发生在首次写库前，因此不会新建库文件。`--db`为符号链接、
目录或非仓SQLite文件时拒绝。库与清单、库与任一源报告必须是不同文件：库路径等于清单或
任一`puts`源报告时退出2，**绝不覆盖源报告**；两个`name`可指向同一源文件（同字节则共享）。
`apply`以SQLite事务＋WAL提交，支持崩溃回滚与提交后幂等重试；单写者、`BEGIN IMMEDIATE`
串行化，未抢到同代的进程退出2。

样例操作：先用既有`demand-backtest`生成两份报告（如
`fixtures/reconcile_calendar_input.json`→日历报告、`fixtures/reconcile_vintage_input.json`
→修订报告），清单`puts`两项后`apply`，得到`generation=1`、60条事实；`query`可复核归属。

## Minimum-cost replenishment plan v1（`demand-plan`）

`demand-plan --db facts.db --config plan.json --output plan_report.json`从预测事实仓的
**只读**截面生成最小成本补货计划；事实仓以`mode=ro`打开，规划运行绝不写入、不修改源库。
`--db`、`--config`、`--output`必须是三个不同文件，输出为符号链接时拒绝；校验或不可行失败
退出2且不触碰已有输出，成功后原子替换。配置沿用严格JSON（拒绝重复键、非有限数值、未定义字段），
上限8MB；配置字节SHA256随结果保留。

### 配置与截面

配置为`{"origin","method","series","daily_capacity","initial_batches"}`：

- `series`：1..20项，每项含`series`、`lead_time_days`（0..100000）、`shelf_life_days`
  （1..100000）、`unit_cost`与`holding_cost`（0..10^9非负整数），序列名唯一。
- `daily_capacity`：逐日共享订货容量，`date/capacity`，容量为0..10^9整数；日期集合必须恰好是
  订单日集合（`origin`至最后一个目标日，含两端）。
- `initial_batches`：0..600个初始批次`(series,available,expires,quantity)`，数量0..10^9；
  `available>=origin`、`expires>available`，批次可用区间为左闭右开`[available,expires)`。

读取条件：同一`(origin,method)`截面非空；其序列集合与配置完全一致；每个序列的目标日期相同、
按日历日连续，且全部严格晚于`origin`、距`origin`不超过30日。预测值为规范Fraction文本，
负值拒绝；需求为预测值**向上取整**后的非负整数，取整后超过10^9拒绝。

### 时间、成本与优化问题

订单日为`origin`至最后目标日；订单日`t`下达、`t+lead_i`到货，保质期从到货日起算，
订购批次可用于目标日`k`当且仅当`arrival<=k<arrival+shelf_i`（左闭右开，到期日不可用）。
库存可免费弃置，需求必须在当日准时满足。单位成本在采购节点一次性计入；持有费为
`holding_i`/单位/日，从批次可用日（订购批次为到货日）计至使用日。所有序列共享每个订单日的
订货容量。容量与数量均为0..10^9整数，成本为非负整数，因此最优流为整数流。

该问题建模为单一联合整数最小费用流：源→共享日容量节点→各（序列,订单日）采购节点→
各（序列,目标日）需求节点→汇；初始批次直接挂在源上。**禁止逐序列贪心、禁止逐单位展开**：
求解采用逐次最短路，每次按瓶颈数量整批增广，最短路在完整残量图上以Bellman-Ford重算，
使增广后才可达的节点仍持有有效势。最坏规模（20序列、30目标日、600批次）约2000节点、
8000条边，实测秒级完成。

### 输出

`schema_version=1`、`mode=demand_plan`、`config_sha256`与`source`（源代号：事实仓类型、
读取时的`generation`与截面三元组及事实数），`result`含：

- `status`：`feasible`/`infeasible`；`origin/method/generation`、截面事实数、目标日期、
  回显的序列参数与逐日容量、初始批次（含`used`，可小于`quantity`即弃置）。
- `demand`：每序列每目标日取整后的整数需求。
- `orders`：`(series,order_date,arrival_date,expires_date,quantity)`整数订单，按
  `(series,order_date)`排序；`allocations`：需求的整数来源分配，标明`order`（引用订单日）
  或`initial`（引用批次号），按序列/目标日/来源排序。
- `purchase_cost`、`holding_cost`、`total_cost`（均为精确整数，满足
  总成本=采购费+从可用日至使用日的持有费）；`flow`给出总需求、已满足与缺口，
  `demand_shortfall`逐行列未满足量。
- `certificate`供独立验算：
  - 可行时`kind=potentials`，给出全部节点（稳定整数id与类型/引用）、全部边
    （容量/流量/费用/类型）与节点势；独立验算只需重建残量弧并验证
    每条残量弧`u→v`满足`cost+potential[u]-potential[v]>=0`，且每条需求边饱和。
  - 不可行时`kind=min_cut`，给出残量可达集`reachable`、前向割边`cut_edges`、
    `cut_capacity=max_flow=satisfied`；可达集对正残量弧封闭、含源不含汇，缺口等于
    总需求减最大流。不可行本身是正常业务结果，退出0；只有坏输入才退出2。

样例`fixtures/plan_config.json`配合`fixtures/input.json`生成的事实仓：naive预测在两个目标日
均为6，6单位初始批次免费覆盖首日，次日由6单位新采购按单位成本5满足，总成本30。
