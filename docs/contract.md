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
