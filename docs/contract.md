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
