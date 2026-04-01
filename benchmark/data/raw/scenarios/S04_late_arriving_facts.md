# S04: 迟到事实 (Late-Arriving Facts)

## 场景描述
事件在过去发生，但信息在很久之后才被记录。event_time远早于record_time，测试Agent是否能正确回溯历史时间线。

## 核心双时间轴测试点
- 双时间轴gap推理（event_time << record_time）
- 迟到事实对已有认知的修正
- 时间线重组能力

## 示例模板
```
领域: [corporate/historical/scientific]
迟到类型: [延迟报告/补充信息/历史发现]

w1: [事实F1]发生在[T1]，但直到[T1+N]才被记录 → event_time=T1, record_time=T1+N
w2: [事实F2]发生在[T2]，在[T2+Δ]被记录     → event_time=T2, record_time=T2+Δ
（其中T1 < T2 但 T1+N > T2+Δ，即较早的事件反而较晚被记录）
```

## 关键约束
- 至少1个memory_write的event_time与record_time间隔 > 1周
- 至少1对writes的event_time顺序与record_time顺序相反
- 迟到事实必须能修正或补充已有认知
- 时间间隔要有意义（如天/周/月级别）

## QA生成指引
- Level 1: "事件X实际发生在什么时间？"（区分event_time和record_time）
- Level 2: "在收到迟到事实之前，Agent对时间T的认知是什么？收到之后呢？"（认知修正）
- Level 3: "如果迟到事实从未被记录，Agent对历史事件的理解会有什么系统性偏差？"
