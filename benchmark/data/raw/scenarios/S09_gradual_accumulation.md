# S09: 渐进积累 (Gradual Accumulation)

## 场景描述
关于同一实体或事件的信息分片段逐步到达。每个片段只提供部分信息，Agent需要逐步拼凑完整图景。

## 核心双时间轴测试点
- 时间点知识完整性评估
- 信息片段的时序组装
- 部分信息状态下的推理准确性

## 示例模板
```
领域: [corporate/scientific/investigative]
积累类型: [逐步披露/分批数据/调查进展]

w1: 在[T1]获得片段1（基本信息）     → event_time=T1, record_time=T1
w2: 在[T2]获得片段2（补充细节）     → event_time=T2, record_time=T2
w3: 在[T3]获得片段3（关键缺失部分）  → event_time=T3, record_time=T3
w4: 在[T4]获得片段4（最终确认）     → event_time=T4, record_time=T4
```

## 关键约束
- 至少4个memory_writes（体现"渐进"特征）
- 每个片段单独不完整，需要组合才能获得全貌
- 至少1个关键片段迟到到达（event_time较早但record_time较晚）
- 需要有"信息完整度"从低到高的明显梯度

## QA生成指引
- Level 1: "在时间T（收到前2个片段后），Agent知道X的哪些信息？"（部分知识查询）
- Level 2: "从T1到T4，Agent对X的认知经历了哪些关键变化？每次变化的触发信息是什么？"（积累过程）
- Level 3: "如果片段3从未到达，Agent基于片段1、2、4会得出什么不完整的结论？"
