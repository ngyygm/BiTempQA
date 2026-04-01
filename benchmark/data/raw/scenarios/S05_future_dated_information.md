# S05: 未来信息 (Future-Dated Information)

## 场景描述
记录包含关于未来计划或预期事件的信息。event_time在record_time之后，测试Agent区分"已发生"和"计划中"的能力。

## 核心双时间轴测试点
- 计划vs实际区分（event_time > record_time表示未来事件）
- 计划的后续更新（计划变更/取消/实现）
- 查询时间在计划时间前后的不同回答

## 示例模板
```
领域: [corporate/academic/personal]
未来类型: [计划/预测/安排/承诺]

w1: 在[T1]记录了将在[T3>T1]发生的计划P  → record_time=T1, event_time=T3
w2: 在[T2]记录了当前状态S               → record_time=T2, event_time=T2
w3: 在[T4>T3]确认计划P的实际结果        → record_time=T4, event_time=T4
```

## 关键约束
- 至少2个memory_writes的event_time > record_time
- 必须包含计划的后续状态（实现/变更/取消）
- 查询时间需要覆盖"计划前""计划中""计划后"三个阶段
- 未来信息与当前事实需要有关联

## QA生成指引
- Level 1: "在时间T（计划执行前），Agent知道X将发生什么？"（未来查询）
- Level 2: "关于事件X，从计划到最终结果，Agent的认知经历了哪些变化？"（计划追踪）
- Level 3: "在计划时间T之前查询，Agent的回答是什么？在T之后呢？如果计划被取消了回答又会变成什么？"
