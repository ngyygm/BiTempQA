# S01: 实体属性演化 (Entity Attribute Evolution)

## 场景描述
追踪一个或多个实体的属性随时间变化的过程。Agent的记忆系统中需要正确维护实体不同时间点的属性版本。

## 核心双时间轴测试点
- event_time排序 vs record_time排序的差异
- 同一属性在不同时间点的值变化
- 属性版本的覆盖式更新

## 示例模板
```
领域: [corporate/academic/social/fictional/historical/scientific]
实体: [人物/组织/产品]
属性: [职位/地点/状态/关系]

w1: [实体]在[时间T1]具有属性[A1]  → event_time=T1, record_time=T1+Δ
w2: [实体]在[时间T2>T1]变为属性[A2]  → event_time=T2, record_time=T2+Δ
w3: [实体]在[时间T3>T2]变为属性[A3]  → event_time=T3, record_time=T3+Δ
```

## 关键约束
- 每个场景至少3个memory_writes
- 属性变化必须有时间顺序（event_time单调递增）
- record_time可以与event_time有gap（模拟信息延迟到达）
- 至少包含1个跨版本查询的QA对

## QA生成指引
- Level 1: "在时间T，X的属性A是什么？"（单时间点查询）
- Level 2: "在[T1,T2]期间，X的属性A经历了什么变化？"（时间段查询）
- Level 3: "如果在T2之前Agent不知道w2的信息，Agent会认为X的属性A是什么？"（反事实）
