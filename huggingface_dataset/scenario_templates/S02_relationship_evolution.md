# S02: 关系演化 (Relationship Evolution)

## 场景描述
追踪两个或多个实体之间关系随时间变化的过程。关系可能建立、变化、结束，甚至重新建立。

## 核心双时间轴测试点
- 关系版本链的追溯
- 关系开始/结束时间点与记录时间的差异
- 多对多关系的并行存在

## 示例模板
```
领域: [corporate/social/fictional]
实体对: [人物-人物/公司-公司/人物-公司]
关系类型: [合作/竞争/上下级/师生/朋友]

w1: [A]和[B]在[T1]建立了[关系R]  → event_time=T1
w2: [A]和[B]的关系在[T2]变为[R'] → event_time=T2
w3: [A]和[B]在[T3]结束了关系     → event_time=T3
```

## 关键约束
- 每个场景至少涉及2对关系
- 至少1对关系有完整生命周期（建立→变化→结束）
- 关系的valid_from/valid_until必须精确
- 至少1个memory_write的event_time与record_time有显著gap

## QA生成指引
- Level 1: "在时间T，A和B是什么关系？"（单时间点关系查询）
- Level 2: "A和B的关系经历了哪些变化？"（关系变更序列）
- Level 3: "如果在T2时Agent只收到了部分信息，它对A和B关系的理解会有什么不同？"
