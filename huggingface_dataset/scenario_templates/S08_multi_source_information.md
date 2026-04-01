# S08: 多源信息 (Multi-Source Information)

## 场景描述
同一事实从不同来源在不同时间到达Agent。不同来源可能有不同的详细程度、准确性或视角。

## 核心双时间轴测试点
- 多源时间线融合
- 不同来源到达时间对认知的影响
- 多源信息冲突时的版本选择

## 示例模板
```
领域: [corporate/news/scientific]
来源类型: [内部报告/外部新闻/同事告知/系统通知]

w1: 来源A在[T1]提供了事实F（概要版）     → source=A, event_time=T1, record_time=T1
w2: 来源B在[T2]提供了事实F（详细版）     → source=B, event_time=T2, record_time=T2
w3: 来源C在[T3]提供了关于F的补充信息G    → source=C, event_time=T3, record_time=T3
```

## 关键约束
- 至少3个不同来源
- 来源到达时间(record_time)必须不同
- 不同来源的信息内容有重叠但不完全相同
- 至少1对来源提供的信息有细微差异

## QA生成指引
- Level 1: "在时间T，Agent知道关于F的哪些信息？"（综合多源）
- Level 2: "关于F，Agent的认知如何随着不同来源信息的到达而逐步丰富？"（渐进认知）
- Level 3: "如果只收到来源A和B（没有C），Agent对F的理解会有什么缺失？"
