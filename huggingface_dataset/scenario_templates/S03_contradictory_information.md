# S03: 矛盾信息 (Contradictory Information)

## 场景描述
关于同一事实的不同版本在不同时间被记录。Agent需要处理版本冲突，判断哪个信息是"当前有效"的。

## 核心双时间轴测试点
- 版本冲突检测
- 按event_time判断信息的时效性
- 按record_time判断信息的接收顺序
- 两个时间轴可能给出不同的"最新"信息

## 示例模板
```
领域: [corporate/academic/scientific]
冲突事实: [数据/结论/状态]

w1: 来源A在[T1]记录了事实F1  → event_time=T1, record_time=T1
w2: 来源B在[T2]记录了事实F2（与F1矛盾）→ event_time=T2, record_time=T2
w3: 来源C在[T3]确认了F1或F2  → event_time=T3, record_time=T3
```

## 关键约束
- 必须包含至少2个相互矛盾的信息
- 矛盾信息应有不同的来源或记录时间
- 需要有明确的"正确"版本（通过后续信息确认）
- 矛盾的检测需要区分event_time和record_time

## QA生成指引
- Level 1: "在时间T，Agent认为事实F是什么？"（选择正确版本）
- Level 2: "关于事实F，Agent在不同时间点的认知发生了什么变化？"（版本切换追踪）
- Level 3: "如果按record_time排序，Agent对F的认知是什么？按event_time排序呢？两种排序的结果有何不同？"
