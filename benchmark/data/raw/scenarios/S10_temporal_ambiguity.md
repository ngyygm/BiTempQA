# S10: 时间歧义消解 (Temporal Ambiguity Resolution)

## 场景描述
查询中包含时间歧义，需要Agent区分event_time和record_time来给出正确答案。例如"什么时候知道的"vs"什么时候发生的"。

## 核心双时间轴测试点
- 双轴语义区分（"什么时候发生"vs"什么时候知道"）
- 歧义查询的消解策略
- 同一查询在不同语义下的不同答案

## 示例模板
```
领域: [任意]
歧义类型: [发生时间vs记录时间/当前状态vs历史状态]

w1: 事件E发生在[T1]，但直到[T1+N]才被Agent记录 → event_time=T1, record_time=T1+N
w2: 事件F发生在[T2]，在[T2+Δ]被记录           → event_time=T2, record_time=T2+Δ

歧义查询: "Agent什么时候知道E发生了？"
  - event_time语义: T1（事件发生时间）
  - record_time语义: T1+N（信息到达时间）
  正确答案取决于查询的具体含义
```

## 关键约束
- 每个场景至少包含2个有明显时间gap的writes
- 必须设计至少2个在event_time语义和record_time语义下答案不同的查询
- QA对需要明确标注需要哪种时间推理
- 不能通过简单的"最新记录"策略得出正确答案

## QA生成指引
- Level 1: "事件X是什么时候发生的？" vs "Agent什么时候知道事件X的？"（双语义对比）
- Level 2: "在[T1,T2]期间，哪些事件已经发生但Agent还不知道？"（知识滞后检测）
- Level 3: "如果Agent不区分event_time和record_time，在回答以下查询时会犯什么错误？请列举3个具体例子。"
