# Manas 中文持续学习增强路线

本 fork 的第一阶段目标是让 Manas 在保持本地、CPU-only、无外部 ML 框架的前提下，更稳定地支持中文事实学习与中文问答。

## 当前观察到的问题

1. `manas teach` 原始解析逻辑主要面向英文 `is / are / means / refers to` 句式。
2. 中文 `问题：... 答案：...`、`...？：...`、`...是...` 等格式无法被稳定解析为 `input -> target`。
3. 中文无空格文本在 tokenizer 中容易被当成一个长词，导致相似度、泛化和解码都不稳定。
4. 连续 teach 多条事实时，新事实容易覆盖旧事实，例如 `cat` 在追加 `Eiffel Tower` 后会返回后者答案。
5. 当前 decoder 更像从词表里拼相关词，而不是稳定返回完整 target。

## 第一阶段改造目标

本阶段先做最小可验证改造：

- 支持中文 QA 训练格式：
  - `问题：山是怎么形成的？ 答案：山是由构造力或火山活动形成的。`
  - `山是怎么形成的？：山是由构造力或火山活动形成的。`
  - `山是怎么形成的？ : 山是由构造力或火山活动形成的。`
- 支持基础中文陈述事实：
  - `猫是家养哺乳动物。`
  - `长城是为保护古代中国免受入侵而建造的。`
- 支持中文无空格文本的字符 n-gram tokenization。
- teach 新事实时把对应神经路径及时固化，降低后续学习覆盖旧知识的概率。

## 非目标

本阶段不引入：

- 云端服务
- GPU 依赖
- 外部 ML 框架
- 大语言模型推理依赖
- 普通文本搜索式 RAG

## 验收样例

```bash
./target/release/manas reset

./target/release/manas teach "A cat is a small domesticated animal with fur"
./target/release/manas ask "cat"

./target/release/manas teach "The Eiffel Tower is located in Paris France."
./target/release/manas ask "cat"
./target/release/manas ask "Eiffel Tower"

./target/release/manas teach "问题：山是怎么形成的？ 答案：山是由构造力或火山活动经过漫长地质时期形成的。"
./target/release/manas ask "山是怎么形成的"
```
