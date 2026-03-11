✅ 第一周核心目标 (OKRs)
1、掌握 LangChain 的核心架构，能够使用 LangChain 稳定连接并调用大语言模型（LLM）。
2、熟练掌握 PromptTemplate（提示词模板）与 LCEL（LangChain 表达式语言），能够构建模块化的数据处理链条。

📅 日程安排
Day 1-2:
目标：LangChain 环境配置与模型实例化

任务 1：
描述：在你的 Python 环境中安装 langchain、langchain-openai（或其他你选定的大模型供应商对应的包）。配置好 API Key 环境变量，确保开发环境安全规范。

任务 2：
描述：编写代码，使用 LangChain 的 ChatOpenAI（或对应模型的类）实例化大模型。对比使用 LangChain 调用模型与你以前可能用过的原生 API 调用方式，体会 SystemMessage、HumanMessage 等消息类的规范化封装。

Day 3-4:
目标：Prompt 模板化与 LCEL 链式调用

任务 3：
描述：学习 PromptTemplate 和 ChatPromptTemplate。在企业应用中，提示词不是写死的字符串，而是需要动态填入变量的模板。尝试创建一个用于“电商用户评论情感分析”的模板，并动态传入不同的评论变量。

任务 4：
描述：掌握 LangChain 最核心的 LCEL（LangChain 表达式语言）。使用 | 管道符，将你写好的 PromptTemplate 和实例化的 LLM 连接起来，形成一个基础的 Chain（链），并运行它获取结果。

Day 5-7:
目标：结合业务场景引入 Output Parser（输出解析器）

任务 5：
描述：在数据分析中，我们需要大模型输出结构化的数据（如 JSON），而不是长篇大论。学习 LangChain 的 PydanticOutputParser 或 JsonOutputParser，定义你期望的数据结构。

任务 6：
描述：将 Output Parser 整合进你的 LCEL Chain 中（Prompt | LLM | Parser）。测试传入一段非结构化的业务文本，让大模型直接返回可以直接被 Python 字典或 Pandas DataFrame 解析的格式。

💻 本周实战作业
描述：
开发一个**“自动化品类趋势洞察抽取器”**。回想你在字节跳动做品类趋势跟踪和用户分析的工作 ，业务团队经常会收到大量无结构的行业新闻、竞品动态和消费者零散反馈。请利用 LangChain 开发一个处理脚本。

功能要求：

准备 5-10 段模拟的“行业趋势新闻”或“消费者偏好长文本”。

使用 LangChain 构建一条完整的处理链条（Chain）。

针对每段文本，AI 必须自动提取出：品类名称（字符串）、市场趋势倾向（上升/下降/平稳）、核心消费者痛点（列表格式）、流失风险预警（布尔值 True/False）。

技术要求：

必须使用 LangChain 框架，严禁使用原生 API 裸写请求。

必须使用 ChatPromptTemplate 动态传入文本。

必须使用 Output Parser，确保最终打印或保存的结果是严格的 JSON 格式或 Python 字典形式，不能包含任何 markdown 标记或多余的自然语言。

使用 LCEL 语法（chain = prompt | model | parser）完成串联。

周三和周日晚上我会看你的实战进度。掌握了 LangChain 的链式思维，后续我们将能轻松引入你在凯捷和字节做过的数据源。遇到卡点随时找我，祝本周编码顺利！
