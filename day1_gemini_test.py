import os
from dotenv import load_dotenv
# 注意看这里！我们引入的是 google 的包，而不是 openai 的包了
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# 1. 让保安队长加载 .env 里的密码 (它会自动把 GOOGLE_API_KEY 塞进内存)
load_dotenv()

# 2. 实例化 Gemini 大模型 (Gemini 1.5 Flash 速度极快，适合数据处理)
# 注意：你不需要传 api_key 参数，因为底层代码会自动去系统环境变量里找 GOOGLE_API_KEY
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0  # 做数据抽取任务时，把温度设为0，让它少发散，更严谨
)

# 3. 构造标准化消息
messages = [
    SystemMessage(content="你是一个资深的数据分析师，擅长从业务视角解读信息。"),
    HumanMessage(content="用一句话解释什么是LangChain？")
]

# 4. 调用模型并打印结果
print("正在呼叫 Gemini，请稍候...\n")
response = llm.invoke(messages)
print("大模型回复：", response.content)