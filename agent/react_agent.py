from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize,get_weather,get_user_location,
                                     get_user_id,get_current_month,fetch_external_data,fill_context_for_report)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch

class ReactAgent:
    def __init__(self):
        #会话记忆：按 thread_id 分区保存每场对话的完整状态（含工具调用观察）
        self.memory = MemorySaver()

        self.agent = create_agent(
            model = chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize,get_weather,get_user_location,get_user_id,
                   get_current_month,fetch_external_data,fill_context_for_report],
            middleware=[monitor_tool,log_before_model,report_prompt_switch],
            checkpointer=self.memory
        )


    def execute_stream(self, query: str, thread_id: str = "default"):
        #checkpointer 会自动把历史状态挂到同一个 thread_id 下，
        #所以每次只需要传入"本轮新消息"，模型自然带着全部上下文
        input_dict = {
            "messages":[
                {"role": "user","content": query}
            ]
        }

        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self.agent.stream(input_dict,stream_mode="values",config=config,context={"report":False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()
    thread_id = "demo-001"

    print("========== 第 1 轮 ==========")
    for chunk in agent.execute_stream("小户型适合哪种扫地机器人？", thread_id=thread_id):
        print(chunk,end="",flush=True)

    print("\n========== 第 2 轮（追问） ==========")
    for chunk in agent.execute_stream("你刚才推荐的那种机器人日常该怎么保养？", thread_id=thread_id):
        print(chunk,end="",flush=True)