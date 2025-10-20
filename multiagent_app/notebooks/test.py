# crew_gemini_test_fixed.py
from crewai import Agent, Task, Crew, LLM
import os

# -------------------------------
# 🔹 Step 1: Set your Gemini API Key
# -------------------------------
# ⚠️ Make sure this is your *real* key from Google AI Studio or Vertex AI Console
# Example format: AIzaSyXXXX-XXXXXXXXXXXXXXXXXXXXXXX
os.environ["GEMINI_API_KEY"] = "AIzaSyCeE1VRZw8mP2Rw0smVA2PuxjpQh_KyiXg"

# -------------------------------
# 🔹 Step 2: Initialize Gemini LLM properly
# -------------------------------
# CrewAI uses LiteLLM internally → expects provider prefix like "gemini/"
llm = LLM(
    model="gemini/gemini-1.5-flash",  # ✅ Correct model format
    api_key=os.getenv("GEMINI_API_KEY"),  # ✅ Explicitly pass it
    temperature=0.7,
    max_tokens=512
)

# -------------------------------
# 🔹 Step 3: Define the Agent
# -------------------------------
research_agent = Agent(
    role="Researcher",
    goal="Provide clear, structured insights on educational technology trends.",
    backstory="An experienced AI research analyst with deep understanding of modern education systems.",
    llm=llm,
    verbose=True
)

# -------------------------------
# 🔹 Step 4: Define the Task
# -------------------------------
research_task = Task(
    description="Summarize the impact of AI in modern education within 5 bullet points.",
    expected_output="A concise bullet list of AI's impacts in education.",
    agent=research_agent
)

# -------------------------------
# 🔹 Step 5: Define the Crew
# -------------------------------
crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    verbose=True
)

# -------------------------------
# 🔹 Step 6: Run the Crew
# -------------------------------
if __name__ == "__main__":
    print("🚀 Crew is kicking off!\n")
    final_output = crew.kickoff()
    print("\n✅ --- Final Report ---")
    print(final_output)
