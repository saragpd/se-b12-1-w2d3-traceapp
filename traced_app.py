
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def main():
    # Connect to OpenAI through LangChain
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompts = [
        """
        Explain Generative AI to a beginner.
        Give its definition, three real-world applications,
        and one limitation. Keep the answer under 100 words.
        """,
        """
        Explain the difference between AI agents and Agentic AI.
        Compare their autonomy, decision-making, tool usage,
        and ability to complete multi-step tasks. Give examples.
        """,
        """
        Design a detailed Agentic AI workflow for an IT support
        assistant. Explain how it receives a request, plans steps,
        uses tools, checks results, handles failures, and escalates
        to a human. Include a practical example.
        """
    ]

    for number, prompt in enumerate(prompts, start=1):
        print(f"\n{'=' * 60}")
        print(f"PROMPT {number}")
        print(f"{'=' * 60}")

        response = model.invoke(prompt)

        print("\nANSWER:")
        print(response.content)

if __name__ == "__main__":
    main()
