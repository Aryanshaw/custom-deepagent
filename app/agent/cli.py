from app.agent.deep_agent import DeepAgent
from app.agent.tools.testing_tool import add_two_numbers


def main():
    dp_agent = DeepAgent("groq")
    print("Agent started. Type 'exit' to quit.\n")
    history = []

    while True:
        try:
            prompt = input("You: ")

            if prompt == "exit":
                break

            if not prompt.strip():
                continue

            print("\nDeepAgent: ", end="", flush=True)

            tools = [add_two_numbers]  # register tools here

            response_stream = dp_agent._invoke(prompt, model="llama-3.3-70b-versatile", history=history, max_tokens=2000, stream=True, tools=tools)

            # Collect the complete response while printing chunks
            response_parts = []

            for chunk in response_stream:
                print(chunk, end="", flush=True)
                response_parts.append(chunk)

            response = "".join(response_parts)

            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": response})

            print("\n")
        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            break
        except Exception as e:
            print("\nError:", e, "\n")


if __name__ == "__main__":
    main()
