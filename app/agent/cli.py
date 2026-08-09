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

            response_stream, new_messages = dp_agent._invoke(
                prompt, model="qwen/qwen3.6-27b", history=history, max_tokens=2000, stream=True, tools=tools , cache=True
            )

            for chunk in response_stream:
                print(chunk, end="", flush=True)

            # `history` is ours — `_invoke` never touches it. `new_messages`
            # is only fully populated once the stream above is drained
            # (the final assistant turn is appended as the generator closes).
            history.extend(new_messages)

            print("\n")
        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            break
        except Exception as e:
            print("\nError:", e, "\n")


if __name__ == "__main__":
    main()
