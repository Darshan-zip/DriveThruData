from core.orchestrator import RAGOrchestrator

def main():
    orchestrator = RAGOrchestrator()
    print("--- Multi-RAG System Integrated ---")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nQuery: ")
        if user_input.lower() == 'exit':
            break
        
        try:
            response = orchestrator.route_query(user_input)
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
