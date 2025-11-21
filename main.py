import argparse
import os
from dotenv import load_dotenv
from src.generators import GeminiGenerator, OpenAIGenerator, FalGenerator
from src.editors import GeminiEditor, OpenAIEditor, FalEditor
from src.analyzers import QwenAnalyzer, GeminiAnalyzer
from src.evaluator import EvaluationLoop

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Image Generation and Analysis Evaluation")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--count", type=int, required=True, help="Target count of objects")
    parser.add_argument("--object", type=str, help="Specific object name to count (overrides extraction from prompt)")
    parser.add_argument("--generator", type=str, default="gemini", choices=["gemini", "openai", "fal"], help="Generator model")
    parser.add_argument("--editor", type=str, default="openai", choices=["gemini", "openai", "fal"], help="Editor model")
    parser.add_argument("--analyzer", type=str, default="qwen", choices=["qwen", "gemini"], help="Analyzer model")
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "loop"], help="Evaluation mode")
    
    args = parser.parse_args()
    
    # Initialize Generator
    if args.generator == "gemini":
        generator = GeminiGenerator()
    elif args.generator == "openai":
        generator = OpenAIGenerator()
    elif args.generator == "fal":
        generator = FalGenerator() # Default Recraft
        
    # Initialize Editor
    if args.editor == "gemini":
        print("Warning: Gemini Editor is not fully implemented yet. Using OpenAI Editor as fallback.")
        editor = OpenAIEditor()
    elif args.editor == "openai":
        editor = OpenAIEditor()
    elif args.editor == "fal":
        editor = FalEditor()
        
    # Initialize Analyzer
    if args.analyzer == "qwen":
        analyzer = QwenAnalyzer()
    elif args.analyzer == "gemini":
        analyzer = GeminiAnalyzer()
        
    loop = EvaluationLoop(generator, analyzer, editor if args.mode == "loop" else None)
    try:
        result = loop.run(args.prompt, args.count, args.mode, args.object)
        
        print("\n--- Result ---")
        print(f"Target Count: {result['target_count']}")
        print(f"Detected Count: {result['detected_count']}")
        print(f"Match: {result['match']}")
        print(f"Final Image: {result['image_path']}")
        print("Steps:")
        for step in result['steps']:
            print(f"  - {step['action']}: {step['count']}")
            
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
