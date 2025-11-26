import argparse
import os
import sys
import time
from dotenv import load_dotenv

# Add project root to path to ensure imports work correctly when running from any directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import (
    GithubClient, 
    DeepSeekClient, 
    setup_logger, 
    get_pr_details,
    parse_github_url
)
from strategies import (
    SearchRAGStrategy, 
    AgenticRAGStrategy, 
    CodeMeshStrategy
)

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = setup_logger()

def main():
    # Configure CLI Argument Parser
    parser = argparse.ArgumentParser(
        description="LlamaPReview Context Intelligence Research CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Example:\n  python main.py https://github.com/psf/requests/pull/6666 --strategy agent"
    )
    
    parser.add_argument(
        "url", 
        help="GitHub Pull Request URL (e.g., https://github.com/owner/repo/pull/123)"
    )
    
    parser.add_argument(
        "--strategy", 
        choices=["search", "agent", "mesh"], 
        default="search",
        help="Context retrieval strategy to use:\n"
             "  search:  Fast, keyword-based retrieval (Solution 2)\n"
             "  agent:   Deep reasoning ReAct loop (Solution 3)\n"
             "  mesh:    Preview of the deterministic Graph Engine (Solution 4)"
    )
    
    parser.add_argument(
        "--output", 
        help="Output file path for the generated context (default: context_<strategy>_<pr>.md)"
    )

    args = parser.parse_args()

    # 1. Environment Validation
    github_token = os.environ.get("GITHUB_TOKEN")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")

    if not github_token:
        logger.critical("❌ GITHUB_TOKEN not found in environment variables. Please check your .env file.")
        sys.exit(1)
    
    if not deepseek_key:
        logger.critical("❌ DEEPSEEK_API_KEY not found in environment variables. Please check your .env file.")
        sys.exit(1)

    # 2. Parse Arguments
    try:
        repo_full_name, pr_number = parse_github_url(args.url)
    except ValueError as e:
        logger.critical(str(e))
        sys.exit(1)

    # 3. Initialization
    logger.info(f"🔌 Initializing clients for {repo_full_name} PR #{pr_number}...")
    try:
        github_client = GithubClient(token=github_token)
        llm_client = DeepSeekClient(api_key=deepseek_key)
    except Exception as e:
        logger.critical(f"Client initialization failed: {e}")
        sys.exit(1)

    # 4. Fetch PR Data
    logger.info("📥 Fetching Pull Request metadata...")
    try:
        # Note: This uses the llama_github wrapper internally via GithubClient
        raw_pr_content = github_client.get_pr_content(repo_full_name, pr_number)
        
        # We need the repo object for get_pr_details. 
        # Accessing protected member _get_llama_repo is acceptable here as main.py acts as a controller.
        repo_obj = github_client._get_llama_repo(repo_full_name)
        _, pr_details_md = get_pr_details(repo_obj, pr_number)
        
        logger.info(f"   PR Title: {raw_pr_content.get('pr_metadata', {}).get('title', 'Unknown')}")
    except Exception as e:
        logger.critical(f"Failed to fetch PR data: {e}")
        sys.exit(1)

    # 5. Select Strategy
    strategies = {
        "search": SearchRAGStrategy(llm_client, github_client),
        "agent": AgenticRAGStrategy(llm_client, github_client),
        "mesh": CodeMeshStrategy(llm_client, github_client)
    }
    
    strategy = strategies[args.strategy]
    logger.info(f"🧠 Selected Strategy: {strategy.name}")

    # 6. Execute Strategy
    start_time = time.time()
    try:
        context = strategy.execute(
            pr_details=pr_details_md,
            pr_content=raw_pr_content,
            repo_full_name=repo_full_name
        )
    except KeyboardInterrupt:
        logger.warning("\n🛑 Execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Strategy execution failed: {e}", exc_info=True)
        sys.exit(1)
    
    duration = time.time() - start_time

    # 7. Output Results
    output_file = args.output or f"context_{args.strategy}_{pr_number}.md"
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(context)
        
        logger.info(f"✅ Success! Context generated in {duration:.2f}s")
        logger.info(f"📄 Output saved to: {output_file}")
        
        # Print a preview
        print("\n" + "="*50)
        print(f"CONTEXT PREVIEW ({len(context)} chars)")
        print("="*50)
        preview_len = 1000
        print(context[:preview_len] + ("...\n[Content Truncated]" if len(context) > preview_len else ""))
        print("="*50 + "\n")
        
    except IOError as e:
        logger.error(f"Failed to write output file: {e}")

if __name__ == "__main__":
    main()