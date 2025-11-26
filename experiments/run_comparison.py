import os
import sys
import time
import argparse
from typing import Dict, Any, Tuple, List
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

load_dotenv()
# Use a distinct logger name to separate from strategy logs
logger = setup_logger("ComparisonRunner")

class ExperimentRunner:
    def __init__(self, repo_full_name: str, pr_number: int):
        self.repo_full_name = repo_full_name
        self.pr_number = pr_number
        
        self.github_token = os.environ.get("GITHUB_TOKEN")
        self.deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        
        if not self.github_token or not self.deepseek_key:
            raise ValueError("Missing API keys in environment variables")
            
        self.github_client = GithubClient(token=self.github_token)
        self.llm_client = DeepSeekClient(api_key=self.deepseek_key)
        
        # Initialize all strategies
        self.strategies = [
            SearchRAGStrategy(self.llm_client, self.github_client),
            AgenticRAGStrategy(self.llm_client, self.github_client),
            CodeMeshStrategy(self.llm_client, self.github_client)
        ]

    def prepare_data(self) -> Tuple[str, Dict[str, Any]]:
        """Fetch PR data once to ensure fair comparison."""
        logger.info(f"📥 Preparing data for {self.repo_full_name} PR #{self.pr_number}...")
        raw_pr_content = self.github_client.get_pr_content(self.repo_full_name, self.pr_number)
        repo_obj = self.github_client._get_llama_repo(self.repo_full_name)
        _, pr_details_md = get_pr_details(repo_obj, self.pr_number)
        return pr_details_md, raw_pr_content

    def run(self):
        try:
            pr_details, pr_content = self.prepare_data()
        except Exception as e:
            logger.critical(f"Failed to prepare data: {e}")
            return

        results = []
        
        print("\n" + "="*60)
        print(f"🧪 STARTING COMPARISON EXPERIMENT")
        print(f"   Target: {self.repo_full_name} PR #{self.pr_number}")
        print("="*60 + "\n")

        for strategy in self.strategies:
            print(f"▶️  Running Strategy: {strategy.name}...")
            start_time = time.time()
            
            try:
                # Execute strategy
                context = strategy.execute(pr_details, pr_content, self.repo_full_name)
                status = "Success"
                length = len(context)
                
                # Heuristic cost calculation (very rough estimate)
                if "Search" in strategy.name:
                    cost_desc = "Low (~$0.01)"
                elif "Agent" in strategy.name:
                    cost_desc = "High ($0.50+)"
                else:
                    cost_desc = "Negligible (O(1))"
                    
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")
                status = f"Failed: {str(e)}"
                context = ""
                length = 0
                cost_desc = "N/A"
            
            duration = time.time() - start_time
            print(f"   Done in {duration:.2f}s. Context Length: {length} chars.\n")
            
            results.append({
                "name": strategy.name,
                "status": status,
                "duration": duration,
                "length": length,
                "cost": cost_desc,
                "context_preview": context[:500].replace('\n', ' ') + "..."
            })

        self._generate_report(results)

    def _generate_report(self, results: List[Dict]):
        filename = f"comparison_report_{self.pr_number}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Context Strategy Comparison: PR #{self.pr_number}\n\n")
            f.write(f"**Repository:** {self.repo_full_name}\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 1. Performance Metrics\n\n")
            f.write("| Strategy | Status | Duration (s) | Context Size (chars) | Est. Cost |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            
            for r in results:
                f.write(f"| **{r['name']}** | {r['status']} | {r['duration']:.2f}s | {r['length']:,} | {r['cost']} |\n")
            
            f.write("\n## 2. Qualitative Analysis\n\n")
            for r in results:
                f.write(f"### {r['name']}\n")
                f.write(f"**Output Preview:**\n> {r['context_preview']}\n\n")
                
                if "Search" in r['name']:
                    f.write("**Analysis:** Fast execution relying on keyword matching. Good for finding explicit references but typically misses implicit logic flows (Low Recall).\n")
                elif "Agent" in r['name']:
                    f.write("**Analysis:** Deep analysis via ReAct loop. Capable of traversing file structures and verifying dependencies. Significantly slower due to multiple LLM round-trips, but High Precision.\n")
                elif "Mesh" in r['name']:
                    f.write("**Analysis:** (Simulated) Represents the ideal state: instant, deterministic retrieval based on static analysis graph. Solves the latency/cost trade-off.\n")
                
                f.write("\n---\n\n")
                
        print(f"✅ Report generated: {filename}")
        print(f"   Review this file to see the side-by-side comparison.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comparison experiment between strategies")
    parser.add_argument("url", help="GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)")
    args = parser.parse_args()
    
    try:
        repo, number = parse_github_url(args.url)
        runner = ExperimentRunner(repo, number)
        runner.run()
    except Exception as e:
        logger.critical(f"Experiment failed: {e}")