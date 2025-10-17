import os
import re
import logging
import argparse
import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Optional, TypedDict, List, Tuple, Dict, Union, Literal, Any
import subprocess
import sys
import json
import yaml
import time
import nest_asyncio
from dp.agent.server import CalculationMCPServer
import dotenv
dotenv.load_dotenv()

# Import the agents from the local module
from agents import DeepResearchAgent, ReviewAgent, get_llms

# 获取logger
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="CoI Agent Server")
    parser.add_argument('--port', type=int, default=50011, help='Server port (default: 50011)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50011
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
mcp = CalculationMCPServer("coi_agent_server", host=args.host, port=args.port)

logging.basicConfig(
    level=args.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Apply nest_asyncio for nested event loops
nest_asyncio.apply()

# def load_config():
#     """Load configuration from config.yaml"""
#     try:
#         with open('/Users/xhxu/Documents/CoI-Agent/config.yaml', 'r') as file:
#             config = yaml.safe_load(file)
#         for key, value in config.items():
#             if value == "":
#                 continue
#             else:
#                 os.environ[key] = str(value)
#         return True
#     except Exception as e:
#         logger.error(f"Failed to load config: {e}")
#         return False

# Setup logging function
def setup_logging(save_file="."):
    save_dir = save_file
    log_file = os.path.join(save_dir, "coi_agent.log")
    # Configure file handler for this specific run
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Get logger and add handler
    run_logger = logging.getLogger(f"coi_run")
    run_logger.addHandler(file_handler)
    run_logger.setLevel(logging.INFO)
    
    return run_logger, save_dir

@mcp.tool()
def run_coi_research(
    topic: str,
    anchor_paper_path: Optional[str] = None,
    save_file: str = ".",
    improve_cnt: int = 1,
    max_chain_length: int = 5,
    min_chain_length: int = 3,
    max_chain_numbers: int = 1
) -> Dict[str, Any]:
    """
    Run Chain of Ideas (CoI) research agent to generate research ideas and experiments.

    Args:
        topic: The research topic to investigate
        anchor_paper_path: Optional PDF path of the anchor paper
        save_file: Directory to save results (default: "saves/")
        improve_cnt: Number of experiment refinement iterations (default: 1)
        max_chain_length: Maximum chain length (default: 5)
        min_chain_length: Minimum chain length (default: 3)
        max_chain_numbers: Maximum number of chains (default: 1)

    Returns:
        A dictionary containing the research results including idea, experiment,
        related experiments, entities, idea chain, trends, and future directions.
    """
    try:
        # Load configuration
        # if not load_config():
        #     return {
        #         "status": "error",
        #         "message": "Failed to load configuration from config.yaml"
        #     }

        # Setup logging for this run
        run_logger, save_dir = setup_logging(save_file)
        
        # Get LLMs
        main_llm, cheap_llm = get_llms()
        
        # Initialize agents
        review_agent = ReviewAgent(save_file=save_dir, llm=main_llm, cheap_llm=cheap_llm)
        deep_research_agent = DeepResearchAgent(
            save_file=save_dir,
            llm=main_llm,
            cheap_llm=cheap_llm,
            topic=topic,
            anchor_paper_path=anchor_paper_path,
            improve_cnt=improve_cnt,
            max_chain_length=max_chain_length,
            min_chain_length=min_chain_length,
            max_chain_numbers=max_chain_numbers
        )

        run_logger.info(f"Starting CoI research for topic: {topic}")

        # Run the research process
        async def run_research():
            # Generate idea with chain
            idea, related_experiments, entities, idea_chain, ideas, trend, future, human, year = await deep_research_agent.generate_idea_with_chain(
                topic, anchor_paper_path
            )
            if not idea:
                return {
                    "status": "error",
                    "message": "Failed to generate idea with chain",
                    "topic": topic
                }
            
            # Generate experiment
            experiment = await deep_research_agent.generate_experiment(idea, related_experiments, entities)
            
            # Improve experiment iteratively
            for i in range(improve_cnt):
                experiment = await deep_research_agent.improve_experiment(review_agent, idea, experiment, entities)
            
            return {
                "idea": idea,
                "experiment": experiment,
                "related_experiments": related_experiments,
                "entities": entities,
                "idea_chain": idea_chain,
                "ideas": ideas,
                "trend": trend,
                "future": future,
                "year": year,
                "human": human
            }

        # Run the async research process
        result = asyncio.run(run_research())
        
        run_logger.info(f"Successfully completed CoI research for topic: {topic}")
        
        # Save results to JSON file
        result_file = os.path.join(save_dir, "result.json")
        with open(result_file, "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Add topic to result and return only the core research data
        result["topic"] = topic
        
        return result

    except Exception as e:
        error_msg = f"Error during CoI research: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "topic": topic
        }


if __name__ == "__main__":
    logging.info("Starting CoI Agent Server with research tools...")
    mcp.run(transport="sse")