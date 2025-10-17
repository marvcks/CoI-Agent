from agents import DeepResearchAgent,ReviewAgent,get_llms
import asyncio
import json
import argparse
import yaml
import os
import nest_asyncio
import logging
import os.path
import time
import dotenv
dotenv.load_dotenv()


# 配置日志
def setup_logging(save_file="saves/"):
    # 创建与agents.py中相同的run_xxx文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_file, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 日志文件路径
    log_file = os.path.join(save_dir, "coi_agent.log")
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    # 获取根日志记录器
    logger = logging.getLogger()
    return logger, save_dir

# 设置日志
logger, save_dir = setup_logging()

nest_asyncio.apply()
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
for key, value in config.items():
    if value == "":
        continue
    else:
        os.environ[key] = str(value)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--topic",type=str,help="research topic",default="Using diffusion to generate urban road layout map")
    argparser.add_argument("--anchor_paper_path",type=str,help="PDF path of the anchor paper",default= None)
    argparser.add_argument("--save_file",type=str,default="saves/",help="save file path")
    argparser.add_argument("--improve_cnt",type=int,default= 1,help="experiment refine count")
    argparser.add_argument("--max_chain_length t",type=int,default=5,help="max chain length")
    argparser.add_argument("--min_chain_length",type=int,default=3,help="min chain length")
    argparser.add_argument("--max_chain_numbers",type=int,default=1,help="max chain numbers")    
    args = argparser.parse_args()

    main_llm , cheap_llm = get_llms()

    topic = args.topic
    anchor_paper_path = args.anchor_paper_path

    review_agent = ReviewAgent(save_file=save_dir,llm=main_llm,cheap_llm=cheap_llm)
    args_dict = vars(args)
    args_dict.pop('save_file', None)  # Remove save_file from args to avoid duplicate
    deep_research_agent = DeepResearchAgent(save_file=save_dir, llm=main_llm, cheap_llm=cheap_llm, **args_dict)

    logger.info(f"begin to generate idea and experiment of topic {topic}")

    idea,related_experiments,entities,idea_chain,ideas,trend,future,human,year =  asyncio.run(
        deep_research_agent.generate_idea_with_chain(topic,anchor_paper_path)
        )
    experiment = asyncio.run(deep_research_agent.generate_experiment(idea,related_experiments,entities))

    for i in range(args.improve_cnt):
        experiment = asyncio.run(deep_research_agent.improve_experiment(review_agent,idea,experiment,entities))
        
    logger.info(f"succeed to generate idea and experiment of topic {topic}")

    res = {
        "idea":idea,
        "experiment":experiment,
        "related_experiments":related_experiments,
        "entities":entities,
        "idea_chain":idea_chain,
        "ideas":ideas,
        "trend":trend,
        "future":future,
        "year":year,
        "human":human
        }
    with open("result.json","w") as f:
        json.dump(res,f)