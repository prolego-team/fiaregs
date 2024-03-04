import os
from typing import Callable
from pathlib import Path
import logging
import sys
import time

from datasets import Dataset
import ragas
from langchain_openai.chat_models import ChatOpenAI

from fiaregs import drivers, custom_eval_metrics
from fiaregs import openaiapi as openai

# Suppress a runtime warning re: tokenizer parallelism and multiple threads.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Basic logger setup ===================================================
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logging.getLogger('search').setLevel(logging.DEBUG)
log = logging.getLogger('search')
# ===================================================================

DOC_DIR = Path('data/docs')
DATA_DIR = Path('data')

RERANK = True
PRE_EXPAND = False
POST_EXPAND = True

REGS = {
    '2023 FIA Formula One Sporting Regulations': 'fia_2023_formula_1_sporting_regulations_-_issue_6_-_2023-08-31.yaml',
    '2023 FIA International Sporting Code': '2023_international_sporting_code_fr-en_clean_9.01.2023.yaml',
    '2023 FIA International Sporting Code, Appendix L, Chapter II': 'appendix_l_iii_2023_publie_le_20_juin_2023.yaml',
    '2023 FIA International Sporting Code, Appendix L, Chapter IV': 'appendix_l_iv_2023_publie_le_20_juin_2023.yaml',
    '2023 FIA Formula One Financial Regulations': 'fia_formula_1_financial_regulations_-_issue_16_-_2023-08-31.yaml',
    '2023 FIA Formula One Technical Regulations': 'fia_2023_formula_1_technical_regulations_-_issue_7_-_2023-08-31.yaml'
}

MAX_LLM_CALLS_PER_INTERACTION = 5

DEF_DIVIDER = '\n\n'


def evaluate(eval_set: Dataset):
    gpt4 = ChatOpenAI(model_name="gpt-4")
    eval = ragas.evaluate(
        eval_set,
        metrics = [
            # ragas.metrics.answer_relevancy,
            # ragas.metrics.faithfulness,
            # custom_eval_metrics.answer_correctness,
            custom_eval_metrics.binary_answer_correctness
        ],
        llm=gpt4
    )

    return eval


def generate_responses(eval_set: Dataset, search: Callable) -> Dataset:

    wait_time = 0
    responses = []
    for query in eval_set['question']:
        responses.append(search(query))
        time.sleep(wait_time)

    return eval_set.add_column('answer', responses)


def main():

    model_name = 'all-mpnet-base-v2'
    cross_encoder_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    # cross_encoder_name = None
    # OpenAI
    # llm_model_name = 'gpt-4-0125-preview'
    llm_model_name = 'gpt-3.5-turbo-0125'
    llm_api_key = 'OPENAI_API_KEY'

    # Perplexity
    # llm_model_name = 'mistral-7b-instruct'
    # llm_model_name = 'llama-2-70b-chat'
    # llm_model_name = 'mixtral-8x7b-instruct'
    # llm_api_key = 'PERPLEXITY_API_KEY'

    use_definitions = False
    top_k = 10
    n_runs = 3

    if 'OPENAI' in llm_api_key:
        api_client = openai.get_openaiai_client(
            os.environ[llm_api_key],
        )
    elif 'PERPLEXITY' in llm_api_key:
        api_client = openai.get_openaiai_client(
            os.environ[llm_api_key],
            base_url='https://api.perplexity.ai'
        )
    else:
        print('Unrecognized API')
        exit()

    llm_model = openai.start_chat(llm_model_name, api_client)

    # search = drivers.driver_llm_only(llm_model)
    search = drivers.driver_llm_with_search(
        llm_model,
        DATA_DIR,
        DOC_DIR,
        REGS,
        PRE_EXPAND,
        POST_EXPAND,
        model_name,
        cross_encoder_name,
        top_k,
        include_definitions=use_definitions
    )
    # search = drivers.driver_llm_with_agentic_search(
    #     llm_model,
    #     DATA_DIR,
    #     DOC_DIR,
    #     REGS,
    #     PRE_EXPAND,
    #     POST_EXPAND,
    #     model_name,
    #     cross_encoder_name,
    #     top_k,
    #     include_definitions=use_definitions
    # )

    eval_set = Dataset.from_json('data/eval_set.json', field='eval_set')
    # eval_set = Dataset.from_dict(eval_set[:1])
    print(eval_set)

    evals = []
    for i in range(n_runs):
        if 'answer' in eval_set.column_names:
            eval_set = eval_set.remove_columns('answer')
        eval_set = generate_responses(eval_set, search)
        eval = evaluate(eval_set)
        evals.append(eval)
        print(eval)
        eval.to_pandas().to_csv(f'eval_{i+1}.csv')

    metrics = evals[0].keys()
    avg_metrics = {k:0 for k in metrics}
    for eval in evals:
        for metric in metrics:
            avg_metrics[metric] += eval[metric]/n_runs
    print(avg_metrics)


if __name__=='__main__':
    main()