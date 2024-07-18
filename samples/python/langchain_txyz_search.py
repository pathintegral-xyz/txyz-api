import argparse
import asyncio
from typing import Type, Optional

import requests

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool


_API_BASE_URL = "https://api.txyz.ai/v1"
_API_KEY = "your_api_key_here"


def search_fn(query, max_num_results, search_type):
    """Seach function

    Args:
        query (str): The query to search for
        max_num_results (int): The maximum number of results to return
        search_type (str): The type of search to perform, it can be either 'scholar' or 'web'

    Returns:
        dict: The search results

    Raises:
        ValueError: If search_type is not either 'scholar' or 'web'
        ValueError: If the search fails
    """
    if search_type not in {"scholar", "web"}:
        raise ValueError("search_type must be either 'scholar' or 'web'")

    if search_type == "scholar":
        url = f"{_API_BASE_URL}/search/scholar"
    else:
        url = f"{_API_BASE_URL}/search/web"

    headers = {"Authorization": f"Bearer {_API_KEY}"}
    params = {
        "query": query,
        "max_num_results": max_num_results
    }

    response = requests.post(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(f"Failed to get search results: {response.text}")

    response_json = response.json()
    return response_json


def explain_fn(id):
    """Get explanation of the search results

    Args:
        id (str): The search id

    Returns:
        dict: The explanation of the search results

    Raises:
        ValueError: If the explanation fails
    """
    payload = {
        "search_id": id,
        "response_mode": "NON_STREAMING"
    }

    headers = {
        "Authorization": f"Bearer {_API_KEY}"
    }

    explaination = requests.post(f"{_API_BASE_URL}/search/explain", json=payload, headers=headers)
    if explaination.status_code != 200:
        raise ValueError(f"Failed to get explanation: {explaination.text}")
    return explaination.json()


class TxyzSearchInput(BaseModel):
    query: str = Field(description="The query to search for")
    max_num_results: int = Field(description="The maximum number of results to return")
    search_type: str = Field(description="The type of search to perform, it can be either 'scholar' or 'web'")
    skip_explain: bool = Field(description="Whether to return an explanation of the search results")


class TxyzSearch(BaseTool):
    name = "txyz_search"
    description = "useful for when you need search and explain something recent and professional."
    arg_schema: Type[BaseModel] = TxyzSearchInput

    def _run_helper(self, query, max_num_results, search_type, skip_explain):
        search_results = search_fn(query, max_num_results, search_type)
        response = {'search_results': search_results}
        if not skip_explain:
            search_id = search_results["id"]
            explanation = explain_fn(search_id)
            response['explanation'] = explanation
        return response

    def _run(self, query, max_num_results, search_type, skip_explain,
             run_manager: Optional[CallbackManagerForToolRun] = None):
        return self._run_helper(query, max_num_results, search_type, skip_explain)

    async def _arun(self, query, max_num_results, skip_explain, search_type,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None):
        return self._run_helper(query, max_num_results, search_type, skip_explain)


async def run(query, max_num_results, search_type, skip_explain):
    txyz_search = TxyzSearch()
    query = "Large language models"
    response = await txyz_search.arun({
        "query": query, "max_num_results": max_num_results, "search_type": search_type, "skip_explain": skip_explain})
    return response

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-q", "--query", type=str, help="The query to search for")
    arg_parser.add_argument(
        "-n",
        "--max_num_results",
        type=int,
        default=10,
        help="The maximum number of search results to return")
    arg_parser.add_argument(
        "-t",
        "--search_type",
        type=str,
        default="scholar",
        choices=["scholar", "web"],
        help="The type of search to perform, it can be either 'scholar' or 'web'")
    arg_parser.add_argument(
        "-s",
        "--skip_explain",
        action="store_true",
        help="Whether to skip generating explanations for the search results")

    args = arg_parser.parse_args()

    txyz_llm_search_results = asyncio.run(run(
        args.query, args.max_num_results,
        args.search_type, args.skip_explain))
    print(txyz_llm_search_results)
