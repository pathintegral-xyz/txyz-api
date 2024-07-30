"""Example code for integrating txyz search with langchain.

Please follow the doc https://platform.txyz.ai to generate
txyz api key. Assign the api key to the environment variable TXYZ_API_KEY.

Example command:

Search with explanation:

python samples/python/langchain_txyz_search.py -q 'large language model'

Search with raw results:

python samples/python/langchain_txyz_search.py -q 'large language model' -s
"""
import argparse
import asyncio
from typing import Any

import requests

from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env


_API_BASE_ENDPOINT = "https://api.txyz.ai/v1"

# Uncomment the following for api key assignment, alternatively,
# set the environment variable "TXYZ_API_KEY" in your environment
# import os
# os.environ["TXYZ_API_KEY"] = ""


class TxyzClient:
    def __init__(self, api_key: str, api_base_endpoint: str):
        self._api_key = api_key
        self._api_base_url =  api_base_endpoint

    def search_fn(self, query: str, max_num_results: int, search_type: str) -> dict:
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
            url = f"{self._api_base_url}/search/scholar"
        else:
            url = f"{self._api_base_url}/search/web"

        headers = {"Authorization": f"Bearer {self._api_key}"}
        params = {
            "query": query,
            "max_num_results": max_num_results
        }

        response = requests.post(url, headers=headers, params=params)
        if response.status_code != 200:
            raise ValueError(f"Failed to get search results: {response.text}")

        response_json = response.json()
        return response_json


    def explain_fn(self, id: str) -> dict:
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
            "Authorization": f"Bearer {self._api_key}"
        }

        explaination = requests.post(f"{self._api_base_url}/search/explain", json=payload, headers=headers)
        if explaination.status_code != 200:
            raise ValueError(f"Failed to get explanation: {explaination.text}")
        return explaination.json()


class TxyzSearch(BaseModel):
    client: Any

    @root_validator(pre=True)
    def validate_environment(cls, values: dict) -> dict:
        """Validate that api key exists in environment."""
        txyz_api_key = get_from_dict_or_env(values, "txyz_api_key", "TXYZ_API_KEY")

        values['client'] = TxyzClient(txyz_api_key, _API_BASE_ENDPOINT)

        return values

    def _run_helper(self, query: str, max_num_results: int, search_type: str, skip_explain: bool) -> dict:
        """Helper function to run the search"""
        search_results = self.client.search_fn(query, max_num_results, search_type)
        response = {'search_results': search_results}
        if not skip_explain:
            search_id = search_results["id"]
            explanation = self.client.explain_fn(search_id)
            response['explanation'] = explanation
        return response

    def run(self, query: str, max_num_results: int, search_type: str, skip_explain: bool) -> dict:
        """Run the search"""
        return self._run_helper(query, max_num_results, search_type, skip_explain)

    async def arun(self, query: str, max_num_results: int, search_type: str, skip_explain: bool) -> dict:
        """Run the search asynchronously"""
        return self._run_helper(query, max_num_results, search_type, skip_explain)


async def run_wrapper(query: str, max_num_results: int, search_type: str, skip_explain: bool) -> dict:
    """Run the search"""
    txyz_search = TxyzSearch()
    response = await txyz_search.arun(query, max_num_results, search_type, skip_explain)
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

    txyz_llm_search_results = asyncio.run(run_wrapper(
        args.query, args.max_num_results,
        args.search_type, args.skip_explain))
    print(txyz_llm_search_results)
