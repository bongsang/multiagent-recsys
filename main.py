import os
import re
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent


class LLMRecSys:
    def __init__(self, api_key: str, api_model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = api_model
        self.df_review = None
        self.df_meta = None

    def load_data(self) -> None:
        review_data_file = BASE_DIR / "dataset" / "All_Beauty.jsonl"
        meta_data_file = BASE_DIR / "dataset" / "meta_All_Beauty.jsonl"

        logging.info("Loading data...")
        self.df_review = pd.read_json(review_data_file, lines=True)
        self.df_meta = pd.read_json(meta_data_file, lines=True)
        logging.info("Data loaded successfully.")

    def get_users(self, num_tests: int) -> list:
        unique_users = self.df_review["user_id"].drop_duplicates()
        return unique_users.sample(num_tests).tolist()

    def get_user_data(self, users: list) -> pd.DataFrame:
        df_data = self.df_review[self.df_review["user_id"].isin(users)]
        df_data = pd.merge(
            df_data, self.df_meta, on="parent_asin", suffixes=("", "_meta")
        )
        return df_data

    def get_single_agent_recommendation(self, user_prompt: str):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides product recommendations.",
            },
            {"role": "user", "content": user_prompt},
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            agent_response = completion.choices[0].message.content
            return agent_response
        except Exception as e:
            logging.error(f"OpenAI API Error: {e}")
            return None

    def get_multi_agent_recommendation(self, user_prompt: str, num_agents: int = 3):
        agent_responses = []
        logging.info("Starting multi-agent recommendation.")

        for i in range(num_agents):
            agent_prompt = f"Agent {i+1}: {user_prompt}"
            agent_response = self.get_single_agent_recommendation(agent_prompt)
            if agent_response:
                agent_responses.append(agent_response)
            else:
                logging.error(f"No response from Agent {i+1}")

        if not agent_responses:
            logging.error("No agent responses received.")
            return None

        debate_prompt = get_debate_prompt(agent_responses)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that provides product recommendations. "
                    "Your task is to provide a product rating and reasoning based on the user's "
                    "purchase history and the new item information."
                ),
            },
            {"role": "user", "content": debate_prompt},
        ]

        try:
            final_response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            final_agent_response = final_response.choices[0].message.content
            return final_agent_response
        except Exception as e:
            logging.error(f"API Error: {e}")
            return None


def get_user_prompt(df_data: pd.DataFrame, user_id: str) -> str:
    prompt = f"\n### User Purchase History of {user_id} ###\n"
    user_data = df_data[df_data["user_id"] == user_id]
    purchase_history = user_data.iloc[:-1]  # Exclude the last item
    last_item = user_data.iloc[-1]  # The item to predict

    for _, row in purchase_history.iterrows():
        prompt += f"\nProduct Title: {row.get('title_meta', '')}\n"
        prompt += f"Brand: {row.get('brand', '')}\n"
        prompt += f"Categories: {row.get('categories', '')}\n"
        prompt += f"Description: {row.get('description', '')}\n"
        prompt += f"Item Price: {row.get('price', '')}\n"
        prompt += f"User Rating: {row.get('rating', '')}\n"
        prompt += f"User Review Title: {row.get('title', '')}\n"
        prompt += f"User Review: {row.get('text', '')}\n"

    prompt += f"\n### New Item Information: ###\n"
    prompt += f"New Product Title: {last_item.get('title_meta', '')}\n"
    prompt += f"Brand: {last_item.get('brand', '')}\n"
    prompt += f"Categories: {last_item.get('categories', '')}\n"
    prompt += f"Description: {last_item.get('description', '')}\n"
    prompt += f"Item Price: {last_item.get('price', '')}\n"
    prompt += "\n######\n"

    prompt += (
        "Given the user's past purchase history and the new item information, "
        "what can you infer about the user's preferences, and how they will rate the new product?\n\n"
        "Your reasoning explanation should be based on any commonalities in the user history items and inferred user tastes or preferences.\n\n"
        "After your reasoning, predict a numerical rating.\n\n"
        "Please follow the format below:\n"
        "### Reason: ###\n"
        "Write your reasoning explanation here. You can have line breaks.\n\n"
        "### Rating: ###\n"
        "Give a single numerical rating, e.g., 1-5.\n"
    )
    return prompt


def get_debate_prompt(other_agents_answers: list) -> str:
    prompt = "These are the recent opinions from other agents:\n"
    for answer in other_agents_answers:
        prompt += f"\n### One agent response: ###\n{answer}"
    prompt += (
        "\nUse these opinions carefully as additional advice. Can you provide an updated answer? "
        "Make sure to state your answer at the end of the response.\n\n"
        "Please follow the format below:\n"
        "### Reason: ###\n"
        "Write your reasoning explanation here. You can have line breaks.\n\n"
        "### Rating: ###\n"
        "Give a single numerical rating, e.g., 1-5.\n"
    )
    return prompt


def get_rating(response: str) -> float:
    # Extract the numerical rating from the assistant's response.
    match = re.search(r"###\s*Rating:\s*###\s*(\d+(\.\d+)?)", response, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("Rating not found in the response.")


def get_reasoning(response: str) -> str:
    # Extract the reasoning explanation from the assistant's response.
    match = re.search(
        r"###\s*Reason:\s*###\s*(.*?)\s*###\s*Rating:\s*###",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Reasoning not found in the response.")


if __name__ == "__main__":
    print("Starting the recommendation system.")

    # RecSys Temporary API Key
    OPENAI_API_KEY = None
    assert OPENAI_API_KEY, "Please provide your OpenAI API key."

    num_tests = 20
    api_models = [
        "gpt-4o-mini",  # default model
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ]
    recsys = LLMRecSys(
        api_key=OPENAI_API_KEY,
        api_model=api_models[0],  # Using "gpt-4o-mini"
    )
    recsys.load_data()

    users = recsys.get_users(num_tests)
    df_data = recsys.get_user_data(users)

    real_ratings = []
    single_agent_ratings = []
    multi_agent_ratings = []

    for i, user in enumerate(users):
        start = time.time()
        user_prompt = get_user_prompt(df_data, user)

        # Get the real rating for the last item
        user_data = df_data[df_data["user_id"] == user]
        real_rating = user_data.iloc[-1]["rating"]
        real_ratings.append(real_rating)

        # Single Agent(Zero-shot) Recommendation
        single_response = recsys.get_single_agent_recommendation(user_prompt)
        if single_response:
            try:
                predicted_single = get_rating(single_response)
                reason_single = get_reasoning(single_response)
                print("=" * 200)
                print(
                    f"Single Agent predicted rating: {predicted_single} (ground truth: {real_rating})"
                )
                print(f"Single Agent reasoning:\n{reason_single.strip()}")
                print("_" * 200)
            except ValueError as e:
                logging.error(f"Single agent error for user {user}: {e}")
        else:
            logging.error(f"No response received from single agent for user {user}")

        # Multi-Agent Recommendation
        multi_response = recsys.get_multi_agent_recommendation(user_prompt)
        if multi_response:
            try:
                predicted_multi = get_rating(multi_response)
                reason_multi = get_reasoning(multi_response)
                print("=" * 200)
                print(
                    f"Multi-Agent predicted rating:{predicted_multi} (ground truth: {real_rating})"
                )
                print(f"Multi-Agent reasoning:\n{reason_multi.strip()}")
                print("_" * 200)
            except ValueError as e:
                logging.error(f"Multi-agent error for user {user}: {e}")
        else:
            logging.error(f"No response received from multi-agent for user {user}")

        end = time.time()
        elapsed_time = end - start
        elapsed_time = round(elapsed_time, 2)
        logging.info(f"Total time taken: {elapsed_time} seconds.")

        # Save the results to a CSV file including user_id, real_rating, single_agent_rating, multi_agent_rating, elapsed_time
        results = pd.DataFrame(
            {
                "test": [i],
                "user_id": [user],
                "real_rating": [real_rating],
                "single_agent_rating": [predicted_single],
                "multi_agent_rating": [predicted_multi],
                "elapsed_time": [elapsed_time],
            }
        )

        # Save the results to a CSV file
        csv_dir = BASE_DIR / "results"
        os.makedirs(csv_dir, exist_ok=True)
        csv_file = csv_dir / "metrics.csv"
        results.to_csv(
            csv_dir / csv_file,
            mode="a",
            header=not os.path.exists(csv_dir / csv_file),
            index=False,
        )
