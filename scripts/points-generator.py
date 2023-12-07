from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


skeleton_generator_template = """[User:] As an organizer, provide a detailed skeleton to answer the following question. The skeleton should include key factors, potential risks, and relevant market indicators. Each point should be a concise phrase, approximately 5-10 words, covering different aspects relevant to the question. Aim for 5-12 points to capture a comprehensive overview.
{question}
Skeleton:
[Assistant:] 1."""


skeleton_generator_prompt = ChatPromptTemplate.from_template(
    skeleton_generator_template
)

skeleton_generator_chain = (
    skeleton_generator_prompt | ChatOpenAI() | StrOutputParser() | (lambda x: "1. " + x)
)


point_expander_template = """[User:] Your task is to elaborate on one specific point from the provided skeleton. Focus on current data, recent events, or specific market trends relevant to this point. Provide a detailed yet concise explanation in 1-3 sentences, avoiding broad generalizations. Do not expand on other points.
{question}
The skeleton of the answer is:
{skeleton}
Expand on point {point_index}, adding relevant details and insights.
[Assistant:] {point_index}. {point_skeleton}"""


point_expander_prompt = ChatPromptTemplate.from_template(point_expander_template)

point_expander_chain = RunnablePassthrough.assign(
    continuation=point_expander_prompt | ChatOpenAI() | StrOutputParser()
) | (lambda x: x["point_skeleton"].strip() + " " + x["continuation"])


def parse_numbered_list(input_str):
    """Parses a numbered list into a list of dictionaries

    Each element having two keys:
    'index' for the index in the numbered list, and 'point' for the content.
    """
    # Split the input string into lines
    lines = input_str.split("\n")

    # Initialize an empty list to store the parsed items
    parsed_list = []

    for line in lines:
        # Split each line at the first period to separate the index from the content
        parts = line.split(". ", 1)

        if len(parts) == 2:
            # Convert the index part to an integer
            # and strip any whitespace from the content
            index = int(parts[0])
            point = parts[1].strip()

            # Add a dictionary to the parsed list
            parsed_list.append({"point_index": index, "point_skeleton": point})

    return parsed_list


def create_list_elements(_input):
    skeleton = _input["skeleton"]
    numbered_list = parse_numbered_list(skeleton)
    for el in numbered_list:
        el["skeleton"] = skeleton
        el["question"] = _input["question"]
    return numbered_list


def get_final_answer(expanded_list):
    final_answer_str = "Here's a comprehensive answer:\n\n"
    for i, el in enumerate(expanded_list):
        final_answer_str += f"{i+1}. {el}\n\n"
    return final_answer_str


class ChainInput(BaseModel):
    question: str


chain = (
    RunnablePassthrough.assign(skeleton=skeleton_generator_chain)
    | create_list_elements
    | point_expander_chain.map()
    | get_final_answer
).with_types(input_type=ChainInput)


if __name__ == "__main__":
    response = chain.invoke({"question": "What are the key factors influencing the SPY's performance in the upcoming quarter, considering current economic trends and recent market developments?"})
    print(response)