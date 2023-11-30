from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import PromptTemplate

from langchain.utilities import DuckDuckGoSearchAPIWrapper
import json
from fpdf import FPDF
from dotenv import find_dotenv, load_dotenv
import os
import re
from datetime import datetime
from langchain.chains import LLMChain
import pandas as pd
import json
import ast


load_dotenv(find_dotenv())


RESULTS_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()


def word_count(text):
    return len(re.findall(r"\w+", text))


# Function to check if the report meets the word count requirement
def check_report_length(report, min_word_count=1000):
    return word_count(report) >= min_word_count


def prompt_for_more_content(current_report, iteration, max_iterations=5):
    # Areas for generating sub-questions focused on recent analysis and current information
    detail_areas = {
        "recent economic indicators": "What are the most recent economic indicators affecting SPY, and how do they compare to the previous quarter's data?",
        "latest analyst opinions": "Can you find the latest analyst opinions on SPY, especially any published in the last month, and summarize their key points?",
        "current market trends": "What are the current market trends impacting SPY, and how might these trends influence its performance in the near future?",
        "historical vs current data": "How does SPY's performance in the current quarter compare to the same quarter in previous years?",
        "recent news events": "Are there any recent news events or developments that have had a significant impact on SPY, and what are the implications of these events?"
    }

    areas_to_expand = []

    # Check for areas to generate sub-questions focused on recent information
    for area, prompt in detail_areas.items():
        if area not in current_report.lower():
            areas_to_expand.append(prompt)

    # Prompt for general expansion if no specific areas are identified
    if not areas_to_expand:
        areas_to_expand.append("What are some additional questions we can ask to understand the latest developments and forecasts regarding SPY's future performance?")

    additional_prompt = f"Iteration {iteration}/{max_iterations}:\nPlease consider the following areas and questions for further exploration with a focus on the most recent information: {', '.join(areas_to_expand)}"
    return additional_prompt


def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


SUMMARY_TEMPLATE = """{text} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


def scrape_text(url: str):
    # Send a GET request to the webpage
    try:
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"


url = "https://en.wikipedia.org/wiki/SPDR_S%26P_500_Trust_ETF"

scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary=RunnablePassthrough.assign(text=lambda x: scrape_text(x["url"]))
    | SUMMARY_PROMPT
    | ChatOpenAI(model="gpt-3.5-turbo-1106")
    | StrOutputParser()
) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")

web_search_chain = (
    RunnablePassthrough.assign(urls=lambda x: web_search(x["question"]))
    | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]])
    | scrape_and_summarize_chain.map()
)


SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

search_question_chain = (
    SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads
)

full_research_chain = (
    search_question_chain
    | (lambda x: [{"question": q} for q in x])
    | web_search_chain.map()
)

current_date = datetime.now().strftime("%B %d, %Y")


WRITER_SYSTEM_PROMPT = """
You are an AI critical thinker and research assistant. Your primary role is to produce well-structured, objective, critically acclaimed, and detailed reports on provided texts. Your reports should be analytical, comprehensive, and exhibit clarity in argumentation."
"""
RESEARCH_REPORT_TEMPLATE = """
## Research Report

### Background Information:
{research_summary}

### Research Question/Topic:
"**{question}**"

### Report Guidelines:
- **Objective**: Formulate a clear, unbiased opinion based on the provided information.
- **Structure**: Adhere to a well-organized format with a distinct introduction, body, and conclusion.
- **Detail**: Include in-depth analysis, focusing on extracting sentiment scores (bullish, neutral, bearish), key economic indicators (like inflation rates, employment figures), and specific forecasts or numerical data from each source. Incorporate facts, figures, and data where applicable.
- **Length**: Target a minimum of 10,000 words, extending as necessary to encompass all relevant information.
- **Formatting**: Use markdown syntax for clear formatting and APA style for academic rigor.
- **Citations**: List all used sources with URLs at the end of the report. Avoid duplicate sources and cite inline using APA format.
- **Tone**: Maintain a journalistic and unbiased tone throughout the report.
- **Deadline Awareness**: Assume the current date is 11/26/2023. Your report should prioritize recently produced news.

### Additional Instructions:
- **Comprehensiveness**: Strive to cover all aspects of the question, using the most relevant and updated information. Pay special attention to recent economic indicators, market sentiment, and analyst forecasts.
- **Critical Analysis**: Evaluate sources critically, focusing on relevance, reliability, and significance. Highlight the sentiment score for each source and any specific economic forecasts or indicators mentioned.
- **Original Insights**: Provide unique perspectives or solutions where possible, grounded in the research.

### Importance:
This report is crucial for my professional development; your highest quality of work is essential.

### Role Emphasis:
As an AI assistant, your contribution is to augment the understanding of the topic and support informed decision-making through insightful, well-researched analysis. Your analysis should include a summary of sentiment scores, economic indicators, and forecasts as extracted from the sources.
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)


def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)


chain = (
    RunnablePassthrough.assign(
        research_summary=full_research_chain | collapse_list_of_lists
    )
    | prompt
    | ChatOpenAI(model="gpt-3.5-turbo-1106")
    | StrOutputParser()
)


def convert_to_pdf(text, filename="../data/reports/11-26-2023.pdf"):
    if not text:
        print("No content to convert to PDF.")
        return

    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "Research Report", 0, 1, "C")

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    try:
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        lines = text.split("\n")
        for line in lines:
            pdf.cell(0, 10, line, ln=True)

        pdf.output(filename)
        print(f"PDF successfully saved as {filename}")
    except Exception as e:
        print(f"An error occurred while creating the PDF: {e}")


def generate_report(question):
    response = chain.invoke({"question": question})
    iteration = 1
    max_iterations = 5 # Set the maximum number of iterations

    while not check_report_length(response, 1000) and iteration <= max_iterations:
        additional_content_prompt = prompt_for_more_content(
            response, iteration, max_iterations
        )
        response += chain.invoke(
            {"question": question, "additional_prompt": additional_content_prompt}
        )
        iteration += 1

    return response

def synthesize_report(initial_report):
    llm = ChatOpenAI(api_key="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")

    prompt = PromptTemplate(
        input_variables=["initial_report"],
        template="""
        SystemMessage: Your task is to analyze the following research report and refine it by summarizing the key information for each source mentioned into a structured format. Your response should be in the form of a list of dictionaries, where each dictionary represents a source with the following keys:

        source_name: [Name of the Source]
        sentiment_score: [1 for bullish, 0 for neutral, -1 for bearish]. This score should be inferred from the key insights and specific forecasts. 
        key_insights: [Summarize important insights, including economic indicators, forecasts, and other relevant analysis]
        specific_forecasts: [List specific numerical forecasts or predictions, if available]

        Example of expected output:
        
        source_name: Source Name,
        sentiment_score: 1,
        key_insights: Detailed insights here...,
        specific_forecasts: Specific forecasts here...

        This format will make it easier to convert the information into a pandas DataFrame. Now, analyze and refine the report as follows:

        ----
        {initial_report}
        ----
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(initial_report=initial_report, temperature=1)
    return response


def process_report_for_lstm(report):
    try:
        # Safely evaluate the string as a Python literal
        data_list = ast.literal_eval(report)

        # Check if the evaluated data is a list
        if not isinstance(data_list, list):
            raise ValueError("Report is not in the expected list format.")

        # Prepare data for DataFrame
        data = {
            "source": [],
            "sentiment_score": []
        }

        # Extract 'source_name' and 'sentiment_score' from each dictionary in the list
        for item in data_list:
            source = item.get('source_name', 'Unknown')
            sentiment_score = item.get('sentiment_score', 0)

            # Append to the data dictionary
            data["source"].append(source)
            data["sentiment_score"].append(sentiment_score)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        df.to_csv("../data/predictions/output.csv")


    except (SyntaxError, ValueError) as e:
        print(f"Error processing report: {e}")
        return pd.DataFrame()

    return df



initial_report = generate_report("Conduct an in-depth analysis of the SPDR S&P 500 ETF Trust (SPY) for the next quarter, focusing on extracting quantifiable sentiment scores, economic indicators, and specific numerical forecasts from various sources. Analyze recent trends, market sentiment, and analyst predictions to determine potential impacts on SPY's performance. Summarize these in a structured format suitable for time series analysis.")

# print(f"Report prior to synthesizing: {initial_report}")
synthesized_report = synthesize_report(initial_report)

print(f"Synthesized report prior to parsing: {synthesized_report}")

# Process the report
report = process_report_for_lstm(synthesized_report)

# Check if the report DataFrame is not empty
if not report.empty:
    # Convert to PDF
    convert_to_pdf(synthesized_report)
    print("PDF successfully saved.")
else:
    print("No report generated.")

# Printing the synthesized report for reference
print(f"Extracted features: {report}")
