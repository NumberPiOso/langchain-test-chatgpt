import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate)
from pydantic import BaseModel, Field


# Define your desired data structure.
class Recipe(BaseModel):
    ingredients: list = Field(description="List of ingredients")
    instructions: str = Field(description="Step by step instructions")


load_dotenv()

OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

PROMPT_RECIPE_INFO = """
    be concise, Provide recipe needed ingredients and step by step instructions
    about a recipe that uses the following ingredients.
    {ingredients}

    {format_instructions}
    """

def main():
    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=Recipe)

    # setup the chat model
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)
    message = HumanMessagePromptTemplate.from_template(
        template=PROMPT_RECIPE_INFO,
    )
    chat_prompt = ChatPromptTemplate.from_messages([message])

    # get user input
    ingredients = input("Enter the ingredients you want separated by comma: ")

    print("Generating response...")
    chat_prompt_with_values = chat_prompt.format_prompt(
        ingredients=ingredients,
        format_instructions=parser.get_format_instructions(),
    )
    print(chat_prompt_with_values.to_messages())
    output = llm(chat_prompt_with_values.to_messages())
    print(output)
    # country = parser.parse(output.content)

    # print the response
    # print(f"The capital of {country.name} is {country.capital}.")


if __name__ == "__main__":
    main()
