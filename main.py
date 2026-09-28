from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    information = """
    Elon Reeve Musk is a businessman known for his leadership of Tesla, SpaceX,
    X (formerly Twitter), and xAI. Born in Pretoria, South Africa, in 1971, he
    moved to Canada at 17 and later to the United States, where he studied
    physics and economics at the University of Pennsylvania. He co-founded the
    online payments company that became PayPal, founded SpaceX in 2002, and
    joined Tesla as an early investor in 2004, later becoming its CEO.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
