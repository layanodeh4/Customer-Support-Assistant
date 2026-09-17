from agents import (
    Agent,
    Runner,
    function_tool,
)

from src.agent.tools import (
    customer_lookup,
    classify_support,
    search_policy,
)

from src.utils.config import (
    OPENAI_MODEL,
)


@function_tool
def lookup_customer(
    customer_id: str
) -> str:
    """
    Look up customer information
    from the SQL database.
    """

    return str(
        customer_lookup(
            customer_id
        )
    )


@function_tool
def classify_customer_issue(
    message: str
) -> str:
    """
    Classify the customer's support issue.
    """

    return str(
        classify_support(
            message
        )
    )


@function_tool
def retrieve_policy(
    question: str
) -> str:
    """
    Search company policies
    using semantic retrieval.
    """

    return str(
        search_policy(
            question
        )
    )


support_agent = Agent(

    name="E-Commerce Support Agent",

    model=OPENAI_MODEL,

    instructions="""
You are an e-commerce customer support assistant.

Your job is to help customers using the available tools.

Use lookup_customer when you need customer information.

Use classify_customer_issue when you need to understand
the category of a support message.

Use retrieve_policy when the user asks about company
policies such as returns, shipping, or payments.

Do not invent company policy.

If information is not available, say that clearly.

Keep responses helpful and concise.
""",

    tools=[
        lookup_customer,
        classify_customer_issue,
        retrieve_policy,
    ],
)


def ask_agent(
    question: str
):

    result = Runner.run_sync(
        support_agent,
        question
    )

    return result.final_output


if __name__ == "__main__":

    question = input(
        "Ask the support agent: "
    )

    answer = ask_agent(
        question
    )

    print("\nAnswer:")
    print(answer)