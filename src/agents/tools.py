from typing import Optional
import numexpr
import math
import re

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


class VectorSearchInput(BaseModel):
    query: str = Field(description="The search query for similarity search")
    user_id: Optional[str] = Field(description="The user ID to filter results", default=None)
    thread_id: Optional[str] = Field(description="The thread ID to filter results", default=None)
    k: int = Field(description="Number of results to return", default=4)


class VectorSearchTool(BaseTool):
    """A tool for performing similarity search in vector store."""
    
    name: str = "vector_search"
    description: str = "Useful for searching similar messages or content in the conversation history."
    args_schema: type[BaseModel] = VectorSearchInput
    vector_manager: Optional[object] = None

    def __init__(self, vector_manager=None):
        super().__init__()
        # We'll get the vector manager from the FastAPI app state when the tool is called
        # This requires passing the app state to the tool, which we'll handle in the agent
        object.__setattr__(self, 'vector_manager', vector_manager)

    def _run(self, query: str, user_id: Optional[str] = None, thread_id: Optional[str] = None, k: int = 4) -> str:
        """Synchronous version - not used in async context."""
        raise NotImplementedError("This tool only supports async execution")

    async def _arun(self, query: str, user_id: Optional[str] = None, thread_id: Optional[str] = None, k: int = 4) -> str:
        """Asynchronous version of the tool."""
        # This will be set by the agent when the tool is called
        if self.vector_manager is None:
            return "Error: Vector manager not initialized"
        
        try:
            results = await self.vector_manager.asimilarity_search(
                query=query,
                user_id=user_id,
                thread_id=thread_id,
                k=k
            )
            
            if not results:
                return "No similar content found."
                
            # Format the results
            formatted_results = []
            for i, doc in enumerate(results, 1):
                content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                metadata = doc.metadata
                formatted_results.append(
                    f"{i}. {content} (type: {metadata.get('message_type', 'unknown')}, "
                    f"index: {metadata.get('message_index', 'unknown')})"
                )
            
            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error during vector search: {str(e)}"


# Global instance of the vector search tool (without vector manager initially)
vector_search_tool = VectorSearchTool()


def database_search(query: str) -> str:
    """
    A placeholder function for database search.
    In a real implementation, this would connect to your database and perform the search.
    """
    # This is a placeholder implementation
    # In a real system, you would implement actual database search logic here
    return f"Database search for: {query} - No results found (placeholder implementation)"