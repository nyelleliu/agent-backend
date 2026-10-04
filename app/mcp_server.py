from mcp.server.mcpserver import MCPServer

server = MCPServer("enterprise-tools")


@server.tool()
def get_company_info(company_name: str) -> str:
    """Get basic information about a company."""
    companies = {
        "OpenAI": "AI research and deployment company.",
        "DeepSeek": "AI research company.",
    }

    return companies.get(
        company_name,
        f"No information found for {company_name}.",
    )


if __name__ == "__main__":
    server.run()
