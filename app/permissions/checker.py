class PermissionChecker:
    ROLE_PERMISSIONS = {
        "employee": {
            "knowledge_search",
            "calculate",
            "get_current_time",
            "calculate_travel_budget",
        },
        "finance": {
            "knowledge_search",
            "calculate",
            "get_current_time",
            "data_analysis",
            "calculate_travel_budget",
        },
        "admin": {
            "knowledge_search",
            "calculate",
            "get_current_time",
            "data_analysis",
            "calculate_travel_budget",
        },
    }

    # Explicit allow-list for dynamically discovered MCP tools.
    # Maps MCP tool name -> roles allowed to execute it. MCP tools
    # NOT listed here are denied for every role (default deny); they
    # never inherit employee/finance/admin permissions implicitly.
    # get_company_info only returns static public company facts
    # (same sensitivity level as get_current_time), so it is granted
    # to all existing roles.
    MCP_ROLE_PERMISSIONS = {
        "get_company_info": {"employee", "finance", "admin"},
    }

    DOCUMENT_PERMISSIONS = {
        "employee": {
            "public",
            "employee",
        },
        "finance": {
            "public",
            "employee",
            "finance",
        },
        "admin": {
            "public",
            "employee",
            "finance",
            "admin",
        },
    }

    def has_permission(self, role: str, skill_name: str) -> bool:
        permissions = self.ROLE_PERMISSIONS.get(role, set())
        return skill_name in permissions

    def has_mcp_permission(self, role: str, tool_name: str) -> bool:
        """Whether ``role`` may execute the MCP tool ``tool_name``.

        Deny by default: only tools present in MCP_ROLE_PERMISSIONS
        with an explicit role grant can run.
        """
        allowed_roles = self.MCP_ROLE_PERMISSIONS.get(tool_name)

        if not allowed_roles:
            return False

        return role in allowed_roles

    def can_upload_document(
        self,
        role: str,
        document_permission: str,
    ) -> bool:
        allowed_permissions = self.DOCUMENT_PERMISSIONS.get(
            role,
            {"public"},
        )

        return document_permission in allowed_permissions
