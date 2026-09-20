class PermissionChecker:
    ROLE_PERMISSIONS = {
        "employee": {
            "knowledge_search",
            "calculate",
            "get_current_time",
        },
        "finance": {
            "knowledge_search",
            "calculate",
            "get_current_time",
            "data_analysis",
        },
        "admin": {
            "knowledge_search",
            "calculate",
            "get_current_time",
            "data_analysis",
        },
    }

    def has_permission(self, role: str, skill_name: str) -> bool:
        permissions = self.ROLE_PERMISSIONS.get(role, set())
        return skill_name in permissions

        return skill_name in permissions
