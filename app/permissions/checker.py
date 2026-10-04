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
