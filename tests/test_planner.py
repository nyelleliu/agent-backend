from types import SimpleNamespace

from app.planner.planner import Planner


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "查询销售数据\n"
                            "分析销售变化\n"
                            "生成分析报告"
                        )
                    )
                )
            ]
        )


def test_planner_creates_plan():
    planner = Planner(FakeClient())

    plan = planner.create_plan(
        "分析今年销售额下降的原因"
    )

    assert plan == [
        "查询销售数据",
        "分析销售变化",
        "生成分析报告",
    ]
