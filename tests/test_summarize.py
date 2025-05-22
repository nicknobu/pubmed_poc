# -------------------------------
# file: tests/test_summarize.py
# -------------------------------
"""OpenAI とネットワークをモックした単体テスト"""
import builtins
from unittest import mock
from services.summarize import summarize_pdf


class FakeChat:
    def __init__(self, *_, **__):
        pass

    def invoke(self, _):  # noqa: D401, ANN001
        class _Resp:
            content = "要約1\n要約2\n要約3"
        return _Resp()


@mock.patch("services.summarize.ChatOpenAI", FakeChat)
@mock.patch("infra.pdf_fetcher.fetch_pdf", lambda url: __file__)  # 任意のローカルファイル
def test_summarize_pdf_mock():
    ans = summarize_pdf("dummy")
    assert "要約1" in ans

# =============================================
# これで `streamlit run app.py` または pytest で PoC が動作します。
# =============================================
