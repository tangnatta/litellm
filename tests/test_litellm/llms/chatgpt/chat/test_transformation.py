from unittest.mock import MagicMock, patch

from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig


@patch("litellm.llms.chatgpt.chat.transformation.Authenticator")
def test_validate_environment_uses_selected_account_oauth_token(
    mock_authenticator_class: MagicMock,
) -> None:
    authenticator = mock_authenticator_class.return_value
    authenticator.get_access_token.return_value = "oauth-token"
    authenticator.get_account_id.return_value = "account-id"
    config = ChatGPTConfig()

    headers = config.validate_environment(
        headers={"originator": "custom-origin"},
        model="gpt-5",
        messages=[],
        optional_params={},
        litellm_params={"chatgpt_account": "work"},
    )

    assert headers["Authorization"] == "Bearer oauth-token"
    assert headers["ChatGPT-Account-Id"] == "account-id"
    assert headers["originator"] == "custom-origin"
    authenticator.get_access_token.assert_called_once_with(account="work")
    authenticator.get_account_id.assert_called_once_with(account="work")
