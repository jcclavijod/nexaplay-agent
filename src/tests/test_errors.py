import pytest

from src.skills.nexaplay_api.errors import (
    APICallError,
    ErrorCode,
    classify_http_status,
    should_retry,
)


class TestClassifyHttpStatus:
    def test_500_returns_server_error(self):
        assert classify_http_status(500) == ErrorCode.SERVER_ERROR

    def test_502_returns_server_error(self):
        assert classify_http_status(502) == ErrorCode.SERVER_ERROR

    def test_400_returns_validation_error(self):
        assert classify_http_status(400) == ErrorCode.VALIDATION_ERROR

    def test_422_returns_validation_error(self):
        assert classify_http_status(422) == ErrorCode.VALIDATION_ERROR

    def test_200_raises_value_error(self):
        with pytest.raises(ValueError, match="success status"):
            classify_http_status(200)


class TestShouldRetry:
    def test_network_error_retries(self):
        assert should_retry(ErrorCode.NETWORK_ERROR) is True

    def test_server_error_retries(self):
        assert should_retry(ErrorCode.SERVER_ERROR) is True

    def test_timeout_error_retries(self):
        assert should_retry(ErrorCode.TIMEOUT_ERROR) is True

    def test_validation_error_no_retry(self):
        assert should_retry(ErrorCode.VALIDATION_ERROR) is False

    def test_silent_write_failure_no_retry(self):
        assert should_retry(ErrorCode.SILENT_WRITE_FAILURE) is False


class TestAPICallError:
    def test_attributes_stored_correctly(self):
        err = APICallError(
            code=ErrorCode.SERVER_ERROR,
            message="upstream failure",
            status_code=503,
            retries_used=3,
            details={"raw": "body"},
        )
        assert err.code == ErrorCode.SERVER_ERROR
        assert err.message == "upstream failure"
        assert err.status_code == 503
        assert err.retries_used == 3
        assert err.details == {"raw": "body"}

    def test_defaults(self):
        err = APICallError(code=ErrorCode.NETWORK_ERROR, message="timeout")
        assert err.status_code is None
        assert err.retries_used == 0
        assert err.details is None

    def test_is_exception(self):
        err = APICallError(code=ErrorCode.NETWORK_ERROR, message="conn refused")
        with pytest.raises(APICallError):
            raise err
