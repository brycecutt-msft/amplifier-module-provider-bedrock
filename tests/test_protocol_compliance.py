"""Tests for Provider protocol compliance."""

import pytest
from amplifier_module_provider_bedrock import BedrockProvider


class TestGetInfoMethod:
    """Test get_info() protocol method."""

    def test_has_get_info_method(self):
        """Provider implements get_info() method."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        assert hasattr(provider, "get_info")
        assert callable(provider.get_info)

    def test_get_info_returns_provider_info(self):
        """get_info() returns ProviderInfo instance."""
        from amplifier_core import ProviderInfo
        
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        info = provider.get_info()
        
        assert isinstance(info, ProviderInfo)

    def test_get_info_has_required_fields(self):
        """get_info() returns ProviderInfo with all required fields."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        info = provider.get_info()
        
        assert info.id == "bedrock"
        assert info.display_name == "AWS Bedrock"
        assert isinstance(info.credential_env_vars, list)
        assert isinstance(info.capabilities, list)
        assert isinstance(info.defaults, dict)
        assert isinstance(info.config_fields, list)


class TestListModelsMethod:
    """Test list_models() protocol method."""

    def test_has_list_models_method(self):
        """Provider implements list_models() method."""
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        assert hasattr(provider, "list_models")
        assert callable(provider.list_models)

    @pytest.mark.asyncio
    async def test_list_models_returns_list(self):
        """list_models() returns list of ModelInfo objects."""
        from amplifier_core import ModelInfo
        
        provider = BedrockProvider(config={"aws_region": "us-east-1"})
        models = await provider.list_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, ModelInfo) for m in models)
