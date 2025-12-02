"""
Tests for graceful degradation.

Verifies that BedrockProvider handles missing credentials gracefully
by returning None from mount() instead of raising exceptions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os


class TestGracefulDegradation:
    """Test mount() returns None when credentials are missing."""

    @pytest.mark.asyncio
    async def test_mount_returns_none_without_credentials(self):
        """mount() returns None when no AWS credentials available."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()
        coordinator.register_contributor = MagicMock()
        
        # Clear all AWS environment variables
        with patch.dict(os.environ, {}, clear=True):
            from amplifier_module_provider_bedrock import mount
            
            # No aws_profile, no env vars
            result = await mount(coordinator, config={})
            
            # Should return None for graceful degradation
            assert result is None, "Should return None when credentials missing"
            
            # Should NOT have mounted the provider
            coordinator.mount.assert_not_called()

    @pytest.mark.asyncio
    async def test_mount_logs_warning(self):
        """Logs helpful warning message when credentials missing."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()
        coordinator.register_contributor = MagicMock()
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('amplifier_module_provider_bedrock.logger') as mock_logger:
                from amplifier_module_provider_bedrock import mount
                
                await mount(coordinator, config={})
                
                # Should have logged a warning
                assert mock_logger.warning.called, "Should log warning about missing credentials"

    @pytest.mark.asyncio
    async def test_mount_succeeds_with_profile(self):
        """Returns cleanup function when aws_profile provided."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()
        coordinator.register_contributor = MagicMock()
        
        # Mock boto3 to avoid actual AWS calls
        with patch('amplifier_module_provider_bedrock.AsyncAnthropicBedrock'):
            from amplifier_module_provider_bedrock import mount
            
            # Provide aws_profile in config
            result = await mount(coordinator, config={"aws_profile": "test-profile", "aws_region": "us-east-1"})
            
            # Should return cleanup function (not None)
            assert result is not None, "Should return cleanup function when credentials present"
            assert callable(result), "Should return callable cleanup function"
            
            # Should have mounted the provider
            coordinator.mount.assert_called_once()

    @pytest.mark.asyncio
    async def test_mount_succeeds_with_explicit_keys(self):
        """Returns cleanup function when explicit AWS keys provided."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()
        coordinator.register_contributor = MagicMock()
        
        with patch('amplifier_module_provider_bedrock.AsyncAnthropicBedrock'):
            from amplifier_module_provider_bedrock import mount
            
            # Provide explicit AWS credentials
            result = await mount(
                coordinator,
                config={
                    "aws_access_key": "AKIATEST",
                    "aws_secret_key": "secret",
                    "aws_region": "us-east-1"
                }
            )
            
            assert result is not None, "Should return cleanup function with explicit credentials"
            assert callable(result)
            coordinator.mount.assert_called_once()

    @pytest.mark.asyncio
    async def test_mount_succeeds_with_env_vars(self):
        """Returns cleanup function when AWS env vars present."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()
        coordinator.register_contributor = MagicMock()
        
        # Set AWS environment variables
        env_vars = {
            "AWS_PROFILE": "default",
            "AWS_REGION": "us-east-1"
        }
        
        with patch.dict(os.environ, env_vars):
            with patch('amplifier_module_provider_bedrock.AsyncAnthropicBedrock'):
                from amplifier_module_provider_bedrock import mount
                
                result = await mount(coordinator, config={})
                
                assert result is not None, "Should return cleanup function with env vars"
                assert callable(result)
                coordinator.mount.assert_called_once()

    @pytest.mark.asyncio
    async def test_contribution_channel_always_registered(self):
        """Contribution channel registered even when mount fails gracefully."""
        coordinator = MagicMock()
        coordinator.mount = AsyncMock()
        coordinator.register_contributor = MagicMock()
        
        with patch.dict(os.environ, {}, clear=True):
            from amplifier_module_provider_bedrock import mount
            
            await mount(coordinator, config={})
            
            # Should still register contribution channel (for documentation)
            coordinator.register_contributor.assert_called_once()
