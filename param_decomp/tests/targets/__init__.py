import pytest

# Preserve pytest's assertion diagnostics in the pooled tests' shared implementation.
pytest.register_assert_rewrite("param_decomp.tests.targets.qwen36_pooled_source_helpers")
