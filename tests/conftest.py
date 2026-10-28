import pytest
import torch


@pytest.fixture(scope="session")
def steerer():
    """One loaded model for the whole session; GPU tests skip when there is no room for it."""
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 9 * 2**30:
        pytest.skip("needs CUDA with about 9 GB free; stop server.py first")
    from steer import Steerer

    return Steerer()
