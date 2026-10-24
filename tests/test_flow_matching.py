import torch
from flow_matching import sample_path

def test_t_zero():
    # B=2, C=3, H=4, W=4
    x0 = torch.randn(2, 3, 4, 4)
    x1 = torch.randn(2, 3, 4, 4)
    t = torch.zeros(2)

    returned_t, xt, ut = sample_path(x0, x1, t)

    torch.testing.assert_close(xt, x0)
    torch.testing.assert_close(ut, x1 - x0)

    assert xt.shape == (2, 3, 4, 4)
    assert ut.shape == (2, 3, 4, 4)
    assert returned_t.shape == (2,)

def test_t_one():
    # B=2, C=3, H=4, W=4
    x0 = torch.randn(2, 3, 4, 4)
    x1 = torch.randn(2, 3, 4, 4)
    t = torch.ones(2)

    returned_t, xt, ut = sample_path(x0, x1, t)

    torch.testing.assert_close(ut, x1 - x0)

    assert torch.equal(xt, x1)

def test_t_half():
    x0 = torch.randn(2, 3, 4, 4)
    x1 = torch.randn(2, 3, 4, 4)
    t = torch.full(size=(2,), fill_value=0.5)

    returned_t, xt, ut = sample_path(x0, x1, t)
    assert torch.equal(xt, (x0 + x1) / 2)

