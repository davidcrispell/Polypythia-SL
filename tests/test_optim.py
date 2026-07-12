import torch

from polypythia_sl.optim import HybridMuon, zeropower_via_newton_schulz


class TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(11, 4)
        self.hidden = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 11, bias=False)

    def forward(self, token_ids):
        hidden = self.norm(self.hidden(self.embed(token_ids)))
        return self.lm_head(hidden)


def test_newton_schulz_preserves_shape_and_is_finite():
    matrix = torch.randn(7, 3)
    result = zeropower_via_newton_schulz(matrix, steps=5)
    assert result.shape == matrix.shape
    assert torch.isfinite(result).all()


def test_hybrid_muon_updates_hidden_and_auxiliary_parameters():
    torch.manual_seed(7)
    model = TinyLanguageModel()
    optimizer = HybridMuon(model, muon_lr=0.01, aux_lr=0.001)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    logits = model(torch.tensor([[1, 2, 3]]))
    logits.square().mean().backward()
    optimizer.step()

    assert "hidden.weight" in optimizer.group_metadata["muon_parameter_names"]
    assert "embed.weight" in optimizer.group_metadata["auxiliary_parameter_names"]
    assert "lm_head.weight" in optimizer.group_metadata["auxiliary_parameter_names"]
    assert all(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
