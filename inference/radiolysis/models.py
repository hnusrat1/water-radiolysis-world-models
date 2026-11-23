"""Isolated strengthening architectures; no original model file is modified."""
from .recurrent import (CompactEquivariantClosure, NativeRecurrentClosure,
                       MLP, make_model as original_make_model)
from torch import nn
import torch


class Compact64EquivariantClosure(CompactEquivariantClosure):
    """Exactly doubled C32 channels: 12 vectors, 4 symmetric tensors, 4 scalars.

    All encoders, invariant features, jump, hazards and flows are inherited.
    The only new architecture choice is channel multiplicity: 36+24+4=64.
    Complete state is (7 counts+64 memory+3 origin+water+clock)*8=608B,
    versus 600B for the 63-component physical-moment memory.
    Parameter count and computation are measured, not matched by this design.
    """
    def __init__(self, hidden=48, length_nm=10.):
        NativeRecurrentClosure.__init__(self, memory_size=64, length_nm=length_nm)
        self.nv, self.nt, self.ns = 12, 4, 4
        self.vector_logits = nn.Parameter(torch.randn(self.nv, 7) * .15)
        self.tensor_logits = nn.Parameter(torch.randn(self.nt, 7) * .15)
        invariants = 9 + self.nv*self.nv + self.nt*2 + self.nt*self.nv + self.ns
        self.hazard_head = MLP(invariants, 9, hidden, zero_last=True)
        self.flow_head = MLP(invariants, self.nv*self.nv+self.nv+self.nt+self.ns,
                             hidden, zero_last=True)
        self.scalar_encoder = MLP(7+2*7*7, self.ns, hidden)
        self.jump_head = MLP(invariants*2, self.ns, hidden, zero_last=True)


def make_model(method, *, hidden=48, capture_scale=1.):
    if method == 'compact64_active':
        return Compact64EquivariantClosure(hidden=hidden)
    if method not in {'compact32_active', 'moment_learned'}:
        raise ValueError('strengthening method is not predeclared: '+method)
    return original_make_model(method, hidden=hidden, capture_scale=capture_scale)
