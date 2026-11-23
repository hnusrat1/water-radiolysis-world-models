"""Conservative recurrent closures for source-conditioned native water chemistry.

These are deterministic models of expected molecule counts, not exact SBS
generators. Native identical-reactant EVENT coefficients use k*c**2. Replacing
the unknown factorial moment by a square of the mean is an explicit closure.
The retained state is never refreshed from a future reference particle cloud.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from .network import option1_network, FORMULA, WATER_FORMULA
except ImportError:
    from network import option1_network, FORMULA, WATER_FORMULA


def pack_symmetric(x: Tensor) -> Tensor:
    return torch.stack([x[..., 0, 0], x[..., 1, 1], x[..., 2, 2],
                        x[..., 0, 1], x[..., 0, 2], x[..., 1, 2]], -1)


def unpack_symmetric(x: Tensor) -> Tensor:
    a, b, c, d, e, f = x.unbind(-1)
    return torch.stack([a, d, e, d, b, f, e, f, c], -1).reshape(*x.shape[:-1], 3, 3)


def outer(x: Tensor) -> Tensor:
    return x[..., :, None] * x[..., None, :]


def cloud_counts(species: Tensor, mask: Tensor, dtype: torch.dtype) -> Tensor:
    ids = species.clamp(0, 6).long()
    return (F.one_hot(ids, 7).to(dtype) * mask[..., None]).sum(-2)


def _safe_mean(total: Tensor, mass: Tensor) -> Tensor:
    # Returns zero at absent species, without assigning a one-molecule floor.
    return total / mass.clamp_min(torch.finfo(total.dtype).eps)


@dataclass
class ChemicalState:
    counts: Tensor                         # [B,7], explicit species
    memory: Tensor                         # [B,C], ALL other physical/latent summaries
    origin_nm: Tensor                      # [B,3] or[B,0] for counts-only
    water_ledger: Tensor                   # [B], net implicit water produced
    time_ns: Tensor                        # [B], absolute native clock

    def detach(self):
        return ChemicalState(*(x.detach() for x in (
            self.counts, self.memory, self.origin_nm, self.water_ledger, self.time_ns)))


class MLP(nn.Module):
    def __init__(self, inputs: int, outputs: int, hidden: int, zero_last=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, outputs))
        if zero_last:
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class NativeRecurrentClosure(nn.Module):
    """Shared positive stoichiometric integrator with exact known source jumps.

    No independent species clipping is used. A COMMON extent limiter scales all
    reaction events in a substep if their combined demand exceeds availability.
    Thus atom/charge budgets, with the implicit water ledger, are preserved.
    Fractional-consumption adaptation reduces reliance on that emergency cap;
    cap counts and minimum step are reported for numerical audits.
    """
    def __init__(self, *, memory_size=0, length_nm=10., max_fraction=.20):
        super().__init__()
        n = option1_network()
        for name, x in (("reactants", n.reactants), ("products", n.products),
                        ("stoich", n.stoichiometry), ("water_stoich", n.water),
                        ("volume_rates", n.volume_rates_nm3_ns),
                        ("diffusion", n.diffusion_nm2_ns),
                        ("capture_radius", n.encounter_radii_nm),
                        ("formula", FORMULA), ("water_formula", WATER_FORMULA)):
            self.register_buffer(name, torch.tensor(x, dtype=torch.get_default_dtype()))
        pairs = []
        for r in range(9):
            ids = []
            for s in range(7): ids.extend([s] * int(n.reactants[s, r]))
            pairs.append(ids)
        self.register_buffer("pair_ids", torch.tensor(pairs, dtype=torch.long))
        self.memory_size = int(memory_size)
        self.length_nm = float(length_nm)
        self.max_fraction = float(max_fraction)
        if self.length_nm <= 0 or not 0 < self.max_fraction <= 1:
            raise ValueError("invalid scale or fractional consumption bound")

    def _t(self, x):
        return torch.as_tensor(x, dtype=self.volume_rates.dtype, device=self.volume_rates.device)

    def conserved_totals(self, state):
        return state.counts @ self.formula.T + state.water_ledger[:, None] * self.water_formula

    def storage(self, precision_bytes=4):
        # Species counts, absolute time, signed water ledger, geometry origin.
        origin = 3 if self.memory_size else 0
        values = 7 + self.memory_size + origin + 1 + 1
        return {"retained_values": values, "precision_bytes": precision_bytes,
                "state_bytes": values * precision_bytes,
                "geometry_origin_values": origin,
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "parameter_bytes": sum(p.numel()*p.element_size() for p in self.parameters()),
                "constant_buffer_bytes":sum(b.numel()*b.element_size() for b in self.buffers())}

    def serialize_state(self,state):
        """Actual retained float tensor; no uncounted coordinate side channel."""
        return torch.cat([state.counts,state.memory,state.origin_nm,
                          state.water_ledger[:,None],state.time_ns[:,None]],-1)

    def deserialize_state(self,values):
        values=self._t(values)
        if values.ndim!=2 or values.shape[-1]!=self.storage()['retained_values']:
            raise ValueError('serialized state shape disagrees with declared budget')
        j=7+self.memory_size;k=j+(3 if self.memory_size else 0)
        return ChemicalState(values[:,:7],values[:,7:j],values[:,j:k],values[:,k],values[:,k+1])

    def encode(self, xyz_nm, species, mask, initial_time_ns):
        xyz = self._t(xyz_nm)
        sp = torch.as_tensor(species, device=xyz.device, dtype=torch.long)
        ma = torch.as_tensor(mask, device=xyz.device, dtype=torch.bool)
        counts = cloud_counts(sp, ma, xyz.dtype)
        mass = ma.sum(-1).to(xyz.dtype)
        origin = _safe_mean((xyz * ma[..., None]).sum(-2), mass[:, None])
        if not self.memory_size:origin=xyz.new_zeros((len(xyz),0))
        state = ChemicalState(counts, xyz.new_zeros((len(xyz), self.memory_size)),
                              origin, xyz.new_zeros(len(xyz)), self._t(initial_time_ns))
        if state.time_ns.shape != (len(xyz),) or (state.time_ns <= 0).any():
            raise ValueError("native replay requires a positive absolute initial clock")
        return self._encode_memory(state, xyz, sp, ma)

    def _encode_memory(self, state, xyz, species, mask):
        return state

    def jump(self, state, xyz_nm, species, mask, active=None):
        xyz = self._t(xyz_nm)
        sp = torch.as_tensor(species, device=xyz.device, dtype=torch.long)
        ma = torch.as_tensor(mask, device=xyz.device, dtype=torch.bool)
        if active is not None: ma = ma & active[:, None]
        added = cloud_counts(sp, ma, xyz.dtype)
        memory = self._jump_memory(state, xyz, sp, ma, added)
        memory = torch.where((added.sum(-1)>0)[:,None],memory,state.memory)
        return ChemicalState(state.counts + added, memory, state.origin_nm,
                              state.water_ledger, state.time_ns)

    def _jump_memory(self, state, xyz, species, mask, added):
        return state.memory

    def _diffuse_memory(self, state, dt):
        return state.memory

    def _hazards(self, state):
        raise NotImplementedError

    def _reaction_memory(self, state, extent, counts_new):
        return state.memory

    def _latent_flow(self, state, dt):
        return state.memory

    def evolve(self, state, duration_ns, *, max_step_ns=.5, max_substeps=20000):
        remaining = self._t(duration_ns).expand_as(state.time_ns)
        if (remaining < -1e-6).any() or max_step_ns <= 0:
            raise ValueError("negative duration or invalid maximum step")
        remaining = remaining.clamp_min(0)
        start_time = state.time_ns
        target_time = start_time + remaining
        steps = 0; capped = 0; fraction_limited=0; smallest = math.inf
        # Adaptive dt is detached. Physical gradients still flow through rates,
        # states and reaction extents; the numerical mesh is not differentiated.
        while bool((remaining > 1e-7).any()):
            if steps >= max_substeps: raise RuntimeError("too many recurrent substeps")
            h0 = self._hazards(state)
            if not torch.isfinite(h0).all() or (h0 < 0).any():
                raise FloatingPointError("hazard must be finite and nonnegative")
            demand = h0 @ self.reactants.T
            ratios = torch.where(demand > 1e-12,
                state.counts / demand.clamp_min(1e-12), torch.full_like(demand, math.inf))
            safe = (self.max_fraction * ratios.min(-1).values).detach()
            fraction_limited+=int(((safe<max_step_ns)&(safe<remaining)&(remaining>0)).sum())
            dt = torch.minimum(remaining, torch.minimum(safe, remaining.new_full(remaining.shape, max_step_ns)))
            # If a species is absent, every channel requiring it is gated to
            # exactly zero. The lower numerical bound only avoids stalling at
            # subnormal counts, and the shared extent cap remains authoritative.
            dt = torch.minimum(remaining, dt.clamp_min(1e-6))
            half = self._diffuse_memory(state, dt / 2)
            mid = ChemicalState(state.counts, half, state.origin_nm, state.water_ledger, state.time_ns)
            hazards = self._hazards(mid)
            extent = hazards * dt[:, None]
            demand = extent @ self.reactants.T
            # This emergency NUMERICAL limiter is a stop-gradient control,
            # like the adaptive mesh. Differentiating an inactive quotient at
            # zero/tiny demand can yield0*inf NaNs in float32 even when cap=1.
            # Dynamics and extents remain differentiable through the rates.
            with torch.no_grad():
                denominator=torch.where(demand>0,demand,torch.ones_like(demand))
                cap=torch.where(demand>0,state.counts/denominator,
                                torch.ones_like(demand)).min(-1).values.clamp(0,1)
            capped += int((cap.detach() < 1-1e-5).sum())
            extent = extent * cap[:, None]
            new_counts = state.counts + extent @ self.stoich.T
            # Algebraic demand limiting guarantees >=0. No species-specific
            # clamp is permitted because it would alter atom/charge balances.
            if (new_counts.detach() < -2e-5).any(): raise FloatingPointError("extent limiter failed")
            memory = self._reaction_memory(mid, extent, new_counts)
            next_state = ChemicalState(new_counts, memory, state.origin_nm,
                state.water_ledger + extent @ self.water_stoich, target_time-(remaining-dt).clamp_min(0))
            memory = self._diffuse_memory(next_state, dt / 2)
            next_state = ChemicalState(new_counts, memory, state.origin_nm,
                                       next_state.water_ledger, next_state.time_ns)
            memory = self._latent_flow(next_state, dt)
            # Preserve completed rows exactly, including undefined geometry of
            # absent species, while unrelated rows take further adaptive steps.
            memory = torch.where((dt>0)[:,None],memory,state.memory)
            new_counts = torch.where((dt>0)[:,None],new_counts,state.counts)
            state = ChemicalState(new_counts, memory, state.origin_nm,
                                   next_state.water_ledger, next_state.time_ns)
            remaining = (remaining - dt).clamp_min(0)
            steps += 1
            if (dt.detach() > 0).any(): smallest = min(smallest, float(dt.detach()[dt.detach()>0].min()))
        state.time_ns = target_time
        return state, {"substeps":steps, "capped_cases":capped,
                       "fraction_limited_case_substeps":fraction_limited,
                       "smallest_step_ns":smallest if steps else 0.}

    def rollout(self, batch: dict[str, Any], *, forced=True, max_step_ns=.5):
        state = self.encode(batch["initial_xyz_nm"], batch["initial_species"],
                            batch["initial_mask"], batch["initial_time_ns"])
        times = self._t(batch["observation_time_ns"])
        if times.ndim != 2 or times.shape[0] != len(state.counts):
            raise ValueError("observation_time_ns must be[B,T]")
        if (times[:,1:] <= times[:,:-1]).any(): raise ValueError("times must increase")
        pulse_times = self._t(batch["pulse_time_ns"])
        pulse_valid = torch.as_tensor(batch["pulse_valid"],device=times.device,dtype=torch.bool)
        done = torch.zeros_like(pulse_valid)
        trajectories=[]; ledgers=[]; stat={"substeps":0,"capped_cases":0,"fraction_limited_case_substeps":0,"smallest_step_ns":math.inf}
        initial_budget = self.conserved_totals(state)
        expected_budget = initial_budget
        maximum_budget_error = 0.
        for j in range(times.shape[1]):
            target = times[:,j]
            if forced:
                for k in range(pulse_times.shape[1]):
                    due = pulse_valid[:,k] & ~done[:,k] & (pulse_times[:,k] <= target + 1e-7)
                    if bool(due.any()):
                        duration = torch.where(due,pulse_times[:,k]-state.time_ns,torch.zeros_like(target))
                        state, s = self.evolve(state,duration,max_step_ns=max_step_ns)
                        for key in ["substeps","capped_cases","fraction_limited_case_substeps"]: stat[key]+=s[key]
                        if s["smallest_step_ns"]: stat["smallest_step_ns"]=min(stat["smallest_step_ns"],s["smallest_step_ns"])
                        before = self.conserved_totals(state)
                        state = self.jump(state,batch["pulse_xyz_nm"][:,k],batch["pulse_species"][:,k],
                                          batch["pulse_mask"][:,k],due)
                        expected_budget = expected_budget + self.conserved_totals(state)-before
                        done = done | F.one_hot(torch.tensor(k,device=times.device),pulse_times.shape[1]).bool()[None,:] & due[:,None]
            state,s = self.evolve(state,target-state.time_ns,max_step_ns=max_step_ns)
            for key in ["substeps","capped_cases","fraction_limited_case_substeps"]: stat[key]+=s[key]
            if s["smallest_step_ns"]:stat["smallest_step_ns"]=min(stat["smallest_step_ns"],s["smallest_step_ns"])
            trajectories.append(state.counts);ledgers.append(state.water_ledger)
            maximum_budget_error=max(maximum_budget_error,float((self.conserved_totals(state)-expected_budget).detach().abs().max()))
        if stat["smallest_step_ns"]==math.inf:stat["smallest_step_ns"]=0.
        stat["maximum_atom_charge_drift"]=maximum_budget_error
        return {"counts":torch.stack(trajectories,1),"water_ledger":torch.stack(ledgers,1),
                "state":state,"statistics":stat}


class CountsOnlyClosure(NativeRecurrentClosure):
    """Counts-only learned inverse encounter volumes, not a fixed-box ODE.

    A learned positive inverse-volume head multiplies native k and mean-count
    products. Native k provides units. The initial 1000nm^3 scale is an optimizer
    initialization, not a claimed physical volume; the head can vary by state,
    absolute time and reaction. No future action enters before its source jump.
    """
    def __init__(self, hidden=48):
        super().__init__(memory_size=0)
        self.hazard_head=MLP(9,9,hidden,zero_last=True)

    def _features(self,state):
        return torch.cat([torch.log1p(state.counts.clamp_min(0)),
                          torch.log1p(state.time_ns[:,None]),
                          torch.log1p(state.time_ns[:,None])**2/10],-1)

    def _hazards(self,state):
        log_inverse = -math.log(1000.) + 8*torch.tanh(self.hazard_head(self._features(state))/8)
        a,b=self.pair_ids.T
        return self.volume_rates * state.counts[:,a].clamp_min(0)*state.counts[:,b].clamp_min(0)*log_inverse.exp()


class CompactEquivariantClosure(NativeRecurrentClosure):
    """C16/C32 tensor state with invariant rates and equivariant source geometry.

    C16 = three vectors (9), one symmetric tensor (6), one scalar (1).
    C32 = six vectors (18), two symmetric tensors (12), two scalars (2).
    A fixed retained origin preserves where later local injections arrive.
    Vector/tensor channels use positive species weights. Continuous mixing is
    rotation equivariant; tensors remain PSD through positive combinations.
    """
    def __init__(self,memory_size=16,hidden=48,length_nm=10.):
        if memory_size not in [16,32]:raise ValueError("compact memory must be16or32")
        super().__init__(memory_size=memory_size,length_nm=length_nm)
        self.nv=3 if memory_size==16 else 6
        self.nt=1 if memory_size==16 else 2
        self.ns=self.nt
        self.vector_logits=nn.Parameter(torch.randn(self.nv,7)*.15)
        self.tensor_logits=nn.Parameter(torch.randn(self.nt,7)*.15)
        invariants=9+self.nv*self.nv+self.nt*2+self.nt*self.nv+self.ns
        self.hazard_head=MLP(invariants,9,hidden,zero_last=True)
        self.flow_head=MLP(invariants,self.nv*self.nv+self.nv+self.nt+self.ns,hidden,zero_last=True)
        # Initial scalar pair summary uses species pair kernel sums at2/10nm.
        self.scalar_encoder=MLP(7+2*7*7,self.ns,hidden)
        self.jump_head=MLP(invariants*2,self.ns,hidden,zero_last=True)

    def _split(self,memory):
        i=self.nv*3;j=i+self.nt*6
        return memory[:,:i].reshape(-1,self.nv,3),unpack_symmetric(memory[:,i:j].reshape(-1,self.nt,6)),memory[:,j:]

    def _join(self,v,t,s):
        return torch.cat([v.flatten(1),pack_symmetric(t).flatten(1),s],-1)

    def _weights(self):
        return self.vector_logits.softmax(-1),self.tensor_logits.softmax(-1)

    def _cloud_memory(self,state,xyz,sp,ma):
        r=(xyz-state.origin_nm[:,None,:])/self.length_nm
        hot=F.one_hot(sp.clamp(0,6).long(),7).to(xyz.dtype)*ma[...,None]
        wv,wt=self._weights()
        vw=hot@wv.T;tw=hot@wt.T
        v=_safe_mean(torch.einsum("bnc,bnd->bcd",vw,r),vw.sum(1)[...,None])
        t=_safe_mean(torch.einsum("bnc,bnij->bcij",tw,outer(r)),tw.sum(1)[...,None,None])
        # Intrinsic distances, no coordinate frame features and nofuturequery.
        dist2=((r[:,:,None,:]-r[:,None,:,:])**2).sum(-1)*self.length_nm**2
        identity=torch.eye(r.shape[1],device=r.device,dtype=torch.bool)[None]
        valid=ma[:,:,None]&ma[:,None,:]&~identity
        pair=[]
        for scale in [2.,10.]:
            kernel=torch.exp(-dist2/(2*scale**2))*valid
            pair.append(torch.einsum("bni,bnm,bmj->bij",hot,kernel,hot).flatten(1))
        counts=cloud_counts(sp,ma,xyz.dtype)
        summary=torch.cat([torch.log1p(counts),torch.log1p(torch.cat(pair,-1))],-1)
        scalar=self.scalar_encoder(summary)
        return self._join(v,t,scalar)

    def _encode_memory(self,state,xyz,species,mask):
        state.memory=self._cloud_memory(state,xyz,species,mask)
        return state

    def _features(self,state):
        v,t,s=self._split(state.memory)
        gram=torch.einsum("bvi,bwi->bvw",v,v)
        trace=t.diagonal(dim1=-2,dim2=-1).sum(-1)
        norm=(t*t).sum((-2,-1))
        contraction=torch.einsum("bvi,btij,bvj->btv",v,t,v)
        raw=torch.cat([gram.flatten(1),trace,norm,contraction.flatten(1)],-1)
        # signed log keeps both large and small invariant contractions stable.
        raw=raw.sign()*torch.log1p(raw.abs())
        return torch.cat([torch.log1p(state.counts.clamp_min(0)),torch.log1p(state.time_ns[:,None]),
                          torch.log1p(state.time_ns[:,None])**2/10,raw,s],-1)

    def _jump_memory(self,state,xyz,species,mask,added):
        incoming=self._cloud_memory(state,xyz,species,mask)
        v,t,s=self._split(state.memory);vi,ti,si=self._split(incoming)
        wv,wt=self._weights()
        oldv=state.counts@wv.T;newv=added@wv.T
        oldt=state.counts@wt.T;newt=added@wt.T
        vm=_safe_mean(v*oldv[...,None]+vi*newv[...,None],(oldv+newv)[...,None])
        tm=_safe_mean(t*oldt[...,None,None]+ti*newt[...,None,None],(oldt+newt)[...,None,None])
        oldmass=state.counts.sum(-1,keepdim=True);newmass=added.sum(-1,keepdim=True)
        sm=_safe_mean(s*oldmass+si*newmass,oldmass+newmass)
        incoming_state=ChemicalState(added,incoming,state.origin_nm,state.water_ledger,state.time_ns)
        delta=self.jump_head(torch.cat([self._features(state),self._features(incoming_state)],-1))
        sm=sm+torch.tanh(delta)*(newmass/(oldmass+newmass).clamp_min(1e-8))
        merged=self._join(vm,tm,sm)
        return torch.where((newmass>0),merged,state.memory)

    def _diffuse_memory(self,state,dt):
        v,t,s=self._split(state.memory);_,wt=self._weights()
        weights=state.counts[:,None,:]*wt[None]
        d=_safe_mean((weights*self.diffusion).sum(-1),weights.sum(-1))
        eye=torch.eye(3,device=t.device,dtype=t.dtype)
        t=t+2*d[...,None,None]*dt[:,None,None,None]/self.length_nm**2*eye
        return self._join(v,t,s)

    def _latent_flow(self,state,dt):
        v,t,s=self._split(state.memory)
        output=self.flow_head(self._features(state));i=self.nv*self.nv;j=i+self.nv;k=j+self.nt
        mix=output[:,:i].reshape(-1,self.nv,self.nv).softmax(-1)
        # Small positive initial relaxation retains nonzero learning gradients.
        rate=F.softplus(output[:,i:j]-4)
        retention=torch.exp(-rate*dt[:,None])
        v=retention[...,None]*v+(1-retention)[...,None]*torch.einsum("bvw,bwj->bvj",mix,v)
        trate=F.softplus(output[:,j:k]-4)
        tr=torch.exp(-trate*dt[:,None])
        target=outer(v).mean(1,keepdim=True).expand(-1,self.nt,-1,-1)
        t=tr[...,None,None]*t+(1-tr)[...,None,None]*target
        s=s*torch.exp(-.05*dt[:,None])+(1-torch.exp(-.05*dt[:,None]))*torch.tanh(output[:,k:])
        return self._join(v,t,s)

    def _hazards(self,state):
        _,t,_=self._split(state.memory)
        variance=t[:,0].diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(0)/3*self.length_nm**2
        # A pooled Gaussian volume evolves with diffusion and source geometry.
        # Native encounter radii regularize singular particle summaries.
        overlap=(4*math.pi*(variance[:,None]+self.capture_radius[None,:]**2)).pow(-1.5)
        correction=torch.exp(6*torch.tanh(self.hazard_head(self._features(state))/6))
        a,b=self.pair_ids.T
        return self.volume_rates*state.counts[:,a].clamp_min(0)*state.counts[:,b].clamp_min(0)*overlap*correction


class GaussianMomentClosure(NativeRecurrentClosure):
    """Per-species Gaussian encounter moment closure with known diffusion.

    Retains7means and7symmetric RAW second moments:63geometry floats. Gaussian
    products locate newly produced species. Uniform within-species removal is
    a closure; no claim of exact pair-correlation or SBS kinetics is made.
    Capture smoothing is predeclared, not fitted on test outcomes.
    """
    def __init__(self,*,learned=False,hidden=48,capture_scale=1.,length_nm=10.):
        super().__init__(memory_size=63,length_nm=length_nm)
        self.learned=bool(learned);self.capture_scale=float(capture_scale)
        if capture_scale<=0:raise ValueError("capture_scale must be positive")
        self.hazard_head=MLP(9+9+14,9,hidden,zero_last=True) if learned else None

    def _split(self,m):return m[:,:21].reshape(-1,7,3),unpack_symmetric(m[:,21:].reshape(-1,7,6))
    def _join(self,mu,second):return torch.cat([mu.flatten(1),pack_symmetric(second).flatten(1)],-1)

    def _cloud(self,state,xyz,sp,ma):
        r=(xyz-state.origin_nm[:,None,:])/self.length_nm
        hot=F.one_hot(sp.clamp(0,6).long(),7).to(xyz.dtype)*ma[...,None]
        n=hot.sum(1)
        mu=_safe_mean(torch.einsum("bns,bnd->bsd",hot,r),n[...,None])
        second=_safe_mean(torch.einsum("bns,bnij->bsij",hot,outer(r)),n[...,None,None])
        return mu,second

    def _encode_memory(self,state,xyz,species,mask):
        state.memory=self._join(*self._cloud(state,xyz,species,mask));return state

    def _jump_memory(self,state,xyz,species,mask,added):
        mu,t=self._split(state.memory);mi,ti=self._cloud(state,xyz,species,mask)
        n=state.counts+added
        mu=_safe_mean(mu*state.counts[...,None]+mi*added[...,None],n[...,None])
        t=_safe_mean(t*state.counts[...,None,None]+ti*added[...,None,None],n[...,None,None])
        return self._join(mu,t)

    def _diffuse_memory(self,state,dt):
        mu,t=self._split(state.memory)
        eye=torch.eye(3,device=t.device,dtype=t.dtype)
        t=t+2*self.diffusion[None,:,None,None]*dt[:,None,None,None]/self.length_nm**2*eye
        return self._join(mu,t)

    def _encounters(self,state):
        mu,second=self._split(state.memory)
        cov=(second-outer(mu))*self.length_nm**2
        cov=(cov+cov.transpose(-1,-2))/2
        # Roundoff may create a tiny negative eigenvalue in a raw-moment
        # covariance. PSD projection is GEOMETRY ONLY, never a species clamp.
        with torch.no_grad():
            ev,q=torch.linalg.eigh(cov)
            projected=(q*ev.clamp_min(0)[...,None,:])@q.transpose(-1,-2)
        # Straight-through numerical PSD repair avoids undefined eigenvector
        # derivatives at repeated eigenvalues (e.g. isotropic empty species).
        cov=cov+(projected-cov).detach()
        a,b=self.pair_ids.T;eye=torch.eye(3,device=cov.device,dtype=cov.dtype)
        floor=(self.capture_radius*self.capture_scale).square()[None,:,None,None]/2
        ca=cov[:,a]+floor*eye;cb=cov[:,b]+floor*eye
        total=ca+cb;difference=(mu[:,a]-mu[:,b])*self.length_nm
        chol=torch.linalg.cholesky(total)
        mahal=(difference*torch.cholesky_solve(difference[...,None],chol).squeeze(-1)).sum(-1)
        logdet=2*chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)
        overlap=torch.exp(-.5*(3*math.log(2*math.pi)+logdet+mahal))
        ia=torch.linalg.inv(ca);ib=torch.linalg.inv(cb)
        product_cov=torch.linalg.inv(ia+ib)
        rhs=torch.einsum("brij,brj->bri",ia,mu[:,a]*self.length_nm)+torch.einsum("brij,brj->bri",ib,mu[:,b]*self.length_nm)
        product_mu=torch.einsum("brij,brj->bri",product_cov,rhs)/self.length_nm
        product_second=product_cov/self.length_nm**2+outer(product_mu)
        return overlap,product_mu,product_second,mu,cov

    def _hazards(self,state):
        overlap,_,_,mu,cov=self._encounters(state)
        a,b=self.pair_ids.T
        rate=self.volume_rates*state.counts[:,a].clamp_min(0)*state.counts[:,b].clamp_min(0)*overlap
        if self.hazard_head is not None:
            traces=cov.diagonal(dim1=-2,dim2=-1).sum(-1)
            radial=(mu*mu).sum(-1)
            features=torch.cat([torch.log1p(state.counts.clamp_min(0)),torch.log1p(state.time_ns[:,None]),
                torch.log1p(state.time_ns[:,None])**2/10,torch.log(overlap.clamp_min(1e-30))/20,
                torch.log1p(traces),torch.log1p(radial)],-1)
            rate=rate*torch.exp(6*torch.tanh(self.hazard_head(features)/6))
        return rate

    def _reaction_memory(self,state,extent,counts_new):
        _,pm,pt,mu,_=self._encounters(state);_,t=self._split(state.memory)
        consumed=extent@self.reactants.T
        remaining=(state.counts-consumed).clamp_min(0)
        numerator_mu=mu*remaining[...,None]+torch.einsum("br,sr,bri->bsi",extent,self.products,pm)
        numerator_t=t*remaining[...,None,None]+torch.einsum("br,sr,brij->bsij",extent,self.products,pt)
        return self._join(_safe_mean(numerator_mu,counts_new[...,None]),
                          _safe_mean(numerator_t,counts_new[...,None,None]))


def make_model(method, *, hidden=48, capture_scale=1.):
    if method=="counts":return CountsOnlyClosure(hidden=hidden)
    if method in ["compact16_active","compact16_passive"]:return CompactEquivariantClosure(16,hidden)
    if method in ["compact32_active","compact32_passive"]:return CompactEquivariantClosure(32,hidden)
    if method=="moment_physics":return GaussianMomentClosure(capture_scale=capture_scale)
    if method=="moment_learned":return GaussianMomentClosure(learned=True,hidden=hidden,capture_scale=capture_scale)
    raise ValueError("unknown recurrent method: "+method)
