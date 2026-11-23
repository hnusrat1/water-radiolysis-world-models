"""Native option1 chemistry contract; event-rate rather than toy factorial units."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

AVOGADRO = 6.02214076e23
SPECIES = ("e_aq", "OH", "H", "H2", "H3O+", "OH-", "H2O2")
FORMULA = np.array([[0,1,1,2,3,1,2], [0,1,0,0,1,1,2], [-1,0,0,0,1,-1,0]],float)
WATER_FORMULA = np.array([2.,1.,0.])


@dataclass
class NativeNetwork:
    species: tuple
    reactants: np.ndarray
    products: np.ndarray
    water: np.ndarray
    rates_M_s: np.ndarray
    diffusion_nm2_ns: np.ndarray
    provenance: dict

    @property
    def stoichiometry(self): return self.products-self.reactants

    @property
    def volume_rates_nm3_ns(self):
        # k [L mol^-1 s^-1] / NA * 10^24 [nm^3/L] * 10^-9 [s/ns]
        return self.rates_M_s * 1e15 / AVOGADRO

    @property
    def encounter_radii_nm(self):
        values=[]
        for r in range(self.reactants.shape[1]):
            ids=np.repeat(np.arange(7),self.reactants[:,r].astype(int))
            d=(self.diffusion_nm2_ns[ids[0]] if ids[0]==ids[1]
               else self.diffusion_nm2_ns[ids].sum())
            values.append(self.volume_rates_nm3_ns[r]/(4*np.pi*d))
        return np.array(values)

    def validate(self):
        S,R=self.reactants.shape
        if tuple(self.species)!=SPECIES or S!=7 or R!=9:
            raise ValueError("native option1 contract requires the canonical7species/9reactions")
        for x in (self.reactants,self.products):
            if x.shape!=(S,R) or np.any(x<0) or np.any(x!=np.floor(x)):
                raise ValueError("invalid stoichiometric matrix")
        if np.any(self.reactants.sum(0)!=2): raise ValueError("option1 requires binary channels")
        residual=FORMULA@self.stoichiometry + WATER_FORMULA[:,None]*self.water[None,:]
        if np.any(residual):raise ValueError("native reaction plus implicit water is unbalanced")
        if np.any(self.rates_M_s<0) or np.any(self.diffusion_nm2_ns<0):raise ValueError("negative rates")
        return self


def option1_network():
    """Pinned audited native9-reaction fixture; replace provenance with exported contract.

    For identical reactants, macroscopic EVENT rate is k[A]^2, not k[A]^2/2.
    N^2 below denotes the Poisson factorial-moment closure for expected counts;
    it must not be sold as exact finite-molecule kinetics.
    """
    react=np.zeros((7,9));prod=np.zeros_like(react)
    pairs=[(0,0),(0,1),(0,2),(0,4),(0,6),(1,1),(1,2),(2,2),(4,5)]
    outputs=[{5:2,3:1},{5:1},{5:1,3:1},{2:1},{5:1,1:1},{6:1},{},{3:1},{}]
    for r,pair in enumerate(pairs):
        for s in pair:react[s,r]+=1
        for s,n in outputs[r].items():prod[s,r]=n
    return NativeNetwork(SPECIES,react,prod,np.array([-2,0,-1,1,0,0,1,0,2.]),
        np.array([6.36e9,2.95e10,2.50e10,2.11e10,1.10e10,5.50e9,1.55e10,5.03e9,1.13e11]),
        np.array([4.9,2.2,7.,4.8,9.46,5.3,2.3]),
        {"constructor":"G4EmDNAChemistry_option1", "geant4":"11.4.3",
         "rate_convention":"native macroscopic event k*c_A*c_B, including identical A+A",
         "source":"source-audited fullstudy chemistry contract; do not fit to new test"}).validate()


def load_network(path=None):
    if path is None:return option1_network()
    d=json.loads(Path(path).read_text())
    return NativeNetwork(tuple(d["species"]),np.array(d["reactants"],float),
        np.array(d["products"],float),np.array(d["water"],float),
        np.array(d["rates_M_s"],float),np.array(d["diffusion_nm2_ns"],float),
        d.get("provenance",{})).validate()
