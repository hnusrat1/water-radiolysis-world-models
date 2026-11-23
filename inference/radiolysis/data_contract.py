"""Strict padded NPZ contract. Validation is not permission to inspect sealed tests."""
import numpy as np

SPECIES_NAMES = ("e_aq","OH","H","H2","H3O+","OH-","H2O2")


def _partition(split):
    # The same HELD-OUT source may appear in ID and several intervention-shift
    # test strata. It must never cross train/validation/test partitions.
    if str(split)=='train':return 'train'
    if str(split) in {'val','validation'}:return 'validation'
    return 'test'


def _species_id(value):
    if isinstance(value,(int,np.integer)) and 0 <= int(value) < 7:return int(value)
    if str(value) in SPECIES_NAMES:return SPECIES_NAMES.index(str(value))
    raise ValueError("unknown canonical stable species: "+str(value))


def _cloud(rows):
    xyz=[];species=[]
    for row in rows:
        if len(row)!=4:raise ValueError("particle row must be[species,x_nm,y_nm,z_nm]")
        species.append(_species_id(row[0]));xyz.append([float(x) for x in row[1:]])
    return np.asarray(xyz,dtype=np.float32).reshape(-1,3),np.asarray(species,dtype=np.int64)


def cases_to_arrays(cases):
    """Convert public input metadata only, without opening reference labels."""
    if isinstance(cases,dict):cases=cases["cases"]
    if not cases:raise ValueError("empty cases")
    E=len(cases);N=max(len(c["initial_particles"]) for c in cases)
    K=max([len(c.get("actions",[])) for c in cases]+[0])
    M=max([len(a["particles"]) for c in cases for a in c.get("actions",[])]+[0])
    T=len(cases[0]["times_ns"])
    d={"initial_xyz_nm":np.zeros((E,N,3),np.float32),"initial_species":np.zeros((E,N),np.int64),
       "initial_mask":np.zeros((E,N),bool),"initial_time_ns":np.empty(E,np.float32),
       "pulse_xyz_nm":np.zeros((E,K,M,3),np.float32),"pulse_species":np.zeros((E,K,M),np.int64),
       "pulse_mask":np.zeros((E,K,M),bool),"pulse_time_ns":np.zeros((E,K),np.float32),
       "pulse_valid":np.zeros((E,K),bool),"observation_time_ns":np.empty((E,T),np.float32),
       "family_id":np.asarray([str(c["family_id"]) for c in cases]),
       "state_id":np.asarray([str(c.get("state_id",c["family_id"])) for c in cases]),
       "action_id":np.asarray([str(c.get("action_id",c["case_id"])) for c in cases]),
       "case_id":np.asarray([str(c["case_id"]) for c in cases]),
       "split":np.asarray([str(c["split"]) for c in cases])}
    for i,c in enumerate(cases):
        x,s=_cloud(c["initial_particles"]);n=len(s)
        d["initial_xyz_nm"][i,:n]=x;d["initial_species"][i,:n]=s;d["initial_mask"][i,:n]=True
        d["initial_time_ns"][i]=c["start_ns"];d["observation_time_ns"][i]=c["times_ns"]
        for k,a in enumerate(c.get("actions",[])):
            x,s=_cloud(a["particles"]);n=len(s)
            d["pulse_xyz_nm"][i,k,:n]=x;d["pulse_species"][i,k,:n]=s;d["pulse_mask"][i,k,:n]=True
            d["pulse_time_ns"][i,k]=a["time_ns"];d["pulse_valid"][i,k]=True
    if len(set(d["case_id"]))!=E:raise ValueError("duplicate case IDs")
    validate_arrays(d,require_labels=False)
    return d


REQUIRED = {
    "initial_xyz_nm", "initial_species", "initial_mask", "initial_time_ns",
    "pulse_xyz_nm", "pulse_species", "pulse_mask", "pulse_time_ns", "pulse_valid",
    "observation_time_ns", "target_counts", "family_id", "state_id", "action_id",
    "split", "replicate_count",
}


def validate_arrays(d, *, require_labels=True):
    missing=REQUIRED-set(d)
    if not require_labels:missing-= {"target_counts","replicate_count"}
    if missing:raise ValueError("missing dataset fields: "+str(sorted(missing)))
    xyz=np.asarray(d["initial_xyz_nm"]);species=np.asarray(d["initial_species"]);mask=np.asarray(d["initial_mask"],bool)
    if xyz.ndim!=3 or xyz.shape[-1]!=3 or species.shape!=xyz.shape[:2] or mask.shape!=species.shape:
        raise ValueError("initial clouds must be xyz[E,N,3],species/mask[E,N]")
    E=len(xyz);pulse=np.asarray(d["pulse_xyz_nm"]);pm=np.asarray(d["pulse_mask"],bool)
    ps=np.asarray(d["pulse_species"]);pt=np.asarray(d["pulse_time_ns"]);pv=np.asarray(d["pulse_valid"],bool)
    if pulse.ndim!=4 or pulse.shape[0]!=E or pulse.shape[-1]!=3 or pm.shape!=pulse.shape[:3] or ps.shape!=pm.shape:
        raise ValueError("pulse clouds must be xyz[E,K,M,3],species/mask[E,K,M]")
    if pt.shape!=pulse.shape[:2] or pv.shape!=pt.shape:raise ValueError("pulse time/valid must be[E,K]")
    initial_time=np.asarray(d["initial_time_ns"])
    obs=np.asarray(d["observation_time_ns"])
    if initial_time.shape!=(E,) or obs.ndim!=2 or obs.shape[0]!=E or np.any(np.diff(obs,axis=1)<=0):
        raise ValueError("actual observation times must be strictly increasing per episode")
    if np.any(~np.isfinite(initial_time)) or np.any(initial_time<=0) or np.any(~np.isfinite(obs)) or np.any(~np.isfinite(pt[pv])):
        raise ValueError('positive absolute start and finite native clocks are required')
    if np.any(obs<initial_time[:,None]):raise ValueError("observation precedes observed initial snapshot")
    if np.any(pt[pv]<np.broadcast_to(initial_time[:,None],pt.shape)[pv]):raise ValueError("pulse precedes initial state")
    for row in range(E):
        if np.any(np.diff(pt[row,pv[row]])<0):raise ValueError("pulses must be chronologically ordered")
    if (np.any(~np.isfinite(xyz[mask])) or np.any(~np.isfinite(pulse[pm]))
            or np.any((species[mask]<0)|(species[mask]>=7)) or np.any((ps[pm]<0)|(ps[pm]>=7))):
        raise ValueError("invalid active coordinates or canonical species IDs")
    splits=np.asarray(d["split"]).astype(str);families=np.asarray(d["family_id"])
    if splits.shape!=(E,) or families.shape!=(E,):raise ValueError("split/family must have[E]shape")
    for fam in np.unique(families):
        if len({_partition(x) for x in splits[families==fam]})>1:raise ValueError("family leaks across train/validation/test partitions")
    if require_labels:
        y=np.asarray(d["target_counts"])
        if y.shape!=(E,obs.shape[1],7) or np.any(~np.isfinite(y)) or np.any(y<0):
            raise ValueError("target_counts must be finite nonnegative[E,T,7]")
        if np.asarray(d["replicate_count"]).shape not in [(),(E,)]:raise ValueError("invalid reference replicate count")
    return {"episodes":E,"families":len(np.unique(families)),"times":obs.shape[1],
            "species_order":["e_aq","OH","H","H2","H3O+","OH-","H2O2"]}
