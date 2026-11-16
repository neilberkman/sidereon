#!/usr/bin/env python3
"""Generate Kalman-gate and IGG-III references.

Kalman equations follow FilterPy's update/mahalanobis definitions. IGG-III is
Yang, Song, and Xu's published weight: 1; k0/a*((k1-a)/(k1-k0))**2; 0.
"""
import hashlib,json
from pathlib import Path
import numpy as np, scipy
from scipy.stats import chi2
HERE=Path(__file__).resolve(); OUT=HERE.parent.parent/"tests/fixtures/fusion_gate_oracles.json"
k0,k1=2.,5.
def weight(a):
    if a<=k0:return 1.
    if a<=k1:return (k0/a)*((k1-a)/(k1-k0))**2
    return 0.
points=[np.nextafter(k0,0),k0,np.nextafter(k0,np.inf),3.,4.,4.952790301466122,np.nextafter(k1,0),k1,np.nextafter(k1,np.inf),10.]
grid=np.linspace(k0,k1,301)
# Correlated case: both 2.5-sigma marginals pass a 3-sigma row gate, but the
# anti-correlated joint Mahalanobis statistic is large.
s=np.array([[1.,-.9],[-.9,1.]]); v=np.array([2.5,2.5]); d2=float(v@np.linalg.inv(s)@v)
p=np.diag([2.,3.]); h=np.array([[1.,0.],[.5,1.]]); r=np.array([[1.,.2],[.2,2.]]); y=np.array([1.25,-4.])
S=h@p@h.T+r; norm=np.abs(y)/np.sqrt(np.diag(S)); keep=np.nonzero(norm<=1.5)[0]; hk=h[keep]; rk=r[np.ix_(keep,keep)]; yk=y[keep]; sk=hk@p@hk.T+rk; K=p@hk.T@np.linalg.inv(sk); x=K@yk; I=np.eye(2); post=(I-K@hk)@p@(I-K@hk).T+K@rk@K.T
igg_scan=np.concatenate([np.linspace(4.9,5.1,41),[10.]])
mixed_v=np.array([0.,0.,0.,.2,.2,.2]); mixed_s=np.diag([100.,100.,100.,.01,.01,.01])
raw_gate_innovation=np.array([6.,4.,6.,6.,6.,6.])*np.sqrt(2.); raw_gate_s=np.diag([2.]*6)
data={"provenance":{"tool":"NumPy/SciPy","numpy":np.__version__,"scipy":scipy.__version__,"generator":str(HERE.relative_to(HERE.parents[3])),"generator_sha256":hashlib.sha256(HERE.read_bytes()).hexdigest(),"references":["FilterPy KalmanFilter.update and mahalanobis equations","Yang, Song, Xu IGG III weight function","SciPy scipy.stats.chi2.ppf"]},"inputs":{"k0":k0,"k1":k1},"weights":[[float(x),float(weight(x))] for x in points],"transition_grid":{"points":grid.tolist(),"weights":[float(weight(x)) for x in grid]},"applied_scale_scan":{"standardized":igg_scan.tolist(),"oracle_weights":[float(weight(x)) for x in igg_scan]},"raw_gate_after_igg_removal":{"innovation":raw_gate_innovation.tolist(),"S":raw_gate_s.tolist(),"raw_standardized":(np.abs(raw_gate_innovation)/np.sqrt(np.diag(raw_gate_s))).tolist(),"retained_rows":[],"gate_threshold_sigma":3.},"yang_mixed_units":{"innovation":mixed_v.tolist(),"S":mixed_s.tolist(),"module_trace_statistic":float(mixed_v@mixed_v/np.trace(mixed_s)),"unit_consistent_mean_squared_standardized":float(np.mean(mixed_v*mixed_v/np.diag(mixed_s))),"velocity_block_statistic":float(mixed_v[3:]@mixed_v[3:]/np.trace(mixed_s[3:,3:]))},"yang_chi2_0p99":[float(chi2.ppf(.99,n)) for n in range(1,13)],"correlated_verdict":{"innovation":v.tolist(),"S":s.tolist(),"row_gate_threshold":3.,"row_gate_pass":True,"joint_d2":d2,"joint_threshold":float(chi2.ppf(.99,2)),"joint_pass":bool(d2<=chi2.ppf(.99,2))},"update":{"P":p.tolist(),"H":h.tolist(),"R":r.tolist(),"innovation":y.tolist(),"threshold":1.5,"normalized":norm.tolist(),"accepted":keep.tolist(),"state_delta":x.tolist(),"covariance":post.tolist(),"S":sk.tolist(),"K":K.tolist()}}
OUT.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
