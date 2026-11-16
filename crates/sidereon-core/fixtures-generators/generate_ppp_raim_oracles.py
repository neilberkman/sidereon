#!/usr/bin/env python3
"""Generate snapshot-RAIM reference values.

Equations follow Brown, "A Baseline GPS RAIM Scheme and a Note on the
Equivalence of Three RAIM Methods", Navigation 39(3), 1992: weighted residual
chi-square, Qvv = I-H(H' H)^-1H', w_i=v_i/sqrt(Qvv_ii), and protection-level
slopes times the detectable bias. Quantiles are from SciPy.
"""
import hashlib, json, multiprocessing
from pathlib import Path
import numpy as np
import scipy
import mpmath as mp
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2, norm

HERE = Path(__file__).resolve()
OUT = HERE.parent.parent / "tests/fixtures/ppp_raim_oracles.json"

SAT = [[20200000.,13000000.,21500000.],[-21300000.,14500000.,20700000.],
       [15200000.,-22000000.,19500000.],[-18200000.,-16000000.,21000000.],
       [22000000.,-12000000.,20200000.],[-12000000.,23000000.,18000000.]]
TRUTH=np.array([3512900.,780500.,5248700.])
# WGS84 geodetic latitude/longitude, iterated independently of sidereon.
lon=float(np.arctan2(TRUTH[1],TRUTH[0])); p=float(np.hypot(TRUTH[0],TRUTH[1])); lat=float(np.arctan2(TRUTH[2],p*(1-6.6943799901413165e-3)))
for _ in range(12):
    n=6378137./np.sqrt(1-6.6943799901413165e-3*np.sin(lat)**2)
    lat=float(np.arctan2(TRUTH[2]+6.6943799901413165e-3*n*np.sin(lat),p))

def los(sat):
    d=np.array(sat)-TRUTH; d/=np.linalg.norm(d); return d.tolist()

def design(n):
    g = np.array([[-x, -y, -z, 1.0] for sat in SAT[:n] for x,y,z in [los(sat)]])
    return g

def slope_case(n):
    g = design(n); a = np.linalg.inv(g.T@g)@g.T; s = np.eye(n)-g@a
    rot=np.array([[-np.sin(lon),np.cos(lon),0],[-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],[np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]])
    ae=rot@a[:3]
    hs = np.hypot(ae[0], ae[1])/np.sqrt(np.diag(s)); vs = np.abs(ae[2])/np.sqrt(np.diag(s))
    dof=n-4; threshold=chi2.isf(1e-3,dof)
    lam=brentq(lambda x:ncx2.cdf(threshold,dof,x)-1e-3,0,1e5)
    pbias=np.sqrt(lam)
    q=np.linalg.inv(g.T@g)
    qe=rot@q[:3,:3]@rot.T; dop_h=np.sqrt(qe[0,0]+qe[1,1]); dop_v=np.sqrt(qe[2,2])
    return {"n":n,"dof":dof,"threshold":float(threshold),"pbias":float(pbias),
      "normal_condition_number_2":float(np.linalg.cond(g.T@g)),
      "hpl_slope":float(hs.max()*pbias),"vpl_slope":float(vs.max()*pbias),
      "hpl_dop_two_sided":float(norm.isf(0.001/2)*dop_h),
      "vpl_dop_two_sided":float(norm.isf(0.001/2)*dop_v)}

def weighted_case(weights):
    weights=np.asarray(weights,dtype=float); g=design(len(weights)); hw=weights[:,None]*g
    a=np.linalg.inv(hw.T@hw)@hw.T
    s=np.eye(len(weights))-hw@a
    rot=np.array([[-np.sin(lon),np.cos(lon),0],[-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],[np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]])
    ae=rot@a[:3]; hs=np.hypot(ae[0],ae[1])/np.sqrt(np.diag(s)); vs=np.abs(ae[2])/np.sqrt(np.diag(s))
    dof=len(weights)-4; threshold=chi2.isf(1e-3,dof)
    lam=brentq(lambda value:ncx2.cdf(threshold,dof,value)-1e-3,0,1e5)
    q=np.linalg.inv(g.T@np.diag(weights*weights)@g); qe=rot@q[:3,:3]@rot.T
    k=float(norm.isf(0.001/2))
    return {"weights_inverse_sigma":weights.tolist(),"hpl_slope_m":float(hs.max()*np.sqrt(lam)),
            "vpl_slope_m":float(vs.max()*np.sqrt(lam)),
            "normal_condition_number_2":float(np.linalg.cond(g.T@np.diag(weights*weights)@g)),
            "fault_free_h_m":float(k*np.sqrt(qe[0,0]+qe[1,1])),
            "fault_free_v_m":float(k*np.sqrt(qe[2,2]))}

def solver_design_case(clock_indices, code_weights, phase_weights):
    n=len(clock_indices); clocks=max(clock_indices)+1; rows=[]
    for idx,(clock,code_weight,phase_weight) in enumerate(zip(clock_indices,code_weights,phase_weights)):
        base=[-value for value in los(SAT[idx])]
        common=base+[1.0 if column==clock else 0.0 for column in range(clocks)]
        code=np.array(common+[0.0]*n)*code_weight
        ambiguity=[0.0]*n; ambiguity[idx]=1.0
        phase=np.array(common+ambiguity)*phase_weight
        rows.extend([code,phase])
    h=np.array(rows); q=np.linalg.inv(h.T@h); residual=np.eye(2*n)-h@q@h.T
    rot=np.array([[-np.sin(lon),np.cos(lon),0],[-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],[np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]])
    gain=q@h.T; enu=rot@gain[:3]; code_columns=np.arange(0,2*n,2)
    hs=np.hypot(enu[0,code_columns],enu[1,code_columns])/np.sqrt(np.diag(residual)[code_columns])
    vs=np.abs(enu[2,code_columns])/np.sqrt(np.diag(residual)[code_columns])
    dof=2*n-h.shape[1]; threshold=chi2.isf(1e-3,dof)
    lam=brentq(lambda value:ncx2.cdf(threshold,dof,value)-1e-3,0,1e5)
    return {"clock_indices":clock_indices,"code_weights_inverse_sigma":code_weights,
            "phase_weights_inverse_sigma":phase_weights,"dof":dof,"threshold":float(threshold),
            "normal_condition_number_2":float(np.linalg.cond(h.T@h)),
            "pbias":float(np.sqrt(lam)),"hpl_m":float(hs.max()*np.sqrt(lam)),
            "vpl_m":float(vs.max()*np.sqrt(lam))}

mp.mp.dps = 50
def mp_ncx2_cdf(x, dof, lam):
    total = mp.mpf("0"); poisson = mp.exp(-lam/2); j = 0
    while True:
        term = poisson * mp.gammainc(mp.mpf(dof)/2+j, 0, x/2, regularized=True)
        total += term
        j += 1
        poisson *= (lam/2)/j
        if j > lam/2 and abs(term) < mp.mpf("1e-70"):
            return total

def mp_ncx2_cdf_radial(x, dof, bias, derivative=False, working_dps=150):
    """50-digit radial integral, stable when bias**2 is enormous."""
    x=mp.mpf(x); bias=mp.mpf(bias); radius=mp.sqrt(x)
    def erfc_tail(value):
        if value < 20:
            return mp.erfc(value)
        term=mp.mpf(1); total=term; previous=abs(term)
        for n in range(1,513):
            term *= -mp.mpf(2*n-1)/(2*value*value)
            if abs(term) >= previous:
                break
            total += term; previous=abs(term)
            if abs(term) <= mp.eps*abs(total):
                break
        return mp.exp(-value*value)*total/(value*mp.sqrt(mp.pi))
    def folded(remaining):
        # Guard digits are required when two far-tail erfc values are close;
        # the returned oracle is rounded back to the configured 50 digits.
        with mp.workdps(mp.mp.dps+80):
            lower=(bias-remaining)/mp.sqrt(2)
            upper=(bias+remaining)/mp.sqrt(2)
            return +(erfc_tail(lower)-erfc_tail(upper))/2
    folded_derivative=lambda remaining: (mp.exp(-(remaining+bias)**2/2)-mp.exp(-(remaining-bias)**2/2))/mp.sqrt(2*mp.pi)
    if dof == 1:
        return folded_derivative(radius) if derivative else folded(radius)
    perpendicular=mp.mpf(dof-1)
    scale=mp.power(2,1-perpendicular/2)/mp.gamma(perpendicular/2)
    # Integrate in angle rather than the perpendicular radius.  The direct
    # radius form has a square-root endpoint singularity at sqrt(x), which can
    # make adaptive quadrature miss relative mass in very small tails.
    def integrand(angle):
        value=radius*mp.sin(angle)
        remaining=radius*mp.cos(angle)
        probability=folded_derivative(remaining) if derivative else folded(remaining)
        jacobian=radius*mp.cos(angle)
        return scale*value**(perpendicular-1)*mp.exp(-value*value/2)*probability*jacobian
    # At a 1e-100 target, mpmath's adaptive error estimator needs guard digits
    # to distinguish the requested 50-digit tail from a prematurely accepted
    # quadrature result.  Round the guarded result back to the oracle precision.
    with mp.workdps(working_dps):
        result=mp.quad(integrand,[0,mp.pi/2])
    return +result

NCX2_DOFS=(1,2,3,4,8,20)
NCX2_THRESHOLDS=(1.0,10.83,100.0,400.0,1500.0,1e4,1e6,1e10)
NCX2_PMDS=(1e-1,1e-3,1e-7,1e-16,1e-25,1e-100)

def noncentral_bias_case(arguments):
    dof,threshold,p_md=arguments
    central_logcdf=float(ncx2.logcdf(threshold,dof,0.0))
    if central_logcdf < np.log(p_md):
        return {"dof":dof,"threshold":threshold,"p_md":p_md,"pbias":None}
    objective=lambda bias:float(ncx2.logcdf(threshold,dof,bias*bias)-np.log(p_md))
    hi=max(8.0,np.sqrt(threshold)+50.0)
    scipy_bias=brentq(objective,0.0,hi,xtol=5e-15,rtol=4*np.finfo(float).eps)
    target=mp.mpf(str(p_md))
    radial_seed=np.sqrt(threshold)-norm.ppf(p_md)
    scipy_residual=objective(scipy_bias)
    seed=scipy_bias if np.isfinite(scipy_residual) and abs(scipy_residual) < 1e-8 else radial_seed
    root=mp.mpf(str(seed))
    working_dps=50+max(20,int(np.ceil(-np.log10(p_md))))
    for _ in range(6):
        probability=mp_ncx2_cdf_radial(mp.mpf(str(threshold)),dof,root,working_dps=working_dps)
        probability_derivative=mp_ncx2_cdf_radial(mp.mpf(str(threshold)),dof,root,True,working_dps)
        derivative=probability_derivative/probability
        correction=(mp.log(probability)-mp.log(target))/derivative
        root-=correction
        if abs(correction) < mp.mpf("1e-45"):
            break
    return {"dof":dof,"threshold":threshold,"p_md":p_md,
            "pbias":mp.nstr(root,52),
            "d_pbias_d_logcdf":mp.nstr(1/derivative,52),
            "scipy_pbias":scipy_bias}

def noncentral_bias_grid():
    arguments=[(dof,threshold,p_md) for dof in NCX2_DOFS
               for threshold in NCX2_THRESHOLDS for p_md in NCX2_PMDS]
    with multiprocessing.get_context("fork").Pool(processes=8) as pool:
        return pool.map(noncentral_bias_case,arguments)

def mp_slope_case(n):
    # Convert the binary64 public inputs to exact high-precision decimals. This
    # isolates the inverse/CDF arithmetic rather than silently changing the
    # geometry that the Rust API receives.
    rows=[]
    for point in SAT[:n]:
        unit=[mp.mpf(str(x)) for x in los(point)]
        rows.append([-unit[0],-unit[1],-unit[2],mp.mpf(1)])
    g=mp.matrix(rows); a=(g.T*g)**-1*g.T; s=mp.eye(n)-g*a
    mlon=mp.mpf(str(lon)); mlat=mp.mpf(str(lat))
    rot=mp.matrix([[-mp.sin(mlon),mp.cos(mlon),0],[-mp.sin(mlat)*mp.cos(mlon),-mp.sin(mlat)*mp.sin(mlon),mp.cos(mlat)],[mp.cos(mlat)*mp.cos(mlon),mp.cos(mlat)*mp.sin(mlon),mp.sin(mlat)]])
    ae=rot*a[:3,:]; hs=[]; vs=[]
    for i in range(n):
        denom=mp.sqrt(s[i,i]); hs.append(mp.sqrt(ae[0,i]**2+ae[1,i]**2)/denom); vs.append(abs(ae[2,i])/denom)
    dof=n-4
    threshold = -2*mp.log(mp.mpf("0.001")) if dof == 2 else 2*mp.erfinv(1-mp.mpf("0.001"))**2
    lo=mp.mpf(0); hi=mp.mpf(100)
    for _ in range(220):
        mid=(lo+hi)/2
        if mp_ncx2_cdf(threshold,dof,mid)>mp.mpf("0.001"): lo=mid
        else: hi=mid
    lam=(lo+hi)/2; pbias=mp.sqrt(lam)
    # Inverse conditioning with respect to CDF probability, evaluated by a
    # symmetric high-precision derivative in noncentrality.
    derivative=mp.diff(lambda value: mp_ncx2_cdf(threshold,dof,value),lam)
    return {"n":n,"dof":dof,"threshold":mp.nstr(threshold,52),"pbias":mp.nstr(pbias,52),
      "hpl_slope":mp.nstr(max(hs)*pbias,52),"vpl_slope":mp.nstr(max(vs)*pbias,52),
      "d_lambda_d_cdf":mp.nstr(1/derivative,52)}

payload={
 "provenance":{"tool":"NumPy/SciPy and mpmath","numpy":np.__version__,"scipy":scipy.__version__,"mpmath":mp.__version__,
   "generator":"crates/sidereon-core/fixtures-generators/generate_ppp_raim_oracles.py",
   "generator_sha256":hashlib.sha256(HERE.read_bytes()).hexdigest(),
   "references":["Brown, Navigation 39(3), 1992","SciPy scipy.stats chi2/ncx2/norm"]},
 "inputs":{"false_alarm_probability":1e-3,"missed_detection_probability":1e-3,
   "receiver_ecef_m":TRUTH.tolist(),"receiver_geodetic_rad":[lat,lon],"satellite_ecef_m":SAT},
 "line_of_sight_ecef":[los(x) for x in SAT],
 "chi_square_isf_0p001":[float(chi2.isf(1e-3,k)) for k in range(1,13)],
 "normal":{"one_sided":float(norm.isf(1e-3)),"two_sided":float(norm.isf(5e-4))},
 "slope_cases":[slope_case(6),slope_case(5)],
 "mpmath_slope_cases":[mp_slope_case(6),mp_slope_case(5)],
 "noncentral_bias_grid":noncentral_bias_grid(),
 "weighted_protection":{"code_weight":0.1,"effective_code_sigma_m":10.0,
   "hpl_slope_m":float(slope_case(6)["hpl_slope"]*10.0),
   "vpl_slope_m":float(slope_case(6)["vpl_slope"]*10.0)},
 "nonuniform_weighted_protection":weighted_case([0.1,0.2,0.15,0.3,0.12,0.25]),
 "solver_design_cases":[
   solver_design_case([0,0,0,1,1,1],[1.0]*6,[1.0]*6),
   solver_design_case([0]*6,[0.1,0.2,0.15,0.3,0.12,0.25],[1.0]*6)],
 "global":{"residuals":[[1.0,0.25,0.5,2.0],[-2.0,0.5,1.5,0.25],[0.75,-1.0,2.0,0.5]],
   "sigma":2.0,"n_states":5,
   "weighted_sse":float(sum((c*cw/2)**2+(p*pw/2)**2 for c,p,cw,pw in [[1,.25,.5,2],[-2,.5,1.5,.25],[.75,-1,2,.5]]))}
}
OUT.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
