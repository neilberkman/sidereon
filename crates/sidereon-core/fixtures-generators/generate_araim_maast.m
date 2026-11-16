% Generate deterministic MAAST 0.4 ARAIM matrix fixtures.
addpath(getenv('MAAST_DIR')); addpath(fileparts(mfilename('fullpath')));
global PHMI_VERT PHMI_HOR P_THRES PFA_VERT PFA_HOR P_EMT PL_TOL FC_THRES;
global SIG_ACC_MAX_VERT SIG_ACC_MAX_HOR1 SIG_ACC_MAX_HOR2;
PHMI_VERT=9.8e-8; PHMI_HOR=2e-9; P_THRES=8e-8;
PFA_VERT=3.9e-6; PFA_HOR=9e-8; P_EMT=1e-5; PL_TOL=5e-2; FC_THRES=1e-2;
SIG_ACC_MAX_VERT=1.87; SIG_ACC_MAX_HOR1=1e6; SIG_ACC_MAX_HOR2=1e6;
specs = {
 'n06_c1_nominal',6,1,1e-5,0,0.75,0.50,0.75,8e-8;
 'n12_c2_consolidation',12,2,5e-8,1e-4,0.75,0.50,0.75,1e-12;
 'n10_c1_zero_probability',10,1,0,0,0.60,0.40,0.20,8e-8;
 'n12_c2_low_prior',12,2,5e-6,1e-5,1.00,0.60,0.30,8e-8;
 'n18_c3_milestone3',18,3,1e-5,1e-4,0.75,0.50,0.75,8e-8;
 'n24_c2_varied',24,2,2e-5,5e-5,1.20,0.70,0.40,8e-8;
 'n30_c3_low_prior',30,3,1e-6,1e-5,0.90,0.55,0.60,8e-8};
cases=cell(rows(specs),1);
for ci=1:rows(specs)
 name=specs{ci,1}; n=specs{ci,2}; nc=specs{ci,3}; ps=specs{ci,4};
 pc=specs{ci,5}; ura=specs{ci,6}; ure=specs{ci,7}; b=specs{ci,8};
 P_THRES=specs{ci,9};
 G=zeros(n,3+nc); elevations=zeros(n,1);
 for ii=0:n-1
  az=mod(137.50776*ii,360)*pi/180; el=(15+60*mod(0.61803*ii,1))*pi/180;
  G(ii+1,1:3)=[-cos(el)*sin(az),-cos(el)*cos(az),-sin(el)];
  G(ii+1,4+mod(ii,nc))=1; elevations(ii+1)=el;
 end
 local=0.25+2*exp(-(elevations*180/pi)/15); cint=ura^2+local; cacc=ure^2+local;
 [subsets,pap,pnm0]=determine_subsets_v4(G,ps,pc,P_THRES,FC_THRES);
 [sigma,sigma_ss,bias,bias_ss,s1,s2,s3,x,chi2]=compute_subset_solutions(G,cint,cacc,b*ones(n,1),zeros(n,1),subsets,zeros(n,1));
 [sigma,sigma_ss,bias,bias_ss,s1,s2,s3,subsets,pap,pnm]=filter_out_subsets(sigma,sigma_ss,bias,bias_ss,s1,s2,s3,subsets,pap,pnm0,x,chi2);
 [t1,t2,t3]=compute_test_thresholds(sigma_ss,bias_ss,PFA_VERT,PFA_HOR);
 [vpl,hpl,sig_acc,emt]=mhss_raim_baseline_v4(G,cint,cacc,b*ones(n,1),zeros(n,1),ps,pc,0,1);
 item.name=name; item.n_sat=n; item.n_const=nc; item.p_sat=ps; item.p_const=pc;
 [item.max_fault_order,~]=determine_nmax([ps*ones(n,1);pc*ones(nc,1)],P_THRES);
 item.p_threshold_unmonitored=P_THRES;
 item.ura_m=ura; item.ure_m=ure; item.bias_m=b; item.elevation_rad=elevations';
 item.cint_m2=cint'; item.cacc_m2=cacc'; item.excluded=1-subsets;
 item.mode_probability=pap'; item.sigma_int_enu_m=sigma; item.bias_enu_m=bias;
 item.threshold_enu_m=[t1,t2,t3]; item.p_unmonitored=pnm;
 item.vpl_m=vpl; item.hpl_m=hpl; item.emt_m=emt; item.sigma_acc_v_m=sig_acc;
 cases{ci}=item; clear item;
end
root.provenance.tool='MAAST ARAIM 0.4; GNU Octave 11.3.0';
root.provenance.source_url='https://web.stanford.edu/group/scpnt/gpslab/website_files/maast/maast_araimpb0.4.zip';
root.provenance.add_url='https://web.stanford.edu/group/scpnt/gpslab/website_files/maast/ARAIM_TSG_Reference_ADD_v3.0.pdf';
root.provenance.generator_sha256=getenv('GENERATOR_SHA256');
root.provenance.float_format='17 significant digits via jsonencode';
root.inputs.specifications=specs; root.cases=cases;
fid=fopen(getenv('OUT_FILE'),'w'); fprintf(fid,'%s\n',jsonencode(root)); fclose(fid);
