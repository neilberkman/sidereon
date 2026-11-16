/* Native execution can be built from RTKLIB/oracle_harness:
 * cc -std=gnu99 -O0 -include stdio.h -I../src \
 * -DRTKPOS_SOURCE='"../src/rtkpos.c"' \
 * ../../sidereon/crates/sidereon-core/fixtures-generators/rtklib_rtk_mask_oracle.c \
 * librtklib.a -lm -o rtk_mask_oracle
 * The archive must use RTKLIB's default compile-time header options.
 * Regenerate the paired-arithmetic fixture with generate_rtklib_rtk_mask.sh;
 * that script runs these same RTKLIB functions with both math libraries.
 */
#include "rtklib.h"
#include <stdio.h>
#include <string.h>

#define rtkopenstat oracle_rtkopenstat
#define rtkclosestat oracle_rtkclosestat
#define rtkoutstat oracle_rtkoutstat
#define rtkinit oracle_rtkinit
#define rtkfree oracle_rtkfree
#define rtkpos oracle_rtkpos
#ifndef RTKPOS_SOURCE
#error "define RTKPOS_SOURCE to the RTKLIB rtkpos.c path"
#endif
#include RTKPOS_SOURCE
#undef rtkopenstat
#undef rtkclosestat
#undef rtkoutstat
#undef rtkinit
#undef rtkfree
#undef rtkpos

#define NSAT 6
#define NEPOCH 8
static const double base[3]={3512900,780500,5248700};
static const double rover[NEPOCH][3]={{3515900,782500,5248800},{3517900,783500,5249000},
                                     {3519900,784500,5249200},{3521900,785500,5249400},
                                     {3521900,785500,5249400},{3521900,785500,5249400},
                                     {3521900,785500,5249400},{3521900,785500,5249400}};

static void satellite_at(const double *rr,double az,double el,double *rs) {
    double pos[3],enu[3],ecef[3];
    ecef2pos(rr,pos);
    enu[0]=cos(el)*sin(az); enu[1]=cos(el)*cos(az); enu[2]=sin(el);
    ecef[0]=-sin(pos[1])*enu[0]-sin(pos[0])*cos(pos[1])*enu[1]+cos(pos[0])*cos(pos[1])*enu[2];
    ecef[1]= cos(pos[1])*enu[0]-sin(pos[0])*sin(pos[1])*enu[1]+cos(pos[0])*sin(pos[1])*enu[2];
    ecef[2]= cos(pos[0])*enu[1]+sin(pos[0])*enu[2];
    for (int i=0;i<3;i++) rs[i]=rr[i]+20200000.0*ecef[i];
}

static void run_epoch(int ep,double rs0[NSAT][3],int *used,int *nu,
                      int *masked,int *nm,int *reference) {
    prcopt_t opt=prcopt_default; nav_t nav={0}; rtk_t rtk={0};
    obsd_t obs[NSAT*2]={{0}};
    double rs[NSAT*12]={0},dts[NSAT*4]={0},var[NSAT*2]={0};
    double y[NSAT*4]={0},e[NSAT*6]={0},azel[NSAT*4]={0},freq[NSAT*2]={0};
    double v[NSAT*2],R[NSAT*NSAT*4];
    int svh[NSAT*2]={0},sat[NSAT],iu[NSAT],ir[NSAT],vflg[NSAT*2];
    opt.mode=PMODE_DGPS; opt.nf=1; opt.navsys=SYS_GPS;
    opt.elmin=15.0*D2R; opt.ionoopt=IONOOPT_OFF; opt.tropopt=TROPOPT_OFF;
    opt.maxinno[1]=1E9;
    oracle_rtkinit(&rtk,&opt); memcpy(rtk.rb,base,sizeof(base)); memcpy(rtk.x,rover[ep],3*sizeof(double));
    for (int i=0;i<NSAT;i++) rtk.ssat[satno(SYS_GPS,i+1)-1].sys=SYS_GPS;
    if (ep==3) rtk.ssat[satno(SYS_GPS,3)-1].slip[0]=LLI_SLIP;
    if (ep==4||ep==5) rtk.ssat[satno(SYS_GPS,2)-1].slip[0]=LLI_SLIP;
    if (ep==5) for (int i=2;i<NSAT;i++) rtk.ssat[satno(SYS_GPS,i+1)-1].slip[0]=LLI_SLIP;
    for (int receiver=0;receiver<2;receiver++) for (int i=0;i<NSAT;i++) {
        int k=receiver*NSAT+i; const double *rr=receiver?base:rover[ep];
        double los[3],range=geodist(rs0[i],rr,los);
        obs[k].time=gpst2time(2200,ep*30.0); obs[k].sat=satno(SYS_GPS,i+1);
        obs[k].code[0]=CODE_L1C; obs[k].P[0]=range+100+i;
        obs[k].L[0]=(range+50+i)/(CLIGHT/FREQL1); obs[k].SNR[0]=45*4;
        memcpy(rs+k*6,rs0[i],3*sizeof(double));
    }
    for (int i=0;i<NSAT;i++) {
        zdres(0,obs+i,1,rs+i*6,dts+i*2,var+i,svh+i,&nav,rover[ep],&opt,
              y+i*2,e+i*3,azel+i*2,freq+i);
        zdres(1,obs+NSAT+i,1,rs+(NSAT+i)*6,dts+(NSAT+i)*2,var+NSAT+i,
              svh+NSAT+i,&nav,base,&opt,y+(NSAT+i)*2,e+(NSAT+i)*3,
              azel+(NSAT+i)*2,freq+NSAT+i);
    }
    *nm=0; for (int i=0;i<NSAT;i++) if (y[i*2+1]==0.0&&y[(NSAT+i)*2+1]!=0.0) masked[(*nm)++]=i+1;
    int ns=selsat(obs,azel,NSAT,NSAT,&opt,sat,iu,ir);
    *nu=0; for (int i=0;i<ns;i++) if (y[iu[i]*2+1]!=0.0&&y[ir[i]*2+1]!=0.0) used[(*nu)++]=sat[i];
    int nv=ddres(&rtk,obs,0.0,rtk.x,rtk.P,sat,y,e,azel,freq,iu,ir,ns,v,NULL,R,vflg);
    *reference=nv?(vflg[0]>>16)&0xFF:0; oracle_rtkfree(&rtk);
}

static void ids(const int *v,int n) { for (int i=0;i<n;i++) printf("%s\"G%02d\"",i?",":"",v[i]); }

int main(int argc,char **argv) {
    (void)argv;
    const double el[NEPOCH][NSAT]={{14.995,15.030,55,48,42,35},
                                   {15.030,14.995,55,48,42,35},
                                   {14.995,30,55,55,42,35},
                                   {14.995,30,60,55,42,35},
                                   {14.995,60,55,42,35,30},
                                   {14.995,30,60,25,22,20},
                                   {14.995,60,60,42,35,30},
                                   {14.995,60,60,42,35,30}};
    const double az[NSAT]={35,35,20,90,180,280}; double rs[NEPOCH][NSAT][3];
    for (int ep=0;ep<NEPOCH;ep++) for (int i=0;i<NSAT;i++) satellite_at(rover[ep],az[i]*D2R,el[ep][i]*D2R,rs[ep][i]);
    memcpy(rs[2][3],rs[2][2],3*sizeof(double));
    memcpy(rs[6][2],rs[6][1],3*sizeof(double));
    memcpy(rs[7][2],rs[7][1],3*sizeof(double));
    /* Move the later satellite just below the earlier one. Its exact elevation
     * operands are recorded, so the strict >= comparison is independently tested. */
    {
        double pos[3],los[3],azel[2],initial;
        ecef2pos(rover[7],pos); geodist(rs[7][1],rover[7],los); satazel(pos,los,azel); initial=azel[1];
        do { rs[7][2][2]=nextafter(rs[7][2][2],-INFINITY);
             geodist(rs[7][2],rover[7],los); satazel(pos,los,azel);
        } while (azel[1]>=initial);
    }
    /* Optional identical ECEF input for alternate arithmetic libraries. */
    if (argc>1) for (int ep=0;ep<NEPOCH;ep++) for (int i=0;i<NSAT;i++)
        for (int j=0;j<3;j++) if (scanf("%lf",&rs[ep][i][j])!=1) return 2;
    printf("{\"provenance\":{\"implementation\":\"RTKLIB demo5\",\"commit\":\"75a2e56275485b21a67bd35bc94bbeb8936e1a74\",\"generator\":\"fixtures-generators/rtklib_rtk_mask_oracle.c\",\"build\":\"cc -std=gnu99 -O0 -include stdio.h -I../src -DRTKPOS_SOURCE=\\\"../src/rtkpos.c\\\" GENERATOR librtklib.a -lm\",\"source_functions\":[\"rtkpos.c:zdres\",\"rtkpos.c:selsat\",\"rtkpos.c:ddres\"]},\"mask_deg\":15,\"base_ecef_m\":[%.17g,%.17g,%.17g],\"rover_ecef_m\":[",base[0],base[1],base[2]);
    for (int ep=0;ep<NEPOCH;ep++) printf("%s[%.17g,%.17g,%.17g]",ep?",":"",rover[ep][0],rover[ep][1],rover[ep][2]);
    printf("],\"epochs\":[");
    for (int ep=0;ep<NEPOCH;ep++) { int used[NSAT],masked[NSAT],nu,nm,ref;
        run_epoch(ep,rs[ep],used,&nu,masked,&nm,&ref);
        printf("%s{\"satellite_ecef_m\":[",ep?",":"");
        for (int i=0;i<NSAT;i++) printf("%s[%.17g,%.17g,%.17g]",i?",":"",rs[ep][i][0],rs[ep][i][1],rs[ep][i][2]);
        printf("],\"used\":["); ids(used,nu); printf("],\"reference\":\"G%02d\",\"rover_masked\":[",ref); ids(masked,nm);
        printf("],\"elevation_rad\":[");
        for (int i=0;i<NSAT;i++) { double pos[3],los[3],azel[2];
            ecef2pos(rover[ep],pos); geodist(rs[ep][i],rover[ep],los); satazel(pos,los,azel);
            printf("%s%.17g",i?",":"",azel[1]); }
        printf("],\"slipped\":["); if (ep==3) printf("\"G03\"");
        if (ep==4) printf("\"G02\"");
        if (ep==5) printf("\"G02\",\"G03\",\"G04\",\"G05\",\"G06\"");
        printf("]}");
    }
    printf("]}\n"); return 0;
}
