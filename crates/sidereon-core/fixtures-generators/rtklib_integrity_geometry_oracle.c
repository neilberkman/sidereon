/* RTKLIB demo5 integrity and antimeridian geometry oracle.
 *
 * Build through generate_rtklib_integrity_geometry.sh with RTKLIB_SRC naming
 * the demo5 source directory at commit 75a2e56275485b21a67bd35bc94bbeb8936e1a74.
 * The harness includes pntpos.c so the recorded valsol verdict comes from the
 * function itself; the build omits the separate pntpos object.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "rtklib.h"
#include "pntpos.c"

extern int showmsg(const char *format, ...) { (void)format; return 0; }
extern void settspan(gtime_t ts, gtime_t te) { (void)ts; (void)te; }
extern void settime(gtime_t time) { (void)time; }

static void print_dop(const char *prefix, const char *name, const double *azel,
                      int ns, double mask_deg)
{
    double out[4];
    int i;
    dops(ns, azel, mask_deg * D2R, out);
    printf("%s{\"name\":\"%s\",\"mask_deg\":%.17g,\"azel_deg\":[",
           prefix, name, mask_deg);
    for (i=0;i<ns;i++) printf("%s[%.17g,%.17g]",i?",":"",azel[2*i]*R2D,azel[2*i+1]*R2D);
    printf("],\"dop\":[%.17g,%.17g,%.17g,%.17g]}",out[0],out[1],out[2],out[3]);
}

static void print_satazel(const char *prefix, const char *name, const double *pos,
                          const double *e)
{
    double azel[2];
    satazel(pos,e,azel);
    printf("%s{\"name\":\"%s\",\"pos_rad_m\":[%.17g,%.17g,%.17g],"
           "\"los_ecef\":[%.17g,%.17g,%.17g],\"az_el_rad\":[%.17g,%.17g]}",
           prefix,name,pos[0],pos[1],pos[2],e[0],e[1],e[2],azel[0],azel[1]);
}

static void print_gate_dop(const double *azel)
{
    double out[4]; int i;
    dops(7,azel,10.0*D2R,out);
    printf(",{\"name\":\"multiclock_above_30_single_below\",\"mask_deg\":10,\"azel_deg\":[");
    for(i=0;i<7;i++) printf("%s[%.17g,%.17g]",i?",":"",azel[2*i]*R2D,azel[2*i+1]*R2D);
    printf("],\"systems\":[\"G\",\"G\",\"G\",\"E\",\"E\",\"E\",\"E\"],\"dop\":[%.17g,%.17g,%.17g,%.17g]}",out[0],out[1],out[2],out[3]);
}

static int valsol_rejects_chisquare(const double *azel, int n)
{
    int vsat[MAXOBS]={0}, i;
    double residuals[5]={100.0,100.0,100.0,100.0,100.0};
    prcopt_t options=prcopt_default;
    char message[128]="";
    for (i=0;i<n;i++) vsat[i]=1;
    options.elmin=10.0*D2R;
    return valsol(azel,vsat,n,&options,residuals,5,4,message)==0;
}

static void integrity(void)
{
    const double single[]={0,85*D2R,45*D2R,30*D2R,135*D2R,20*D2R,225*D2R,40*D2R,315*D2R,25*D2R};
    const double cutoff[]={0,90*D2R,0,10*D2R,90*D2R,10*D2R,180*D2R,10*D2R,270*D2R,9*D2R,315*D2R,35*D2R};
    const double few[]={0,80*D2R,120*D2R,40*D2R,240*D2R,30*D2R};
    const double singular[]={0,45*D2R,0.000001*D2R,45*D2R,0.000002*D2R,45*D2R,0.000003*D2R,45*D2R};
    const double multi[]={5*D2R,75*D2R,70*D2R,35*D2R,145*D2R,20*D2R,220*D2R,50*D2R,300*D2R,15*D2R,340*D2R,42*D2R};
    const double gate[]={269.7455143605609*D2R,36.48822690793828*D2R,262.62724524275774*D2R,30.42170585933661*D2R,317.66930372210123*D2R,11.154867750105367*D2R,245.87928731122895*D2R,76.21329764040074*D2R,214.39229050832233*D2R,62.49820788744436*D2R,357.9599525364323*D2R,35.60033683261509*D2R,231.1080174247327*D2R,71.26351716315052*D2R};
    const double cone[]={0,45*D2R,72*D2R,45*D2R,144*D2R,45*D2R,216*D2R,45*D2R,288*D2R,45*D2R};
    const double coplanar[]={0,0,90*D2R,0,180*D2R,0,270*D2R,0,45*D2R,0};
    const double coincident[]={0,80*D2R,0,80*D2R,120*D2R,30*D2R,240*D2R,30*D2R};
    const double p0[]={0.7,-1.2,50}, zen[]={cos(0.7)*cos(-1.2),cos(0.7)*sin(-1.2),sin(0.7)};
    const double east[]={-sin(-1.2),cos(-1.2),0};
    const double north[]={-sin(0.7)*cos(-1.2),-sin(0.7)*sin(-1.2),cos(0.7)};
    int i;
    printf("{\"provenance\":{\"tool\":\"RTKLIB demo5\",\"commit\":\"75a2e56275485b21a67bd35bc94bbeb8936e1a74\",\"generator\":\"fixtures-generators/generate_rtklib_integrity_geometry.sh\",\"source_functions\":[\"dops\",\"satazel\",\"valsol\"]},\"chisqr_alpha_0_001\":[");
    for(i=0;i<100;i++) printf("%s%.17g",i?",":"",chisqr[i]);
    printf("],\"valsol_chisqr_rejection_active\":%s,\"dop_cases\":[",valsol_rejects_chisquare(single,5)?"true":"false");
    print_dop("","single_constellation",single,5,10);
    print_dop(",","multi_constellation_shared_clock",multi,6,10);
    print_gate_dop(gate);
    print_dop(",","cutoff_equal_and_below",cutoff,6,10);
    print_dop(",","fewer_than_four",few,3,0);
    print_dop(",","near_singular",singular,4,0);
    print_dop(",","rank_deficient_cone",cone,5,0);
    print_dop(",","receiver_coplanar",coplanar,5,-1);
    print_dop(",","two_coincident_of_four",coincident,4,0);
    printf("],\"satazel_cases\":[");
    print_satazel("","zenith",p0,zen); print_satazel(",","horizon_east",p0,east); print_satazel(",","horizon_north",p0,north);
    printf("],\"elevation_mask_cases\":[");
    {
        const double degrees[]={11,15,22,30,44,60,69,88};
        double rr[3]; pos2ecef(p0,rr);
        for(i=0;i<24;i++) {
            double az=37*D2R,el=degrees[i/3]*D2R+(i%3-1)*1E-9,enu[3],e[3],sat[3],azel[2]; int j;
            enu[0]=cos(el)*sin(az); enu[1]=cos(el)*cos(az); enu[2]=sin(el);
            enu2ecef(p0,enu,e);
            for(j=0;j<3;j++) sat[j]=rr[j]+2.0E7*e[j];
            satazel(p0,e,azel);
            printf("%s{\"mask_deg\":%.17g,\"mask_rad\":%.17g,\"receiver_ecef_m\":[%.17g,%.17g,%.17g],\"satellite_ecef_m\":[%.17g,%.17g,%.17g],\"rtklib_az_rad\":%.17g,\"rtklib_el_rad\":%.17g}",i?",":"",degrees[i/3],degrees[i/3]*D2R,rr[0],rr[1],rr[2],sat[0],sat[1],sat[2],azel[0],azel[1]);
        }
    }
    printf("]}\n");
}

static void antimeridian(void)
{
    const double azel[]={0,90*D2R,90*D2R,45*D2R,270*D2R,45*D2R,0,45*D2R};
    double dop[4];
    dops(4,azel,0,dop);
    printf("{\"provenance\":{\"tool\":\"RTKLIB demo5 dops\",\"commit\":\"75a2e56275485b21a67bd35bc94bbeb8936e1a74\",\"generator\":\"fixtures-generators/generate_rtklib_integrity_geometry.sh\"},\"receiver_ecef_m\":[-6378137,0,0],\"azel_rad\":[[0,%.17g],[%.17g,%.17g],[%.17g,%.17g],[0,%.17g]],\"dop\":[%.17g,%.17g,%.17g,%.17g]}\n",90*D2R,90*D2R,45*D2R,270*D2R,45*D2R,45*D2R,dop[0],dop[1],dop[2],dop[3]);
}

int main(int argc,char **argv)
{
    if(argc==2&&strcmp(argv[1],"mask-ecef")==0) {
        double rr[3],rs[3],pos[3],e[3],azel[2],mask;
        while(scanf("%lf %lf %lf %lf %lf %lf %lf",rr,rr+1,rr+2,rs,rs+1,rs+2,&mask)==7) {
            ecef2pos(rr,pos); geodist(rs,rr,e); satazel(pos,e,azel);
            printf("%.17g %d\n",azel[1],azel[1]>=mask);
        }
    }
    else if(argc==2&&strcmp(argv[1],"antimeridian")==0) antimeridian();
    else integrity();
    return 0;
}
