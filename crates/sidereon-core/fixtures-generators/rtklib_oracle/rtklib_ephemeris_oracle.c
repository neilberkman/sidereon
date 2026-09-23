/*
 * RTKLIB broadcast ephemeris oracle: evaluates RTKLIB's eph2pos, eph2clk, geph2pos,
 * geph2clk, seph2pos and seph2clk on cases read from standard input and prints every
 * output as binary64 bit patterns, so sidereon's evaluators can be compared with
 * RTKLIB bit for bit. `rtklib_ephemeris_oracle.py` writes the input from the committed
 * goldens and collects the output as JSON; build and run instructions are there.
 *
 * The functions below the licence notice are copied from RTKLIB demo5
 * (https://github.com/rtklibexplorer/RTKLIB, branch demo5, commit
 * 75a2e56275485b21a67bd35bc94bbeb8936e1a74): ephemeris.c (eph2clk, eph2pos, deq,
 * glorbit, geph2clk, geph2pos, seph2clk, seph2pos) and rtkcmn.c (epoch2time,
 * gpst2time, timediff, dot3), with only these changes:
 *   - the RTKLIB types hold only the fields these functions read;
 *   - `trace` is removed, and with it the `time2str` buffers its calls used;
 *   - `satsys` reads the system and PRN the harness encodes in `sat`;
 *   - the position/clock variance outputs are not computed;
 *   - `sin`, `cos`, `atan2`, `sqrt` and `fabs` are the Rust `libm` crate's, which
 *     sidereon-core uses, through the `sidereon_libm_*` symbols of `libm_shim`.
 * Compile with `-ffp-contract=off` so that no multiply-add is fused, as RTKLIB's
 * statements are written.
 *
 * RTKLIB licence (see also crates/sidereon-core/RTKLIB-LICENSE.txt):
 *
 *         Copyright (c) 2007-2020, T. Takasu, All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this list
 * of conditions and the following disclaimer. Redistributions in binary form must
 * reproduce the above copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided with the
 * distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double sidereon_libm_sin(double x);
double sidereon_libm_cos(double x);
double sidereon_libm_atan2(double y, double x);
double sidereon_libm_sqrt(double x);
double sidereon_libm_fabs(double x);
#define sin sidereon_libm_sin
#define cos sidereon_libm_cos
#define atan2 sidereon_libm_atan2
#define sqrt sidereon_libm_sqrt
#define fabs sidereon_libm_fabs

/* rtklib.h ---------------------------------------------------------------------*/
#define CLIGHT      299792458.0         /* speed of light (m/s) */
#define OMGE        7.2921151467E-5     /* earth angular velocity (IS-GPS) (rad/s) */
#define SYS_GPS     0x01
#define SYS_SBS     0x02
#define SYS_GLO     0x04
#define SYS_GAL     0x08
#define SYS_QZS     0x10
#define SYS_CMP     0x20
#define SYS_IRN     0x40

typedef struct {
    time_t time;
    double sec;
} gtime_t;

typedef struct {
    int sat;
    gtime_t toe, toc;
    double A, e, i0, OMG0, omg, M0, deln, OMGd, idot;
    double crc, crs, cuc, cus, cic, cis;
    double toes;
    double f0, f1, f2;
} eph_t;

typedef struct {
    int sat;
    gtime_t toe;
    double pos[3], vel[3], acc[3];
    double taun, gamn;
} geph_t;

typedef struct {
    int sat;
    gtime_t t0;
    double pos[3], vel[3], acc[3];
    double af0, af1;
} seph_t;

/* harness: the system and PRN are encoded in `sat` as sys*1000+prn -------------*/
static int satsys(int sat, int *prn)
{
    if (prn) *prn = sat % 1000;
    return sat / 1000;
}

/* rtkcmn.c ---------------------------------------------------------------------*/
static const double gpst0[] = {1980, 1, 6, 0, 0, 0};

static gtime_t epoch2time(const double *ep)
{
    const int doy[]={1,32,60,91,121,152,182,213,244,274,305,335};
    gtime_t time={0};
    int days,sec,year=(int)ep[0],mon=(int)ep[1],day=(int)ep[2];

    if (year<1970||2099<year||mon<1||12<mon) return time;

    /* leap year if year%4==0 in 1901-2099 */
    days=(year-1970)*365+(year-1969)/4+doy[mon-1]+day-2+(year%4==0&&mon>=3?1:0);
    sec=(int)floor(ep[5]);
    time.time=(time_t)days*86400+(int)ep[3]*3600+(int)ep[4]*60+sec;
    time.sec=ep[5]-sec;
    return time;
}
static gtime_t gpst2time(int week, double sec)
{
    gtime_t t=epoch2time(gpst0);

    if (sec<-1E9||1E9<sec) sec=0.0;
    t.time+=(time_t)86400*7*week+(int)sec;
    t.sec=sec-(int)sec;
    return t;
}
static double timediff(gtime_t t1, gtime_t t2)
{
    return difftime(t1.time,t2.time)+t1.sec-t2.sec;
}
static double dot3(const double *a, const double *b)
{
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

/* ephemeris.c ------------------------------------------------------------------*/
#define SQR(x)   ((x)*(x))

#define RE_GLO   6378136.0        /* radius of earth (m)            ref [2] */
#define MU_GPS   3.9860050E14     /* gravitational constant         ref [1] */
#define MU_GLO   3.9860044E14     /* gravitational constant         ref [2] */
#define MU_GAL   3.986004418E14   /* earth gravitational constant   ref [7] */
#define MU_CMP   3.986004418E14   /* earth gravitational constant   ref [9] */
#define J2_GLO   1.0826257E-3     /* 2nd zonal harmonic of geopot   ref [2] */

#define OMGE_GLO 7.292115E-5      /* earth angular velocity (rad/s) ref [2] */
#define OMGE_GAL 7.2921151467E-5  /* earth angular velocity (rad/s) ref [7] */
#define OMGE_CMP 7.292115E-5      /* earth angular velocity (rad/s) ref [9] */

#define SIN_5 -0.0871557427476582 /* sin(-5.0 deg) */
#define COS_5  0.9961946980917456 /* cos(-5.0 deg) */

#define TSTEP    60.0             /* integration step glonass ephemeris (s) */
#define RTOL_KEPLER 1E-13         /* relative tolerance for Kepler equation */
#define MAX_ITER_KEPLER 30        /* max number of iteration of Kepler */

static double eph2clk(gtime_t time, const eph_t *eph)
{
    double t,ts;
    int i;

    t=ts=timediff(time,eph->toc);

    for (i=0;i<2;i++) {
        t=ts-(eph->f0+eph->f1*t+eph->f2*t*t);
    }
    return eph->f0+eph->f1*t+eph->f2*t*t;
}
static void eph2pos(gtime_t time, const eph_t *eph, double *rs, double *dts)
{
    double tk,M,E,Ek,sinE,cosE,u,r,i,O,sin2u,cos2u,x,y,sinO,cosO,cosi,mu,omge;
    double xg,yg,zg,sino,coso;
    int n,sys,prn;

    if (eph->A<=0.0) {
        rs[0]=rs[1]=rs[2]=*dts=0.0;
        return;
    }
    tk=timediff(time,eph->toe);

    switch ((sys=satsys(eph->sat,&prn))) {
        case SYS_GAL: mu=MU_GAL; omge=OMGE_GAL; break;
        case SYS_CMP: mu=MU_CMP; omge=OMGE_CMP; break;
        default:      mu=MU_GPS; omge=OMGE;     break;
    }
    M=eph->M0+(sqrt(mu/(eph->A*eph->A*eph->A))+eph->deln)*tk;

    for (n=0,E=M,Ek=0.0;fabs(E-Ek)>RTOL_KEPLER&&n<MAX_ITER_KEPLER;n++) {
        Ek=E; E-=(E-eph->e*sin(E)-M)/(1.0-eph->e*cos(E));
    }
    sinE=sin(E); cosE=cos(E);

    u=atan2(sqrt(1.0-eph->e*eph->e)*sinE,cosE-eph->e)+eph->omg;
    r=eph->A*(1.0-eph->e*cosE);
    i=eph->i0+eph->idot*tk;
    sin2u=sin(2.0*u); cos2u=cos(2.0*u);
    u+=eph->cus*sin2u+eph->cuc*cos2u;
    r+=eph->crs*sin2u+eph->crc*cos2u;
    i+=eph->cis*sin2u+eph->cic*cos2u;
    x=r*cos(u); y=r*sin(u); cosi=cos(i);

    /* beidou geo satellite */
    if (sys==SYS_CMP&&(prn<=5||prn>=59)) { /* ref [9] table 4-1 */
        O=eph->OMG0+eph->OMGd*tk-omge*eph->toes;
        sinO=sin(O); cosO=cos(O);
        xg=x*cosO-y*cosi*sinO;
        yg=x*sinO+y*cosi*cosO;
        zg=y*sin(i);
        sino=sin(omge*tk); coso=cos(omge*tk);
        rs[0]= xg*coso+yg*sino*COS_5+zg*sino*SIN_5;
        rs[1]=-xg*sino+yg*coso*COS_5+zg*coso*SIN_5;
        rs[2]=-yg*SIN_5+zg*COS_5;
    }
    else {
        O=eph->OMG0+(eph->OMGd-omge)*tk-omge*eph->toes;
        sinO=sin(O); cosO=cos(O);
        rs[0]=x*cosO-y*cosi*sinO;
        rs[1]=x*sinO+y*cosi*cosO;
        rs[2]=y*sin(i);
    }
    tk=timediff(time,eph->toc);
    *dts=eph->f0+eph->f1*tk+eph->f2*tk*tk;

    /* relativity correction */
    *dts-=2.0*sqrt(mu*eph->A)*eph->e*sinE/SQR(CLIGHT);
}
static void deq(const double *x, double *xdot, const double *acc)
{
    double a,b,c,r2=dot3(x,x),r3=r2*sqrt(r2),omg2=SQR(OMGE_GLO);

    if (r2<=0.0) {
        xdot[0]=xdot[1]=xdot[2]=xdot[3]=xdot[4]=xdot[5]=0.0;
        return;
    }
    /* ref [2] A.3.1.2 with bug fix for xdot[4],xdot[5] */
    a=1.5*J2_GLO*MU_GLO*SQR(RE_GLO)/r2/r3; /* 3/2*J2*mu*Ae^2/r^5 */
    b=5.0*x[2]*x[2]/r2;                    /* 5*z^2/r^2 */
    c=-MU_GLO/r3-a*(1.0-b);                /* -mu/r^3-a(1-b) */
    xdot[0]=x[3]; xdot[1]=x[4]; xdot[2]=x[5];
    xdot[3]=(c+omg2)*x[0]+2.0*OMGE_GLO*x[4]+acc[0];
    xdot[4]=(c+omg2)*x[1]-2.0*OMGE_GLO*x[3]+acc[1];
    xdot[5]=(c-2.0*a)*x[2]+acc[2];
}
static void glorbit(double t, double *x, const double *acc)
{
    double k1[6],k2[6],k3[6],k4[6],w[6];
    int i;

    deq(x,k1,acc); for (i=0;i<6;i++) w[i]=x[i]+k1[i]*t/2.0;
    deq(w,k2,acc); for (i=0;i<6;i++) w[i]=x[i]+k2[i]*t/2.0;
    deq(w,k3,acc); for (i=0;i<6;i++) w[i]=x[i]+k3[i]*t;
    deq(w,k4,acc);
    for (i=0;i<6;i++) x[i]+=(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i])*t/6.0;
}
static double geph2clk(gtime_t time, const geph_t *geph)
{
    double t,ts;
    int i;

    t=ts=timediff(time,geph->toe);

    for (i=0;i<2;i++) {
        t=ts-(-geph->taun+geph->gamn*t);
    }
    return -geph->taun+geph->gamn*t;
}
static void geph2pos(gtime_t time, const geph_t *geph, double *rs, double *dts)
{
    double t,tt,x[6];
    int i;

    t=timediff(time,geph->toe);

    *dts=-geph->taun+geph->gamn*t;

    for (i=0;i<3;i++) {
        x[i  ]=geph->pos[i];
        x[i+3]=geph->vel[i];
    }
    for (tt=t<0.0?-TSTEP:TSTEP;fabs(t)>1E-9;t-=tt) {
        if (fabs(t)<TSTEP) tt=t;
        glorbit(tt,x,geph->acc);
    }
    for (i=0;i<3;i++) rs[i]=x[i];
}
static double seph2clk(gtime_t time, const seph_t *seph)
{
    double ts = timediff(time, seph->t0), t = ts;

    for (int i = 0; i < 2; i++) {
      t = ts - (seph->af0 + seph->af1 * t);
    }
    return seph->af0 + seph->af1 * t;
}
static void seph2pos(gtime_t time, const seph_t *seph, double *rs, double *dts)
{
    double t;
    int i;

    t=timediff(time,seph->t0);

    for (i=0;i<3;i++) {
        rs[i]=seph->pos[i]+seph->vel[i]*t+seph->acc[i]*t*t/2.0;
    }
    *dts=seph->af0+seph->af1*t;
}

/* harness ----------------------------------------------------------------------*/
#define WEEK 2111 /* any whole week: each case's times are seconds of week */

static double bits_to_double(const char *token)
{
    uint64_t u = strtoull(token, NULL, 16);
    double d;
    memcpy(&d, &u, sizeof d);
    return d;
}

static void print_bits(const char *label, double value)
{
    uint64_t u;
    memcpy(&u, &value, sizeof u);
    printf(" %s=0x%016llx", label, (unsigned long long)u);
}

static int system_code(char letter)
{
    switch (letter) {
        case 'G': return SYS_GPS;
        case 'E': return SYS_GAL;
        case 'C': return SYS_CMP;
        case 'J': return SYS_QZS;
        case 'I': return SYS_IRN;
        default: return 0;
    }
}

/* `sow` in the week that puts it within half a week of `reference`, as sidereon folds
 * the time from a reference epoch. */
static gtime_t near_time(double sow, gtime_t reference)
{
    gtime_t t = gpst2time(WEEK, sow);
    double dt = timediff(t, reference);
    if (dt > 302400.0) t = gpst2time(WEEK - 1, sow);
    if (dt < -302400.0) t = gpst2time(WEEK + 1, sow);
    return t;
}

/* `reference` plus `offset` seconds, split into whole seconds and a fraction as
 * RTKLIB `gtime_t` holds an instant, so that timediff returns `offset` exactly. */
static gtime_t offset_time(gtime_t reference, double offset)
{
    gtime_t t = reference;
    double whole = floor(offset);
    t.time += (time_t)whole;
    t.sec += offset - whole;
    return t;
}

int main(void)
{
    char line[4096];
    while (fgets(line, sizeof line, stdin)) {
        char *tok[64];
        int n = 0;
        for (char *p = strtok(line, " \t\r\n"); p && n < 64; p = strtok(NULL, " \t\r\n")) {
            tok[n++] = p;
        }
        if (n == 0) continue;
        if (!strcmp(tok[0], "K") && n == 25) {
            /* K name sys prn t toe toc sqrtA e M0 deln OMG0 i0 omg OMGd idot cuc cus crc
             * crs cic cis af0 af1 af2 */
            eph_t eph = {0};
            int sys = system_code(tok[2][0]);
            eph.sat = sys * 1000 + atoi(tok[3]);
            double t_sow = bits_to_double(tok[4]);
            double toe_sow = bits_to_double(tok[5]);
            double toc_sow = bits_to_double(tok[6]);
            double sqrt_a = bits_to_double(tok[7]);
            eph.A = SQR(sqrt_a); /* rinex.c decode_eph: eph->A=SQR(data[10]) */
            eph.e = bits_to_double(tok[8]);
            eph.M0 = bits_to_double(tok[9]);
            eph.deln = bits_to_double(tok[10]);
            eph.OMG0 = bits_to_double(tok[11]);
            eph.i0 = bits_to_double(tok[12]);
            eph.omg = bits_to_double(tok[13]);
            eph.OMGd = bits_to_double(tok[14]);
            eph.idot = bits_to_double(tok[15]);
            eph.cuc = bits_to_double(tok[16]);
            eph.cus = bits_to_double(tok[17]);
            eph.crc = bits_to_double(tok[18]);
            eph.crs = bits_to_double(tok[19]);
            eph.cic = bits_to_double(tok[20]);
            eph.cis = bits_to_double(tok[21]);
            eph.f0 = bits_to_double(tok[22]);
            eph.f1 = bits_to_double(tok[23]);
            eph.f2 = bits_to_double(tok[24]);
            eph.toes = toe_sow;
            eph.toe = gpst2time(WEEK, toe_sow);
            gtime_t time = near_time(t_sow, eph.toe);
            eph.toc = near_time(toc_sow, time);
            double rs[3], dts;
            eph2pos(time, &eph, rs, &dts);
            printf("K %s", tok[1]);
            print_bits("x_m", rs[0]);
            print_bits("y_m", rs[1]);
            print_bits("z_m", rs[2]);
            print_bits("eph2pos_dts", dts);
            print_bits("eph2clk_dts", eph2clk(time, &eph));
            printf("\n");
        } else if (!strcmp(tok[0], "G") && n == 14) {
            /* G name tk clk_bias(-TauN) gamma_n x y z vx vy vz ax ay az */
            geph_t geph = {0};
            double tk = bits_to_double(tok[2]);
            geph.taun = -bits_to_double(tok[3]); /* rinex.c: geph->taun=-data[0] */
            geph.gamn = bits_to_double(tok[4]);
            for (int i = 0; i < 3; i++) {
                geph.pos[i] = bits_to_double(tok[5 + i]);
                geph.vel[i] = bits_to_double(tok[8 + i]);
                geph.acc[i] = bits_to_double(tok[11 + i]);
            }
            geph.toe = gpst2time(WEEK, 0.0);
            gtime_t time = offset_time(geph.toe, tk);
            double rs[3], dts;
            geph2pos(time, &geph, rs, &dts);
            printf("G %s", tok[1]);
            print_bits("x_m", rs[0]);
            print_bits("y_m", rs[1]);
            print_bits("z_m", rs[2]);
            print_bits("geph2pos_dts", dts);
            print_bits("geph2clk_dts", geph2clk(time, &geph));
            printf("\n");
        } else if (!strcmp(tok[0], "S") && n == 14) {
            /* S name t x y z vx vy vz ax ay az af0 af1 */
            seph_t seph = {0};
            double t = bits_to_double(tok[2]);
            for (int i = 0; i < 3; i++) {
                seph.pos[i] = bits_to_double(tok[3 + i]);
                seph.vel[i] = bits_to_double(tok[6 + i]);
                seph.acc[i] = bits_to_double(tok[9 + i]);
            }
            seph.af0 = bits_to_double(tok[12]);
            seph.af1 = bits_to_double(tok[13]);
            seph.t0 = gpst2time(WEEK, 0.0);
            gtime_t time = offset_time(seph.t0, t);
            double rs[3], dts;
            seph2pos(time, &seph, rs, &dts);
            printf("S %s", tok[1]);
            print_bits("x_m", rs[0]);
            print_bits("y_m", rs[1]);
            print_bits("z_m", rs[2]);
            print_bits("seph2pos_dts", dts);
            print_bits("seph2clk_dts", seph2clk(time, &seph));
            printf("\n");
        } else {
            fprintf(stderr, "unreadable case line: %s\n", tok[0]);
            return 1;
        }
    }
    return 0;
}
