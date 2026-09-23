/*
 * RTKLIB single-point positioning oracle: runs RTKLIB `pntpos` on every epoch of a
 * RINEX observation file from several initial positions and prints, for each epoch
 * and initial position, the solution status, the position, the receiver clock and
 * the satellites `estpos` used, as one JSON object. `generate.sh` builds it against
 * RTKLIB and writes `tests/fixtures/rtk/rtklib_spp_selection_oracle.json` from three
 * runs of it.
 *
 * RTKLIB is used unmodified (https://github.com/rtklibexplorer/RTKLIB, branch demo5,
 * commit 75a2e56275485b21a67bd35bc94bbeb8936e1a74). `pntpos` starts `estpos` from
 * `sol->rr`, the initial position given here, and from a zero receiver clock.
 *
 * usage: rtklib_spp_oracle <label> <obs> <nav> <iono 0|1> <tropo 0|1>
 *
 * Options: GPS only, L1 C/A (`-GL1C`), broadcast ephemeris, 10 degree elevation
 * mask, broadcast Klobuchar ionosphere when <iono> is 1 and Saastamoinen troposphere
 * when <tropo> is 1 (otherwise off), no RAIM exclusion. The initial positions are the
 * geocentre (`zero`), the header's APPROX POSITION XYZ (`approx`), and that position
 * turned 12 degrees east (`east`) and 12 degrees west (`west`) about the Earth's axis,
 * some 800 km away, where satellites near the mask sit on the other side of it.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "rtklib.h"

#define TURN_DEG 12.0

/* Progress callbacks the RTKLIB library expects its application to define. */
extern int showmsg(const char *format, ...) { (void)format; return 0; }
extern void settspan(gtime_t ts, gtime_t te) { (void)ts; (void)te; }
extern void settime(gtime_t time) { (void)time; }

static void turned(const double *r, double deg, double *out)
{
    double a = deg * D2R;
    out[0] = cos(a) * r[0] - sin(a) * r[1];
    out[1] = sin(a) * r[0] + cos(a) * r[1];
    out[2] = r[2];
}

static void print_position(const double *r)
{
    printf("[%.17g, %.17g, %.17g]", r[0], r[1], r[2]);
}

int main(int argc, char **argv)
{
    obs_t obs = {0};
    nav_t nav = {0};
    sta_t sta = {{0}};
    prcopt_t opt = prcopt_default;
    const char *names[4] = {"zero", "approx", "east", "west"};
    double guesses[4][3] = {{0}};
    int i, j, k, n, first_case = 1;

    if (argc != 6) {
        fprintf(stderr, "usage: %s <label> <obs> <nav> <iono 0|1> <tropo 0|1>\n", argv[0]);
        return 2;
    }
    if (readrnx(argv[2], 1, "-GL1C", &obs, NULL, &sta) <= 0) {
        fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    if (readrnx(argv[3], 1, "", NULL, &nav, NULL) <= 0) {
        fprintf(stderr, "cannot read %s\n", argv[3]);
        return 1;
    }
    sortobs(&obs);
    uniqnav(&nav);

    opt.mode = PMODE_SINGLE;
    opt.navsys = SYS_GPS;
    opt.nf = 1;
    opt.elmin = 10.0 * D2R;
    opt.sateph = EPHOPT_BRDC;
    opt.ionoopt = argv[4][0] == '1' ? IONOOPT_BRDC : IONOOPT_OFF;
    opt.tropopt = argv[5][0] == '1' ? TROPOPT_SAAS : TROPOPT_OFF;
    opt.posopt[4] = 0;

    for (k = 0; k < 3; k++) guesses[1][k] = sta.pos[k];
    turned(sta.pos, TURN_DEG, guesses[2]);
    turned(sta.pos, -TURN_DEG, guesses[3]);

    printf("{\"label\": \"%s\", \"ionosphere\": %s, \"troposphere\": %s,\n", argv[1],
           opt.ionoopt == IONOOPT_BRDC ? "true" : "false",
           opt.tropopt == TROPOPT_SAAS ? "true" : "false");
    printf(" \"guesses\": {");
    for (j = 0; j < 4; j++) {
        printf("%s\"%s\": ", j ? ", " : "", names[j]);
        print_position(guesses[j]);
    }
    printf("},\n \"cases\": [\n");

    for (i = 0; i < obs.n; i += n) {
        obsd_t epoch[MAXOBS];
        double ep[6];
        int m = 0;
        for (n = 1; i + n < obs.n && timediff(obs.data[i + n].time, obs.data[i].time) == 0.0;
             n++) {
        }
        for (k = 0; k < n; k++) {
            if (satsys(obs.data[i + k].sat, NULL) != SYS_GPS || m >= MAXOBS) continue;
            epoch[m++] = obs.data[i + k];
        }
        time2epoch(obs.data[i].time, ep);
        for (j = 0; j < 4; j++) {
            sol_t sol = {{0}};
            ssat_t ssat[MAXSAT];
            char msg[128] = "";
            int stat, used = 0;
            memset(ssat, 0, sizeof(ssat));
            for (k = 0; k < 3; k++) sol.rr[k] = guesses[j][k];
            stat = pntpos(epoch, m, &nav, &opt, &sol, NULL, ssat, msg);
            printf("%s  {\"epoch\": [%d, %d, %d, %d, %d, %.7f], \"guess\": \"%s\", \"stat\": %d, ",
                   first_case ? "" : ",\n", (int)ep[0], (int)ep[1], (int)ep[2], (int)ep[3],
                   (int)ep[4], ep[5], names[j], stat);
            first_case = 0;
            printf("\"position_m\": ");
            print_position(sol.rr);
            printf(", \"clock_m\": %.17g, \"used\": [", sol.dtr[0] * CLIGHT);
            for (k = 0; k < MAXSAT; k++) {
                char id[8];
                if (!ssat[k].vs) continue;
                satno2id(k + 1, id);
                printf("%s\"%s\"", used++ ? ", " : "", id);
            }
            printf("], \"msg\": \"%s\"}", msg);
        }
    }
    printf("\n ]}\n");
    freeobs(&obs);
    freenav(&nav, 0xFF);
    return 0;
}
