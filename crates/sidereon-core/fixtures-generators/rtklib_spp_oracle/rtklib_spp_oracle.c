/*
 * RTKLIB single-point positioning oracle: runs RTKLIB `pntpos` on every epoch of a
 * RINEX observation file from several initial positions and prints, for each epoch
 * and initial position, the solution status, the position, its covariance, the
 * receiver clock, the satellites `estpos` used and their `satposs` inputs and final
 * residuals, as one
 * JSON object. The
 * covariance is `sol.qr`, the single-precision position block of `estpos`'s
 * `Q = (H^T W H)^-1` with the pseudorange variances of `rescode`, in the order
 * xx, yy, zz, xy, yz, zx, printed with the nine significant digits that restate a
 * float exactly. `generate.sh` builds it against
 * RTKLIB and writes `tests/fixtures/rtk/rtklib_spp_selection_oracle.json` from five
 * runs of it.
 *
 * RTKLIB is used unmodified (https://github.com/rtklibexplorer/RTKLIB, branch demo5,
 * commit 75a2e56275485b21a67bd35bc94bbeb8936e1a74). `pntpos` starts `estpos` from
 * `sol->rr`, the initial position given here, and from a zero receiver clock.
 *
 * usage: rtklib_spp_oracle <label> <obs> <nav> <iono 0|1> <tropo 0|1>
 *        rtklib_spp_oracle fde <label> <obs> <nav> <iono 0|1> <tropo 0|1> <stride>
 *
 * The `fde` form runs RTKLIB's RAIM fault detection and exclusion, `raim_fde`, on
 * faulted copies of every <stride>-th epoch (see `run_fde` below).
 *
 * Options: GPS only, L1 C/A (`-GL1C`), broadcast ephemeris, 10 degree elevation
 * mask, broadcast Klobuchar ionosphere when <iono> is 1 and Saastamoinen troposphere
 * when <tropo> is 1 (otherwise off), no RAIM exclusion. The initial positions are the
 * geocentre (`zero`), the header's APPROX POSITION XYZ (`approx`), and that position
 * turned 12 degrees east (`east`) and 12 degrees west (`west`) about the Earth's axis,
 * some 800 km away, where satellites near the mask sit on the other side of it.
 */

#include <math.h>
#include <float.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rtklib.h"
static int capture_lsq(const double *weighted_design, const double *residuals,
                       int parameter_count, int measurement_count, double *step,
                       double *covariance);
#define lsq capture_lsq
#include "pntpos.c"
#undef lsq

static int captured_lsq_valid;
static int captured_lsq_columns;
static double captured_lsq_design[(MAXOBS + NX) * NX];
static double captured_lsq_covariance[NX * NX];
static int captured_lsq_state_valid;
static double tracked_receiver_state[NX];
static double captured_lsq_receiver_state[NX];
static double captured_lsq_step[NX];

#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MIN_EXP != -1021 || DBL_MAX_EXP != 1024
#error "the exact transmit-time export requires IEEE-754 binary64 double"
#endif

#define TURN_DEG 12.0

static int capture_lsq(const double *weighted_design, const double *residuals,
                       int parameter_count, int measurement_count, double *step,
                       double *covariance)
{
    int row, column;
    int status = lsq(weighted_design, residuals, parameter_count, measurement_count,
                     step, covariance);
    if (parameter_count != NX || status != 0) return status;
    memcpy(captured_lsq_receiver_state, tracked_receiver_state,
           sizeof(captured_lsq_receiver_state));
    memcpy(captured_lsq_step, step, sizeof(captured_lsq_step));
    for (row = 0; row < parameter_count; row++) tracked_receiver_state[row] += step[row];
    captured_lsq_state_valid = 1;
    if (measurement_count < parameter_count || measurement_count > MAXOBS + NX) return status;
    for (column = 0; column < measurement_count; column++) {
        for (row = 0; row < parameter_count; row++) {
            captured_lsq_design[row + column * parameter_count] =
                weighted_design[row + column * parameter_count];
        }
    }
    memcpy(captured_lsq_covariance, covariance, sizeof(captured_lsq_covariance));
    captured_lsq_columns = measurement_count;
    captured_lsq_valid = 1;
    return status;
}

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

static void print_captured_lsq(void)
{
    int row, column;
    double receiver_geodetic[3];
    printf(", \"lsq_weighted_design_columns\": [");
    for (column = 0; column < captured_lsq_columns; column++) {
        printf("%s[", column ? ", " : "");
        for (row = 0; row < NX; row++) {
            printf("%s%.17g", row ? ", " : "",
                   captured_lsq_design[row + column * NX]);
        }
        printf("]");
    }
    printf("], \"lsq_covariance\": [");
    for (row = 0; row < NX; row++) {
        printf("%s[", row ? ", " : "");
        for (column = 0; column < NX; column++) {
            printf("%s%.17g", column ? ", " : "",
                   captured_lsq_covariance[row + column * NX]);
        }
        printf("]");
    }
    printf("], \"lsq_receiver_state\": [");
    for (row = 0; row < NX; row++) {
        printf("%s%.17g", row ? ", " : "", captured_lsq_receiver_state[row]);
    }
    printf("], \"lsq_step\": [");
    for (row = 0; row < NX; row++) {
        printf("%s%.17g", row ? ", " : "", captured_lsq_step[row]);
    }
    ecef2pos(captured_lsq_receiver_state, receiver_geodetic);
    printf("], \"lsq_geodetic_rad_m\": [%.17g, %.17g, %.17g]",
           receiver_geodetic[0], receiver_geodetic[1], receiver_geodetic[2]);
}

static eph_t *gps_eph_at(gtime_t teph, int sat, const nav_t *nav)
{
    double age, maximum_age = MAXDTOE + 1.0, minimum_age = maximum_age + 1.0;
    int record_index, selected = -1;
    for (record_index = 0; record_index < nav->n; record_index++) {
        if (nav->eph[record_index].sat != sat) continue;
        if ((age = fabs(timediff(nav->eph[record_index].toe, teph))) > maximum_age) continue;
        if (age <= minimum_age) {
            selected = record_index;
            minimum_age = age;
        }
    }
    return selected < 0 ? NULL : nav->eph + selected;
}

static void print_used_states(const obsd_t *obs, int observation_count, gtime_t teph,
                              const nav_t *nav, const prcopt_t *opt, const sol_t *sol,
                              const ssat_t *ssat)
{
    double rs[MAXOBS * 6] = {0}, dts[MAXOBS * 2] = {0}, sat_var[MAXOBS] = {0};
    double receiver_state[NX] = {0}, residual[MAXOBS + NX] = {0};
    double design[NX * (MAXOBS + NX)] = {0}, fit_var[MAXOBS + NX] = {0};
    double azel[MAXOBS * 2] = {0}, resp[MAXOBS] = {0};
    int svh[MAXOBS] = {0}, vsat[MAXOBS] = {0};
    int reference_used_by_satellite[MAXSAT] = {0};
    int satellite_index, observation_index, first = 1, reference_count = 0;
    int reference_satellite_count = 0;
    receiver_state[0] = sol->rr[0];
    receiver_state[1] = sol->rr[1];
    receiver_state[2] = sol->rr[2];
    receiver_state[3] = sol->dtr[0] * CLIGHT;
    satposs(teph, obs, observation_count, nav, opt->sateph, rs, dts, sat_var, svh);
    reference_count = rescode(1, obs, observation_count, rs, dts, sat_var, svh, nav,
                              receiver_state, opt, ssat, residual, design, fit_var,
                              azel, vsat, resp, &reference_satellite_count);
    if (reference_count < NX || reference_satellite_count <= 0) {
        fprintf(stderr, "cannot evaluate RTKLIB residuals at returned solution\n");
        exit(1);
    }
    for (observation_index = 0; observation_index < observation_count; observation_index++) {
        if (vsat[observation_index]) {
            reference_used_by_satellite[obs[observation_index].sat - 1] = 1;
        }
    }
    for (satellite_index = 0; satellite_index < MAXSAT; satellite_index++) {
        if (ssat[satellite_index].vs != reference_used_by_satellite[satellite_index]) {
            fprintf(stderr, "RTKLIB selected set changed at returned solution for %d\n",
                    satellite_index + 1);
            exit(1);
        }
    }
    printf(", \"satellite_states\": [");
    for (satellite_index = 0; satellite_index < MAXSAT; satellite_index++) {
        char id[8];
        int sat = satellite_index + 1;
        gtime_t tx;
        double ep[6], clock, placement_pseudorange = 0.0;
        int64_t tx_j2000_whole_s;
        uint64_t tx_fraction_bits;
        double reference_design[NX];
        int fit_row;
        int frequency_index;
        eph_t *eph;
        if (!reference_used_by_satellite[satellite_index] || satsys(sat, NULL) != SYS_GPS) {
            continue;
        }
        for (observation_index = 0;
             observation_index < observation_count && obs[observation_index].sat != sat;
             observation_index++) {
        }
        if (observation_index == observation_count || !(eph = gps_eph_at(teph, sat, nav))) {
            fprintf(stderr, "missing used satellite input for %d\n", sat);
            exit(1);
        }
        for (frequency_index = 0; frequency_index < NFREQ; frequency_index++) {
            if (obs[observation_index].P[frequency_index] != 0.0) {
                placement_pseudorange = obs[observation_index].P[frequency_index];
                break;
            }
        }
        if (placement_pseudorange == 0.0) {
            fprintf(stderr, "missing used satellite pseudorange for %d\n", sat);
            exit(1);
        }
        tx = timeadd(obs[observation_index].time, -placement_pseudorange / CLIGHT);
        clock = eph2clk(tx, eph);
        tx = timeadd(tx, -clock);
        tx_j2000_whole_s = (int64_t)tx.time - INT64_C(946728000);
        memcpy(&tx_fraction_bits, &tx.sec, sizeof(tx_fraction_bits));
        time2epoch(tx, ep);
        satno2id(sat, id);
        if (!vsat[observation_index]) {
            fprintf(stderr, "reference state absent for used satellite %d\n", sat);
            exit(1);
        }
        fit_row = 0;
        for (observation_index = 0; observation_index < observation_count;
             observation_index++) {
            if (vsat[observation_index] && obs[observation_index].sat == sat) break;
            if (vsat[observation_index]) fit_row++;
        }
        if (observation_index == observation_count || fit_row >= reference_count) {
            fprintf(stderr, "missing reference design row for %d\n", sat);
            exit(1);
        }
        for (frequency_index = 0; frequency_index < NX; frequency_index++) {
            reference_design[frequency_index] = design[frequency_index + fit_row * NX];
        }
        printf("%s{\"sat\": \"%s\", \"transmit_epoch\": [%d, %d, %d, %d, %d, %.17g], ",
               first ? "" : ", ", id, (int)ep[0], (int)ep[1], (int)ep[2], (int)ep[3],
               (int)ep[4], ep[5]);
        printf("\"transmit_j2000_whole_s\": %" PRId64 ", \"transmit_fraction_bits\": \"%016" PRIx64 "\", ",
               tx_j2000_whole_s, tx_fraction_bits);
        first = 0;
        printf("\"position_m\": ");
        print_position(rs + observation_index * 6);
        printf(", \"velocity_m_s\": ");
        print_position(rs + observation_index * 6 + 3);
        printf(", \"clock_s\": %.17g, \"variance_m2\": %.17g, ",
               dts[observation_index * 2], sat_var[observation_index]);
        printf("\"residual_m\": %.17g, \"design_row\": [%.17g, %.17g, %.17g, %.17g], ",
               resp[observation_index], reference_design[0], reference_design[1],
               reference_design[2], reference_design[3]);
        printf("\"reference_variance_m2\": %.17g}", fit_var[fit_row]);
    }
    printf("]");
}

/* ---------------------------------------------------------------------------
 * RAIM FDE oracle.
 *
 * For every <stride>-th epoch, the unmodified `pntpos` first solves the epoch from
 * the header's approximate position. Then, for each satellite it used, faulted
 * copies of the epoch are handed to RTKLIB's own `raim_fde` with the satellite
 * states `pntpos` computes (`satposs`): one per bias in `fde_biases_m` added to
 * that satellite's L1 pseudorange, and one per bias in `fde_pair_biases_m` added
 * to it and to the next used satellite (two simultaneous faults). demo5's
 * `pntpos` reaches `raim_fde` only when `estpos` fails, because `valsol`'s
 * chi-square rejection is commented out, so `raim_fde` is called directly: its
 * exclusion is the reference, not whether demo5 would have run it. Each case
 * also records whether `estpos` fails on the whole faulted epoch from the
 * approximate position, as `pntpos` runs it; when it does, `pntpos` itself (RAIM
 * FDE on) must return `raim_fde`'s solution bit for bit.
 *
 * `raim_fde` keeps no record of which satellite it removed or of the final
 * least-squares state. A replay of its loop, running the unmodified `estpos` in
 * the same order with the same shared solution state, recovers both; the replay's
 * result must equal `raim_fde`'s bit for bit (position, clock, used satellites) or
 * the run fails. Each case records the faults, the excluded satellite, every
 * candidate's status, used-satellite count, residual RMS and final state and
 * step, and the chosen solution in the same form as a selection case. No message is recorded: `raim_fde` copies its candidate
 * buffer, which a successful `estpos` leaves unwritten, into `msg`.
 * ------------------------------------------------------------------------- */

/* +299792.458 m is a 1 ms receiver-clock-sized blunder. */
#define FDE_BIAS_COUNT 4
static const double fde_biases_m[FDE_BIAS_COUNT] = {5000.0, 300.0, 30.0, 299792.458};
#define FDE_PAIR_BIAS_COUNT 3
static const double fde_pair_biases_m[FDE_PAIR_BIAS_COUNT] = {5000.0, 300.0, 30.0};

typedef struct {
    int valid;
    int state_valid;
    int columns;
    double design[(MAXOBS + NX) * NX];
    double covariance[NX * NX];
    double receiver_state[NX];
    double step[NX];
    double tracked[NX];
} lsq_capture_t;

static lsq_capture_t fde_best_capture;

static void save_capture(lsq_capture_t *capture)
{
    capture->valid = captured_lsq_valid;
    capture->state_valid = captured_lsq_state_valid;
    capture->columns = captured_lsq_columns;
    memcpy(capture->design, captured_lsq_design, sizeof(capture->design));
    memcpy(capture->covariance, captured_lsq_covariance, sizeof(capture->covariance));
    memcpy(capture->receiver_state, captured_lsq_receiver_state,
           sizeof(capture->receiver_state));
    memcpy(capture->step, captured_lsq_step, sizeof(capture->step));
    memcpy(capture->tracked, tracked_receiver_state, sizeof(capture->tracked));
}

static void restore_capture(const lsq_capture_t *capture)
{
    captured_lsq_valid = capture->valid;
    captured_lsq_state_valid = capture->state_valid;
    captured_lsq_columns = capture->columns;
    memcpy(captured_lsq_design, capture->design, sizeof(capture->design));
    memcpy(captured_lsq_covariance, capture->covariance, sizeof(capture->covariance));
    memcpy(captured_lsq_receiver_state, capture->receiver_state,
           sizeof(capture->receiver_state));
    memcpy(captured_lsq_step, capture->step, sizeof(capture->step));
    memcpy(tracked_receiver_state, capture->tracked, sizeof(capture->tracked));
}

static void snr_like_pntpos(const obsd_t *obs, int n, ssat_t *ssat)
{
    int i;
    memset(ssat, 0, sizeof(ssat_t) * MAXSAT);
    for (i = 0; i < n; i++) ssat[obs[i].sat - 1].snr_rover[0] = obs[i].SNR[0];
}

/* Check the replayed final least-squares state against the returned solution, as
 * the selection oracle does after `pntpos`. */
static int verify_capture(const sol_t *sol)
{
    int k;
    double tracked_clock_s = tracked_receiver_state[3] / CLIGHT;
    if (!captured_lsq_valid || !captured_lsq_state_valid) return 0;
    for (k = 0; k < 3; k++) {
        if (memcmp(&tracked_receiver_state[k], &sol->rr[k], sizeof(double)) != 0) return 0;
    }
    if (memcmp(&tracked_clock_s, &sol->dtr[0], sizeof(double)) != 0) return 0;
    for (k = 0; k < NX; k++) {
        double replayed = captured_lsq_receiver_state[k] + captured_lsq_step[k];
        if (memcmp(&replayed, &tracked_receiver_state[k], sizeof(double)) != 0) return 0;
    }
    return (float)captured_lsq_covariance[0] == sol->qr[0] &&
           (float)captured_lsq_covariance[1 + NX] == sol->qr[1] &&
           (float)captured_lsq_covariance[2 + 2 * NX] == sol->qr[2] &&
           (float)captured_lsq_covariance[1] == sol->qr[3] &&
           (float)captured_lsq_covariance[2 + NX] == sol->qr[4] &&
           (float)captured_lsq_covariance[2] == sol->qr[5];
}

static void fde_case(const obsd_t *obs, int n, const int *faulted, int fault_count,
                     double bias_m, const double *guess, const nav_t *nav,
                     const prcopt_t *opt, int first)
{
    static obsd_t obs_e[MAXOBS];
    static double rs[MAXOBS * 6], dts[MAXOBS * 2], var[MAXOBS];
    static double rs_e[MAXOBS * 6], dts_e[MAXOBS * 2], vare_e[MAXOBS];
    static double azel_e[MAXOBS * 2], resp_e[MAXOBS];
    static double azel_r[MAXOBS * 2], resp_r[MAXOBS];
    static int svh[MAXOBS], svh_e[MAXOBS], vsat_e[MAXOBS], vsat_r[MAXOBS];
    static int best_vsat[MAXOBS];
    static double candidate_rms[MAXOBS], candidate_rr[MAXOBS * 3], candidate_clock[MAXOBS];
    static double candidate_step[MAXOBS * 4];
    static int candidate_status[MAXOBS], candidate_nvsat[MAXOBS];
    static ssat_t ssat[MAXSAT], ssat_out[MAXSAT];
    sol_t sol_e = {{0}}, best_sol = {{0}}, sol_r = {{0}}, sol_full = {{0}}, sol_p = {{0}};
    static double azel_full[MAXOBS * 2], resp_full[MAXOBS];
    static int vsat_full[MAXOBS];
    static ssat_t ssat_p[MAXSAT];
    char msg_full[128] = "", msg_p[128] = "";
    int full_stat, pntpos_stat = -1;
    double rms = 100.0, rms_e, ep[6], receiver_geodetic[3];
    char msg_e[128] = "", msg_r[128] = "", id[8];
    int i, j, k, nvsat, best = -1, stat, used = 0;

    snr_like_pntpos(obs, n, ssat);
    satposs(obs[0].time, obs, n, nav, opt->sateph, rs, dts, var, svh);

    /* `estpos` on the whole faulted epoch from the approximate position, as
     * `pntpos` runs it before deciding whether to call `raim_fde`. */
    for (k = 0; k < 3; k++) sol_full.rr[k] = guess[k];
    full_stat = estpos(obs, n, rs, dts, var, svh, nav, opt, ssat, &sol_full, azel_full,
                       vsat_full, resp_full, msg_full);

    /* The loop of `raim_fde`, replayed. */
    for (i = 0; i < n; i++) {
        for (j = k = 0; j < n; j++) {
            if (j == i) continue;
            obs_e[k] = obs[j];
            matcpy(rs_e + 6 * k, rs + 6 * j, 6, 1);
            matcpy(dts_e + 2 * k, dts + 2 * j, 2, 1);
            vare_e[k] = var[j];
            svh_e[k++] = svh[j];
        }
        memset(tracked_receiver_state, 0, sizeof(tracked_receiver_state));
        for (k = 0; k < 3; k++) tracked_receiver_state[k] = sol_e.rr[k];
        captured_lsq_valid = 0;
        captured_lsq_state_valid = 0;
        candidate_rms[i] = 0.0;
        candidate_nvsat[i] = 0;
        if (!estpos(obs_e, n - 1, rs_e, dts_e, vare_e, svh_e, nav, opt, ssat, &sol_e, azel_e,
                    vsat_e, resp_e, msg_e)) {
            candidate_status[i] = 0;
            continue;
        }
        for (j = nvsat = 0, rms_e = 0.0; j < n - 1; j++) {
            if (!vsat_e[j]) continue;
            rms_e += SQR(resp_e[j]);
            nvsat++;
        }
        candidate_nvsat[i] = nvsat;
        for (k = 0; k < 3; k++) candidate_rr[i * 3 + k] = sol_e.rr[k];
        candidate_clock[i] = sol_e.dtr[0] * CLIGHT;
        for (k = 0; k < 4; k++) candidate_step[i * 4 + k] = captured_lsq_step[k];
        if (!captured_lsq_state_valid) {
            fprintf(stderr, "missing candidate least-squares step\n");
            exit(1);
        }
        if (nvsat < 5) {
            candidate_status[i] = 1;
            continue;
        }
        rms_e = sqrt(rms_e / nvsat);
        candidate_status[i] = 2;
        candidate_rms[i] = rms_e;
        if (rms_e > rms) continue;
        for (j = k = 0; j < n; j++) {
            if (j == i) continue;
            best_vsat[j] = vsat_e[k++];
        }
        best_vsat[i] = 0;
        best = i;
        best_sol = sol_e;
        rms = rms_e;
        save_capture(&fde_best_capture);
    }

    /* RTKLIB's own `raim_fde` on the same inputs. */
    sol_r.time = obs[0].time;
    memset(azel_r, 0, sizeof(azel_r));
    memset(vsat_r, 0, sizeof(vsat_r));
    stat = raim_fde(obs, n, rs, dts, var, svh, nav, opt, ssat, &sol_r, azel_r, vsat_r,
                    resp_r, msg_r);
    if (stat != (best >= 0)) {
        fprintf(stderr, "raim_fde status differs from its replay\n");
        exit(1);
    }
    if (stat) {
        if (memcmp(sol_r.rr, best_sol.rr, 3 * sizeof(double)) != 0 ||
            memcmp(&sol_r.dtr[0], &best_sol.dtr[0], sizeof(double)) != 0) {
            fprintf(stderr, "raim_fde solution differs from its replay\n");
            exit(1);
        }
        for (j = 0; j < n; j++) {
            if (vsat_r[j] != best_vsat[j]) {
                fprintf(stderr, "raim_fde used satellites differ from its replay\n");
                exit(1);
            }
        }
    }
    if (!full_stat) {
        /* demo5's own path: `pntpos` with RAIM FDE on reaches `raim_fde`. */
        prcopt_t opt_p = *opt;
        opt_p.posopt[4] = 1;
        memset(ssat_p, 0, sizeof(ssat_p));
        for (k = 0; k < 3; k++) sol_p.rr[k] = guess[k];
        pntpos_stat = pntpos(obs, n, nav, &opt_p, &sol_p, NULL, ssat_p, msg_p);
        if (pntpos_stat != stat ||
            (stat && (memcmp(sol_p.rr, best_sol.rr, 3 * sizeof(double)) != 0 ||
                      memcmp(&sol_p.dtr[0], &best_sol.dtr[0], sizeof(double)) != 0))) {
            fprintf(stderr, "pntpos RAIM FDE differs from raim_fde\n");
            exit(1);
        }
    }
    if (stat) {
        restore_capture(&fde_best_capture);
        if (!verify_capture(&best_sol)) {
            fprintf(stderr, "replayed raim_fde least-squares capture does not replay\n");
            exit(1);
        }
    }

    time2epoch(obs[0].time, ep);
    printf("%s  {\"epoch\": [%d, %d, %d, %d, %d, %.7f], \"faults\": [",
           first ? "" : ",\n", (int)ep[0], (int)ep[1], (int)ep[2], (int)ep[3], (int)ep[4],
           ep[5]);
    for (k = 0; k < fault_count; k++) {
        satno2id(obs[faulted[k]].sat, id);
        printf("%s{\"sat\": \"%s\", \"bias_m\": %.17g}", k ? ", " : "", id, bias_m);
    }
    printf("], \"mode\": \"%s\", \"full_estpos_stat\": %d, \"pntpos_stat\": %d, \"stat\": %d, ",
           fault_count == 2 ? "two_fault"
           : bias_m == 30.0 ? "small_bias"
           : bias_m == 299792.458 ? "one_millisecond_blunder"
                                  : "single_fault",
           full_stat, pntpos_stat, stat);
    printf("\"candidates\": [");
    for (i = 0; i < n; i++) {
        satno2id(obs[i].sat, id);
        printf("%s{\"sat\": \"%s\", \"status\": \"%s\", \"nvsat\": %d, \"rms_m\": %.17g",
               i ? ", " : "", id,
               candidate_status[i] == 2 ? "solved"
               : candidate_status[i] == 1 ? "too_few_satellites"
                                          : "failed",
               candidate_nvsat[i], candidate_rms[i]);
        if (candidate_status[i]) {
            printf(", \"position_m\": ");
            print_position(candidate_rr + i * 3);
            printf(", \"clock_m\": %.17g, \"lsq_step\": [%.17g, %.17g, %.17g, %.17g]",
                   candidate_clock[i], candidate_step[i * 4], candidate_step[i * 4 + 1],
                   candidate_step[i * 4 + 2], candidate_step[i * 4 + 3]);
        }
        printf("}");
    }
    printf("]");
    if (!stat) {
        printf(", \"excluded\": null}");
        return;
    }
    satno2id(obs[best].sat, id);
    printf(", \"excluded\": \"%s\", \"position_m\": ", id);
    print_position(best_sol.rr);
    ecef2pos(best_sol.rr, receiver_geodetic);
    printf(", \"geodetic_rad_m\": [%.17g, %.17g, %.17g]", receiver_geodetic[0],
           receiver_geodetic[1], receiver_geodetic[2]);
    printf(", \"qr_m2\": [%.9g, %.9g, %.9g, %.9g, %.9g, %.9g]", best_sol.qr[0],
           best_sol.qr[1], best_sol.qr[2], best_sol.qr[3], best_sol.qr[4], best_sol.qr[5]);
    print_captured_lsq();
    printf(", \"clock_m\": %.17g, \"used\": [", best_sol.dtr[0] * CLIGHT);
    for (j = k = 0; j < n; j++) {
        if (j == best) continue;
        obs_e[k++] = obs[j];
    }
    snr_like_pntpos(obs_e, n - 1, ssat_out);
    for (j = 0; j < n; j++) {
        if (vsat_r[j]) ssat_out[obs[j].sat - 1].vs = 1;
    }
    for (k = 0; k < MAXSAT; k++) {
        if (!ssat_out[k].vs) continue;
        satno2id(k + 1, id);
        printf("%s\"%s\"", used++ ? ", " : "", id);
    }
    printf("]");
    print_used_states(obs_e, n - 1, obs[0].time, nav, opt, &best_sol, ssat_out);
    printf("}");
}

static int run_fde(int argc, char **argv)
{
    obs_t obs = {0};
    nav_t nav = {0};
    sta_t sta = {{0}};
    prcopt_t opt = prcopt_default;
    int i, k, n, epoch_index, stride, first = 1;

    if (argc != 8 || (stride = atoi(argv[7])) <= 0) {
        fprintf(stderr, "usage: %s fde <label> <obs> <nav> <iono 0|1> <tropo 0|1> <stride>\n",
                argv[0]);
        return 2;
    }
    if (readrnx(argv[3], 1, "-GL1C", &obs, NULL, &sta) <= 0) {
        fprintf(stderr, "cannot read %s\n", argv[3]);
        return 1;
    }
    if (readrnx(argv[4], 1, "", NULL, &nav, NULL) <= 0) {
        fprintf(stderr, "cannot read %s\n", argv[4]);
        return 1;
    }
    sortobs(&obs);
    uniqnav(&nav);

    opt.mode = PMODE_SINGLE;
    opt.navsys = SYS_GPS;
    opt.nf = 1;
    opt.elmin = 10.0 * D2R;
    opt.sateph = EPHOPT_BRDC;
    opt.ionoopt = argv[5][0] == '1' ? IONOOPT_BRDC : IONOOPT_OFF;
    opt.tropopt = argv[6][0] == '1' ? TROPOPT_SAAS : TROPOPT_OFF;
    opt.posopt[4] = 1;

    printf("{\"label\": \"%s\", \"ionosphere\": %s, \"troposphere\": %s, \"stride\": %d,\n",
           argv[2], opt.ionoopt == IONOOPT_BRDC ? "true" : "false",
           opt.tropopt == TROPOPT_SAAS ? "true" : "false", stride);
    printf(" \"guess\": ");
    print_position(sta.pos);
    printf(", \"biases_m\": [");
    for (k = 0; k < FDE_BIAS_COUNT; k++) printf("%s%.17g", k ? ", " : "", fde_biases_m[k]);
    printf("], \"pair_biases_m\": [");
    for (k = 0; k < FDE_PAIR_BIAS_COUNT; k++)
        printf("%s%.17g", k ? ", " : "", fde_pair_biases_m[k]);
    printf("],\n \"cases\": [\n");

    for (i = epoch_index = 0; i < obs.n; i += n, epoch_index++) {
        obsd_t epoch[MAXOBS];
        sol_t sol = {{0}};
        ssat_t ssat[MAXSAT];
        char msg[128] = "";
        int m = 0, faulted, b;
        for (n = 1; i + n < obs.n && timediff(obs.data[i + n].time, obs.data[i].time) == 0.0;
             n++) {
        }
        if (epoch_index % stride != 0) continue;
        for (k = 0; k < n; k++) {
            if (satsys(obs.data[i + k].sat, NULL) != SYS_GPS || m >= MAXOBS) continue;
            epoch[m++] = obs.data[i + k];
        }
        if (m < 6) {
            fprintf(stderr, "epoch %d has %d GPS observations, fewer than raim_fde needs\n",
                    epoch_index, m);
            return 1;
        }
        memset(ssat, 0, sizeof(ssat));
        for (k = 0; k < 3; k++) sol.rr[k] = sta.pos[k];
        if (!pntpos(epoch, m, &nav, &opt, &sol, NULL, ssat, msg)) {
            fprintf(stderr, "clean epoch %d does not solve: %s\n", epoch_index, msg);
            return 1;
        }
        for (faulted = 0; faulted < m; faulted++) {
            int pair[2], next;
            if (!ssat[epoch[faulted].sat - 1].vs) continue;
            if (epoch[faulted].P[0] == 0.0) {
                fprintf(stderr, "used satellite without an L1 pseudorange\n");
                return 1;
            }
            for (b = 0; b < FDE_BIAS_COUNT; b++) {
                obsd_t faulted_epoch[MAXOBS];
                memcpy(faulted_epoch, epoch, sizeof(obsd_t) * m);
                faulted_epoch[faulted].P[0] += fde_biases_m[b];
                fde_case(faulted_epoch, m, &faulted, 1, fde_biases_m[b], sta.pos, &nav, &opt,
                         first);
                first = 0;
            }
            /* The next used satellite, wrapping round. */
            for (next = (faulted + 1) % m; !ssat[epoch[next].sat - 1].vs;
                 next = (next + 1) % m) {
            }
            pair[0] = faulted;
            pair[1] = next;
            for (b = 0; b < FDE_PAIR_BIAS_COUNT; b++) {
                obsd_t faulted_epoch[MAXOBS];
                memcpy(faulted_epoch, epoch, sizeof(obsd_t) * m);
                faulted_epoch[pair[0]].P[0] += fde_pair_biases_m[b];
                faulted_epoch[pair[1]].P[0] += fde_pair_biases_m[b];
                fde_case(faulted_epoch, m, pair, 2, fde_pair_biases_m[b], sta.pos, &nav, &opt,
                         first);
                first = 0;
            }
        }
    }
    printf("\n ]}\n");
    freeobs(&obs);
    freenav(&nav, 0xFF);
    return 0;
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

    if (argc >= 2 && strcmp(argv[1], "fde") == 0) return run_fde(argc, argv);
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
            double receiver_geodetic[3];
            char msg[128] = "";
            int stat, used = 0;
            memset(ssat, 0, sizeof(ssat));
            for (k = 0; k < 3; k++) sol.rr[k] = guesses[j][k];
            memset(tracked_receiver_state, 0, sizeof(tracked_receiver_state));
            for (k = 0; k < 3; k++) tracked_receiver_state[k] = guesses[j][k];
            captured_lsq_valid = 0;
            captured_lsq_state_valid = 0;
            stat = pntpos(epoch, m, &nav, &opt, &sol, NULL, ssat, msg);
            if (stat && (!captured_lsq_valid || !captured_lsq_state_valid)) {
                fprintf(stderr, "missing final RTKLIB positioning least-squares capture\n");
                return 1;
            }
            if (stat) {
                double tracked_clock_s = tracked_receiver_state[3] / CLIGHT;
                for (k = 0; k < 3; k++) {
                    if (memcmp(&tracked_receiver_state[k], &sol.rr[k], sizeof(double)) != 0) {
                        fprintf(stderr, "passive RTKLIB receiver replay differs from sol.rr\n");
                        return 1;
                    }
                }
                if (memcmp(&tracked_clock_s, &sol.dtr[0], sizeof(double)) != 0) {
                    fprintf(stderr, "passive RTKLIB clock replay differs from sol.dtr\n");
                    return 1;
                }
                for (k = 0; k < NX; k++) {
                    double replayed = captured_lsq_receiver_state[k] + captured_lsq_step[k];
                    if (memcmp(&replayed, &tracked_receiver_state[k], sizeof(double)) != 0) {
                        fprintf(stderr, "captured final RTKLIB step does not replay\n");
                        return 1;
                    }
                }
            }
            if (stat && ((float)captured_lsq_covariance[0] != sol.qr[0] ||
                         (float)captured_lsq_covariance[1 + NX] != sol.qr[1] ||
                         (float)captured_lsq_covariance[2 + 2 * NX] != sol.qr[2] ||
                         (float)captured_lsq_covariance[1] != sol.qr[3] ||
                         (float)captured_lsq_covariance[2 + NX] != sol.qr[4] ||
                         (float)captured_lsq_covariance[2] != sol.qr[5])) {
                fprintf(stderr, "captured RTKLIB least-squares covariance is not sol.qr\n");
                return 1;
            }
            printf("%s  {\"epoch\": [%d, %d, %d, %d, %d, %.7f], \"guess\": \"%s\", \"stat\": %d, ",
                   first_case ? "" : ",\n", (int)ep[0], (int)ep[1], (int)ep[2], (int)ep[3],
                   (int)ep[4], ep[5], names[j], stat);
            first_case = 0;
            printf("\"position_m\": ");
            print_position(sol.rr);
            if (stat) {
                ecef2pos(sol.rr, receiver_geodetic);
                printf(", \"geodetic_rad_m\": [%.17g, %.17g, %.17g]",
                       receiver_geodetic[0], receiver_geodetic[1], receiver_geodetic[2]);
            }
            printf(", \"qr_m2\": [%.9g, %.9g, %.9g, %.9g, %.9g, %.9g]", sol.qr[0], sol.qr[1],
                   sol.qr[2], sol.qr[3], sol.qr[4], sol.qr[5]);
            if (stat) print_captured_lsq();
            printf(", \"clock_m\": %.17g, \"used\": [", sol.dtr[0] * CLIGHT);
            for (k = 0; k < MAXSAT; k++) {
                char id[8];
                if (!ssat[k].vs) continue;
                satno2id(k + 1, id);
                printf("%s\"%s\"", used++ ? ", " : "", id);
            }
            printf("]");
            print_used_states(epoch, m, epoch[0].time, &nav, &opt, &sol, ssat);
            printf(", \"msg\": \"%s\"}", msg);
        }
    }
    printf("\n ]}\n");
    freeobs(&obs);
    freenav(&nav, 0xFF);
    return 0;
}
