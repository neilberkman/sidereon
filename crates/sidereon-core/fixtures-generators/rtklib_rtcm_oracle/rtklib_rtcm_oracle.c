/*
 * RTKLIB RTCM 3 oracle: decodes RTCM 3 streams with RTKLIB and prints what
 * RTKLIB reads from each frame, and writes RTCM 3 streams with RTKLIB's
 * encoder from real observations, ephemerides and SSR corrections.
 * `generate.sh` builds it against RTKLIB and writes the fixtures under
 * `tests/fixtures/rtcm/families/` that `tests/rtcm_family_rtklib_oracle.rs`
 * reads.
 *
 * RTKLIB is used unmodified (https://github.com/rtklibexplorer/RTKLIB, branch
 * demo5, commit 75a2e56275485b21a67bd35bc94bbeb8936e1a74).
 *
 * usage:
 *   rtklib_rtcm_oracle decode <stream> <gps week> <tow s>
 *       One JSON object per CRC-valid frame, one per line, in stream order:
 *       what `decode_rtcm3` returns and what it stores. Every floating-point
 *       value is printed as the hex digits of its IEEE 754 bits.
 *   rtklib_rtcm_oracle encode-msm <stream> <gps week> <tow s> <epochs> <out>
 *       Reads the MSM messages of <stream> and writes after each one, for
 *       the first <epochs> epochs, the MSM1, MSM2, MSM3 and MSM4 of its system
 *       that RTKLIB encodes from the observations it read from it.
 *   rtklib_rtcm_oracle encode-legacy <stream> <gps week> <tow s> <epochs> <out>
 *       Reads the 1004/1012 observation epochs of <stream> and writes, for
 *       the first <epochs> of them, 1001, 1002, 1003, 1004, 1009, 1010, 1011
 *       and 1012.
 *   rtklib_rtcm_oracle encode-1041 <rinex nav> <out>
 *       Writes one 1041 frame per NavIC ephemeris of the RINEX navigation
 *       file, in file order.
 *   rtklib_rtcm_oracle encode-4076 <stream> <gps week> <tow s> <epochs> <out>
 *       Reads the RTCM SSR corrections of <stream> and writes, for each of the
 *       first <epochs> SSR epochs, the 4076 IGS SSR subtypes 21..27, 41..47, 61..67, 81..87, 101..107 and
 *       121..127. The stream carries no phase biases, so each satellite of a
 *       phase-bias subtype is given one per signal it has a code bias for:
 *       the code bias times 1.37 m/m, with the yaw angle and rate set from the
 *       satellite number. Every other value is the stream's.
 *
 * <gps week> and <tow s> set RTKLIB's receiver time before the first frame,
 * as `convbin -tr` does; RTKLIB places each epoch in the week (GLONASS: the
 * day) nearest it.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rtklib.h"

/* The RTCM 3 frame preamble (rtcm.c keeps its own copy private). */
#define PREAMBLE 0xD3

/* decode_rtcm3 is external in rtcm3.c without a declaration in rtklib.h. */
extern int decode_rtcm3(rtcm_t *rtcm);

/* Progress callbacks the RTKLIB library expects its application to define. */
extern int showmsg(const char *format, ...) { (void)format; return 0; }
extern void settspan(gtime_t ts, gtime_t te) { (void)ts; (void)te; }
extern void settime(gtime_t time) { (void)time; }

static unsigned long long dbits(double x)
{
    uint64_t u;
    memcpy(&u, &x, sizeof u);
    return (unsigned long long)u;
}

static unsigned int fbits(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return (unsigned int)u;
}

static void put_d(const char *key, double x) { printf("\"%s\":\"%016llx\"", key, dbits(x)); }

static void put_time(const char *key, gtime_t t)
{
    printf("\"%s\":[%lld,\"%016llx\"]", key, (long long)t.time, dbits(t.sec));
}

static uint8_t *read_file(const char *path, long *len)
{
    FILE *fp = fopen(path, "rb");
    uint8_t *buf;
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(fp, 0, SEEK_END);
    *len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buf = malloc(*len > 0 ? (size_t)*len : 1);
    if (fread(buf, 1, (size_t)*len, fp) != (size_t)*len) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    fclose(fp);
    return buf;
}

/* Find the next CRC-valid frame at or after *pos; return its offset or -1. */
static long next_frame(const uint8_t *buf, long len, long *pos, int *frame_len)
{
    while (*pos + 6 <= len) {
        long at = *pos;
        int body;
        if (buf[at] != PREAMBLE) {
            (*pos)++;
            continue;
        }
        body = ((buf[at + 1] & 3) << 8) | buf[at + 2];
        if (at + 6 + body > len) return -1;
        if (rtk_crc24q(buf + at, 3 + body) != getbitu(buf + at, (3 + body) * 8, 24)) {
            (*pos)++;
            continue;
        }
        *frame_len = 6 + body;
        *pos = at + 6 + body;
        return at;
    }
    return -1;
}

/* Load one frame into rtcm's buffer and decode it. */
static int decode_one(rtcm_t *rtcm, const uint8_t *frame, int frame_len)
{
    memcpy(rtcm->buff, frame, (size_t)frame_len);
    rtcm->len = frame_len - 3;
    rtcm->nbyte = 0;
    return decode_rtcm3(rtcm);
}

static int is_obs_type(int type)
{
    return (1001 <= type && type <= 1004) || (1009 <= type && type <= 1012) ||
           (1071 <= type && type <= 1137 && type % 10 >= 1 && type % 10 <= 7);
}

static int is_ssr_type(int type)
{
    return (1057 <= type && type <= 1068) || (1240 <= type && type <= 1263) ||
           type == 4076;
}

static void print_obs(const rtcm_t *rtcm)
{
    int i, j, first = 1;
    char id[8];
    printf(",\"obs\":[");
    for (i = 0; i < rtcm->obs.n; i++) {
        const obsd_t *d = rtcm->obs.data + i;
        int firstsig = 1;
        satno2id(d->sat, id);
        printf("%s{\"sat\":\"%s\",\"sig\":[", first ? "" : ",", id);
        first = 0;
        for (j = 0; j < NFREQ + NEXOBS; j++) {
            if (!d->code[j]) continue;
            printf("%s{\"slot\":%d,\"code\":\"%s\",", firstsig ? "" : ",", j,
                   code2obs(d->code[j]));
            firstsig = 0;
            put_d("P", d->P[j]);
            printf(",");
            put_d("L", d->L[j]);
            printf(",\"D\":\"%08x\",\"SNR\":\"%08x\",\"LLI\":%d}", fbits(d->D[j]),
                   fbits(d->SNR[j]), d->LLI[j]);
        }
        printf("]}");
    }
    printf("]");
}

static void print_eph(const eph_t *e)
{
    char id[8];
    satno2id(e->sat, id);
    printf(",\"eph\":{\"sat\":\"%s\",\"iode\":%d,\"iodc\":%d,\"sva\":%d,\"svh\":%d,"
           "\"week\":%d,\"code\":%d,\"flag\":%d,",
           id, e->iode, e->iodc, e->sva, e->svh, e->week, e->code, e->flag);
    put_time("toe", e->toe); printf(",");
    put_time("toc", e->toc); printf(",");
    put_d("A", e->A); printf(",");
    put_d("e", e->e); printf(",");
    put_d("i0", e->i0); printf(",");
    put_d("OMG0", e->OMG0); printf(",");
    put_d("omg", e->omg); printf(",");
    put_d("M0", e->M0); printf(",");
    put_d("deln", e->deln); printf(",");
    put_d("OMGd", e->OMGd); printf(",");
    put_d("idot", e->idot); printf(",");
    put_d("crc", e->crc); printf(",");
    put_d("crs", e->crs); printf(",");
    put_d("cuc", e->cuc); printf(",");
    put_d("cus", e->cus); printf(",");
    put_d("cic", e->cic); printf(",");
    put_d("cis", e->cis); printf(",");
    put_d("toes", e->toes); printf(",");
    put_d("f0", e->f0); printf(",");
    put_d("f1", e->f1); printf(",");
    put_d("f2", e->f2); printf(",");
    put_d("tgd0", e->tgd[0]);
    printf("}");
}

static void print_ssr(const rtcm_t *rtcm)
{
    int i, k, first = 1;
    char id[8];
    printf(",\"ssr\":[");
    for (i = 0; i < MAXSAT; i++) {
        const ssr_t *s = rtcm->ssr + i;
        int any = 0, firstcode;
        for (k = 0; k < 6; k++) any |= s->t0[k].time != 0;
        if (!any) continue;
        satno2id(i + 1, id);
        printf("%s{\"sat\":\"%s\",\"t0\":[", first ? "" : ",", id);
        first = 0;
        for (k = 0; k < 6; k++) {
            printf("%s[%lld,\"%016llx\"]", k ? "," : "", (long long)s->t0[k].time,
                   dbits(s->t0[k].sec));
        }
        printf("],\"udi\":[");
        for (k = 0; k < 6; k++) printf("%s\"%016llx\"", k ? "," : "", dbits(s->udi[k]));
        printf("],\"iod\":[");
        for (k = 0; k < 6; k++) printf("%s%d", k ? "," : "", s->iod[k]);
        printf("],\"iode\":%d,\"iodcrc\":%d,\"ura\":%d,\"refd\":%d,\"deph\":[", s->iode,
               s->iodcrc, s->ura, s->refd);
        for (k = 0; k < 3; k++) printf("%s\"%016llx\"", k ? "," : "", dbits(s->deph[k]));
        printf("],\"ddeph\":[");
        for (k = 0; k < 3; k++) printf("%s\"%016llx\"", k ? "," : "", dbits(s->ddeph[k]));
        printf("],\"dclk\":[");
        for (k = 0; k < 3; k++) printf("%s\"%016llx\"", k ? "," : "", dbits(s->dclk[k]));
        printf("],");
        put_d("hrclk", s->hrclk);
        printf(",\"cbias\":{");
        for (k = 0, firstcode = 1; k < MAXCODE; k++) {
            if (s->cbias[k] == 0.0f) continue;
            printf("%s\"%s\":\"%08x\"", firstcode ? "" : ",", code2obs((uint8_t)(k + 1)),
                   fbits(s->cbias[k]));
            firstcode = 0;
        }
        printf("},\"pbias\":{");
        for (k = 0, firstcode = 1; k < MAXCODE; k++) {
            if (s->pbias[k] == 0.0) continue;
            printf("%s\"%s\":\"%016llx\"", firstcode ? "" : ",", code2obs((uint8_t)(k + 1)),
                   dbits(s->pbias[k]));
            firstcode = 0;
        }
        printf("},");
        put_d("yaw_ang", s->yaw_ang);
        printf(",");
        put_d("yaw_rate", s->yaw_rate);
        printf("}");
    }
    printf("]");
}

static void init_at(rtcm_t *rtcm, int week, double tow)
{
    init_rtcm(rtcm);
    rtcm->time = gpst2time(week, tow);
    strcpy(rtcm->opt, "-EPHALL");
}

static int decode_stream(const char *path, int week, double tow)
{
    static rtcm_t rtcm;
    long len, pos = 0, at;
    int frame_len;
    uint8_t *buf = read_file(path, &len);

    init_at(&rtcm, week, tow);
    while ((at = next_frame(buf, len, &pos, &frame_len)) >= 0) {
        int type = getbitu(buf + at, 24, 12), ret, wk;
        double t;
        if (is_obs_type(type)) rtcm.obs.n = 0;
        if (is_ssr_type(type)) memset(rtcm.ssr, 0, sizeof rtcm.ssr);
        rtcm.ephsat = 0;
        ret = decode_one(&rtcm, buf + at, frame_len);
        t = time2gpst(rtcm.time, &wk);
        printf("{\"offset\":%ld,\"type\":%d,\"ret\":%d,\"staid\":%d,\"week\":%d,", at, type,
               ret, rtcm.staid, wk);
        put_d("tow", t);
        if (is_obs_type(type)) print_obs(&rtcm);
        if (type == 1041 && ret == 2 && rtcm.ephsat > 0) print_eph(rtcm.nav.eph + rtcm.ephsat - 1);
        if (type == 1230 && ret == 5) {
            printf(",\"glo_cp\":{\"align\":%d,\"bias\":[", rtcm.sta.glo_cp_align);
            for (int k = 0; k < 4; k++) {
                printf("%s\"%016llx\"", k ? "," : "", dbits(rtcm.sta.glo_cp_bias[k]));
            }
            printf("]}");
        }
        if (is_ssr_type(type)) print_ssr(&rtcm);
        printf("}\n");
    }
    free(buf);
    return 0;
}

/* Write RTKLIB's frame for (type, subtype) from `rtcm` to `out`. */
static int emit(rtcm_t *rtcm, int type, int subtype, int sync, FILE *out)
{
    if (!gen_rtcm3(rtcm, type, subtype, sync)) return 0;
    if (fwrite(rtcm->buff, 1, (size_t)rtcm->nbyte, out) != (size_t)rtcm->nbyte) {
        fprintf(stderr, "write failed\n");
        exit(2);
    }
    return 1;
}

/* Point an encoding rtcm_t at the observations and navigation data `in` read. */
static void share(rtcm_t *out, const rtcm_t *in)
{
    out->time = in->time;
    out->staid = in->staid;
    out->obs.n = in->obs.n;
    out->obs.data = in->obs.data;
    out->nav = in->nav;
}

static int has_system(const rtcm_t *rtcm, int sys)
{
    for (int i = 0; i < rtcm->obs.n; i++) {
        if (satsys(rtcm->obs.data[i].sat, NULL) == sys) return 1;
    }
    return 0;
}

static int encode_msm(const char *path, int week, double tow, int epochs, const char *outpath)
{
    static rtcm_t in, enc[5];
    static const int systems[] = {SYS_GPS, SYS_GLO, SYS_GAL, SYS_SBS, SYS_QZS, SYS_CMP, SYS_IRN};
    long len, pos = 0, at;
    int frame_len, done = 0;
    uint8_t *buf = read_file(path, &len);
    FILE *out = fopen(outpath, "wb");

    init_at(&in, week, tow);
    for (int k = 1; k <= 4; k++) init_rtcm(enc + k);
    while (done < epochs && (at = next_frame(buf, len, &pos, &frame_len)) >= 0) {
        int type = getbitu(buf + at, 24, 12);
        int base = type - type % 10;
        /* The multiple message bit: 0 on the last message of an epoch. */
        int sync = getbitu(buf + at, 24 + 12 + 12 + 30, 1);
        decode_one(&in, buf + at, frame_len);
        if (!(1071 <= type && type <= 1137)) continue;
        /* Right after its own message, RTKLIB holds this system's
         * observations at this system's epoch: MSM1..MSM4 of it. */
        if (has_system(&in, systems[(base - 1070) / 10])) {
            for (int k = 1; k <= 4; k++) {
                share(enc + k, &in);
                emit(enc + k, base + k, 0, k < 4 ? 1 : sync, out);
            }
        }
        if (!sync) done++;
    }
    fclose(out);
    free(buf);
    return 0;
}

static int encode_legacy(const char *path, int week, double tow, int epochs, const char *outpath)
{
    static rtcm_t in, enc[8];
    static const int types[] = {1001, 1002, 1003, 1004, 1009, 1010, 1011, 1012};
    long len, pos = 0, at;
    int frame_len, done = 0;
    uint8_t *buf = read_file(path, &len);
    FILE *out = fopen(outpath, "wb");

    init_at(&in, week, tow);
    for (int k = 0; k < 8; k++) init_rtcm(enc + k);
    while (done < epochs && (at = next_frame(buf, len, &pos, &frame_len)) >= 0) {
        int type = getbitu(buf + at, 24, 12);
        int ret = decode_one(&in, buf + at, frame_len);
        if (!(type == 1004 || type == 1012) || ret != 1) continue;
        for (int k = 0; k < 8; k++) {
            share(enc + k, &in);
            emit(enc + k, types[k], 0, k != 7, out);
        }
        done++;
    }
    fclose(out);
    free(buf);
    return 0;
}

static int encode_1041(const char *navpath, const char *outpath)
{
    static rtcm_t enc;
    obs_t obs = {0};
    nav_t nav = {0};
    sta_t sta = {{0}};
    FILE *out = fopen(outpath, "wb");

    init_rtcm(&enc);
    if (readrnx(navpath, 0, "", &obs, &nav, &sta) <= 0) {
        fprintf(stderr, "cannot read %s\n", navpath);
        return 2;
    }
    for (int i = 0; i < nav.n; i++) {
        const eph_t *e = nav.eph + i;
        if (satsys(e->sat, NULL) != SYS_IRN) continue;
        enc.nav.eph[e->sat - 1] = *e;
        enc.ephsat = e->sat;
        enc.time = e->toe;
        emit(&enc, 1041, 0, 0, out);
    }
    fclose(out);
    return 0;
}

/* Phase biases for a phase-bias subtype: see the usage note above. */
static void fill_phase_biases(rtcm_t *rtcm)
{
    for (int i = 0; i < MAXSAT; i++) {
        ssr_t *s = rtcm->ssr + i;
        if (!s->update) continue;
        for (int k = 0; k < MAXCODE; k++) {
            s->pbias[k] = s->cbias[k] * 1.37;
        }
        s->t0[5] = s->t0[4];
        s->udi[5] = s->udi[4];
        s->iod[5] = s->iod[4];
        s->yaw_ang = (i + 1) % 360;
        s->yaw_rate = ((i + 1) % 7 - 3) * 0.01;
    }
}

static void flush_4076(rtcm_t *rtcm, FILE *out)
{
    static const int offsets[] = {20, 40, 60, 80, 100, 120};
    fill_phase_biases(rtcm);
    for (int s = 0; s < 6; s++) {
        for (int k = 1; k <= 7; k++) {
            emit(rtcm, 4076, offsets[s] + k, !(s == 5 && k == 7), out);
        }
    }
    for (int i = 0; i < MAXSAT; i++) rtcm->ssr[i].update = 0;
}

static int encode_4076(const char *path, int week, double tow, int epochs, const char *outpath)
{
    static rtcm_t rtcm;
    long len, pos = 0, at;
    int frame_len, pending = 0, done = 0;
    gtime_t epoch = {0};
    uint8_t *buf = read_file(path, &len);
    FILE *out = fopen(outpath, "wb");

    init_at(&rtcm, week, tow);
    while ((at = next_frame(buf, len, &pos, &frame_len)) >= 0) {
        int type = getbitu(buf + at, 24, 12);
        if (!is_ssr_type(type)) continue;
        /* The SSR epoch of this frame, as RTKLIB reads it, from a copy of the
         * decoder state, so the corrections of the epoch before are written
         * before this frame's are read. */
        static rtcm_t probe;
        probe = rtcm;
        decode_one(&probe, buf + at, frame_len);
        if (pending && timediff(probe.time, epoch) != 0.0) {
            flush_4076(&rtcm, out);
            pending = 0;
            if (++done >= epochs) break;
        }
        decode_one(&rtcm, buf + at, frame_len);
        epoch = rtcm.time;
        pending = 1;
    }
    if (pending && done < epochs) flush_4076(&rtcm, out);
    fclose(out);
    free(buf);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 5 && !strcmp(argv[1], "decode")) {
        return decode_stream(argv[2], atoi(argv[3]), atof(argv[4]));
    }
    if (argc == 7 && !strcmp(argv[1], "encode-msm")) {
        return encode_msm(argv[2], atoi(argv[3]), atof(argv[4]), atoi(argv[5]), argv[6]);
    }
    if (argc == 7 && !strcmp(argv[1], "encode-legacy")) {
        return encode_legacy(argv[2], atoi(argv[3]), atof(argv[4]), atoi(argv[5]), argv[6]);
    }
    if (argc == 4 && !strcmp(argv[1], "encode-1041")) {
        return encode_1041(argv[2], argv[3]);
    }
    if (argc == 7 && !strcmp(argv[1], "encode-4076")) {
        return encode_4076(argv[2], atoi(argv[3]), atof(argv[4]), atoi(argv[5]), argv[6]);
    }
    fprintf(stderr, "usage: see the comment at the top of %s\n", __FILE__);
    return 2;
}
