#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rtklib.h"

extern int showmsg(const char *format, ...) { (void)format; return 0; }
extern void settspan(gtime_t start, gtime_t end) { (void)start; (void)end; }
extern void settime(gtime_t epoch) { (void)epoch; }

static const int observation_prns[] = {8, 10, 16, 18, 20, 21, 26, 27};
static const double pseudoranges_m[] = {
    23825519.844459895, 22717690.10174763, 20478653.376262885,
    21768335.23365917, 21248327.738292538, 20808709.800933376,
    21126481.58786735, 21341367.541037586
};
static const double application_receiver_ecef_m[] = {
    0x1.acd2200000000p+21, 0x1.7d1a800000000p+19, 0x1.405af00000000p+22
};
static const uint64_t dop_receiver_bits[] = {
    UINT64_C(0x41511b07ff824402), UINT64_C(0x4120cd6b5f861f3b),
    UINT64_C(0x41511e62229e1c30)
};

static double from_bits(uint64_t bits)
{
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void print_vec3(const double *value)
{
    printf("[%.17g,%.17g,%.17g]", value[0], value[1], value[2]);
}

static void print_matrix4(const double *value)
{
    int row_index, column_index;
    putchar('[');
    for (row_index = 0; row_index < 4; row_index++) {
        for (column_index = 0; column_index < 4; column_index++) {
            printf("%s%.17g", row_index == 0 && column_index == 0 ? "" : ",",
                   value[row_index + column_index * 4]);
        }
    }
    putchar(']');
}

static void print_json_string(const char *value)
{
    const unsigned char *cursor = (const unsigned char *)value;
    putchar('"');
    while (*cursor) {
        if (*cursor == '"' || *cursor == '\\') {
            putchar('\\');
            putchar(*cursor);
        } else if (*cursor < 0x20) {
            printf("\\u%04x", *cursor);
        } else {
            putchar(*cursor);
        }
        cursor++;
    }
    putchar('"');
}

static int compute_dops(int count, const double *azimuth_elevation_rad,
                        double elevation_min_rad, double *dop_values,
                        double *normal_matrix, double *covariance_matrix)
{
    double design[4 * MAXSAT], covariance[16];
    int satellite_index, row_count = 0;

    for (satellite_index = 0; satellite_index < 4; satellite_index++) dop_values[satellite_index] = 0.0;
    dop_values[4] = 0.0;
    for (satellite_index = 0; satellite_index < count && satellite_index < MAXSAT; satellite_index++) {
        double azimuth = azimuth_elevation_rad[2 * satellite_index];
        double elevation = azimuth_elevation_rad[2 * satellite_index + 1];
        double cosine_elevation, sine_elevation;
        if (elevation < elevation_min_rad || elevation <= 0.0) continue;
        cosine_elevation = cos(elevation);
        sine_elevation = sin(elevation);
        design[4 * row_count] = cosine_elevation * sin(azimuth);
        design[4 * row_count + 1] = cosine_elevation * cos(azimuth);
        design[4 * row_count + 2] = sine_elevation;
        design[4 * row_count + 3] = 1.0;
        row_count++;
    }
    if (row_count < 4) {
        fprintf(stderr,
                "RTKLIB dops has only %d GPS satellites above %.17g degrees; need at least 4\n",
                row_count, elevation_min_rad * R2D);
        return 0;
    }
    matmul("NT", 4, 4, row_count, design, design, covariance);
    matcpy(normal_matrix, covariance, 4, 4);
    if (matinv(covariance, 4)) {
        fprintf(stderr,
                "RTKLIB DOP normal-matrix inverse failed for %d GPS satellites above %.17g degrees\n",
                row_count, elevation_min_rad * R2D);
        return 0;
    }
    matcpy(covariance_matrix, covariance, 4, 4);
    dops(count, azimuth_elevation_rad, elevation_min_rad, dop_values);
    dop_values[4] = sqrt(covariance[15]);
    for (satellite_index = 0; satellite_index < 5; satellite_index++) {
        if (!isfinite(dop_values[satellite_index]) || dop_values[satellite_index] <= 0.0) {
            fprintf(stderr, "RTKLIB DOP value %d is invalid: %.17g\n",
                    satellite_index, dop_values[satellite_index]);
            return 0;
        }
    }
    return 1;
}

static int emit_geometry(const char *path)
{
    nav_t navigation = {0};
    gtime_t epoch = epoch2time((double[]){2020, 6, 24, 12, 0, 0});
    double application_position[3], dop_receiver[3], dop_position[3];
    double application_azel[MAXSAT * 2], dop_azel[MAXSAT * 2], dop_values[5];
    double dop_normal_matrix[16], dop_covariance_matrix[16];
    int satellite_numbers[MAXSAT];
    int satellite_prn, satellite_count = 0, first_row = 1;

    readsp3(path, &navigation, 0);
    if (navigation.ne < 11) {
        fprintf(stderr, "RTKLIB readsp3 loaded only %d epochs from %s\n", navigation.ne, path);
        freenav(&navigation, 0xFF);
        return 0;
    }
    ecef2pos(application_receiver_ecef_m, application_position);
    for (satellite_prn = 0; satellite_prn < 3; satellite_prn++)
        dop_receiver[satellite_prn] = from_bits(dop_receiver_bits[satellite_prn]);
    ecef2pos(dop_receiver, dop_position);
    printf("{\"epoch_gps\":[2020,6,24,12,0,0],\"receiver_ecef_m\":");
    print_vec3(application_receiver_ecef_m);
    printf(",\"dop_receiver_ecef_m\":");
    print_vec3(dop_receiver);
    printf(",\"satellites\":[");

    for (satellite_prn = 1; satellite_prn <= MAXPRNGPS; satellite_prn++) {
        int satellite = satno(SYS_GPS, satellite_prn);
        double state[6], clock[2], variance, application_line[3], dop_line[3];
        double application_angles[2], dop_angles[2];
        char satellite_name[8];
        if (!peph2pos(epoch, satellite, &navigation, 0, state, clock, &variance)) continue;
        if (geodist(state, application_receiver_ecef_m, application_line) <= 0.0 ||
            geodist(state, dop_receiver, dop_line) <= 0.0) continue;
        satazel(application_position, application_line, application_angles);
        satazel(dop_position, dop_line, dop_angles);
        satellite_numbers[satellite_count] = satellite;
        application_azel[2 * satellite_count] = application_angles[0];
        application_azel[2 * satellite_count + 1] = application_angles[1];
        dop_azel[2 * satellite_count] = dop_angles[0];
        dop_azel[2 * satellite_count + 1] = dop_angles[1];
        satno2id(satellite, satellite_name);
        printf("%s{\"id\":\"%s\",\"position_ecef_m\":", first_row ? "" : ",", satellite_name);
        print_vec3(state);
        printf(",\"application_azimuth_rad\":%.17g,\"application_elevation_rad\":%.17g,"
               "\"application_azimuth_deg\":%.17g,\"application_elevation_deg\":%.17g,"
               "\"dop_azimuth_rad\":%.17g,\"dop_elevation_rad\":%.17g,"
               "\"dop_visible_at_5_deg\":%s,\"application_visible_at_10_deg\":%s}",
               application_angles[0], application_angles[1],
               application_angles[0] * R2D, application_angles[1] * R2D,
               dop_angles[0], dop_angles[1],
               dop_angles[1] >= 5.0 * D2R ? "true" : "false",
               application_angles[1] >= 10.0 * D2R ? "true" : "false");
        first_row = 0;
        satellite_count++;
    }
    if (satellite_count > MAXSAT) {
        fprintf(stderr, "RTKLIB geometry returned %d satellites; capacity is %d\n",
                satellite_count, MAXSAT);
        freenav(&navigation, 0xFF);
        return 0;
    }
    if (!compute_dops(satellite_count, dop_azel, 5.0 * D2R, dop_values,
                      dop_normal_matrix, dop_covariance_matrix)) {
        fprintf(stderr, "RTKLIB geometry DOP calculation failed for %s\n", path);
        freenav(&navigation, 0xFF);
        return 0;
    }
    printf("],\"dops_mask5_deg\":{\"gdop\":%.17g,\"pdop\":%.17g,"
           "\"hdop\":%.17g,\"vdop\":%.17g,\"tdop\":%.17g},"
           "\"dop_normal_matrix_local\":",
           dop_values[0], dop_values[1], dop_values[2], dop_values[3], dop_values[4]);
    print_matrix4(dop_normal_matrix);
    printf(",\"dop_covariance_local\":");
    print_matrix4(dop_covariance_matrix);
    printf(",\"visible_gps_mask10\":[");
    first_row = 1;
    for (satellite_prn = 0; satellite_prn < satellite_count; satellite_prn++) {
        char satellite_name[8];
        if (application_azel[2 * satellite_prn + 1] < 10.0 * D2R) continue;
        satno2id(satellite_numbers[satellite_prn], satellite_name);
        printf("%s\"%s\"", first_row ? "" : ",", satellite_name);
        first_row = 0;
    }
    printf("]}");
    freenav(&navigation, 0xFF);
    return 1;
}

static int parse_iono_mode(const char *mode, int *option)
{
    if (strcmp(mode, "off") == 0) {
        *option = IONOOPT_OFF;
        return 1;
    }
    if (strcmp(mode, "brdc") == 0) {
        *option = IONOOPT_BRDC;
        return 1;
    }
    return 0;
}

static int parse_tropo_mode(const char *mode, int *option)
{
    if (strcmp(mode, "off") == 0) {
        *option = TROPOPT_OFF;
        return 1;
    }
    if (strcmp(mode, "saas") == 0) {
        *option = TROPOPT_SAAS;
        return 1;
    }
    return 0;
}

static int emit_static_sp3_tx_states(const nav_t *navigation)
{
    gtime_t receive_epoch = epoch2time((double[]){2020, 6, 24, 12, 0, 0});
    int satellite_index;

    putchar('[');
    for (satellite_index = 0;
         satellite_index < (int)(sizeof(observation_prns) / sizeof(observation_prns[0]));
         satellite_index++) {
        int satellite = satno(SYS_GPS, observation_prns[satellite_index]);
        gtime_t clock_epoch = timeadd(receive_epoch, -pseudoranges_m[satellite_index] / CLIGHT);
        gtime_t transmit_epoch;
        double state[6], clock[2], variance, product_clock_s;
        char satellite_name[8];

        if (!peph2pos(clock_epoch, satellite, navigation, 0, state, clock, &variance)) {
            fprintf(stderr, "RTKLIB peph2pos could not interpolate the SP3 clock for %d\n",
                    satellite);
            return 0;
        }
        product_clock_s = clock[0] + 2.0 * dot3(state, state + 3) / CLIGHT / CLIGHT;
        if (!isfinite(product_clock_s) || product_clock_s == 0.0) {
            fprintf(stderr, "RTKLIB SP3 product clock is invalid for %d: %.17g\n",
                    satellite, product_clock_s);
            return 0;
        }
        transmit_epoch = timeadd(clock_epoch, -product_clock_s);
        if (!peph2pos(transmit_epoch, satellite, navigation, 0, state, clock, &variance)) {
            fprintf(stderr, "RTKLIB peph2pos could not interpolate the SP3 transmit state for %d\n",
                    satellite);
            return 0;
        }
        satno2id(satellite, satellite_name);
        printf("%s{\"id\":\"%s\",\"clock_epoch_offset_s\":%.17g,"
               "\"sp3_product_clock_s\":%.17g,\"transmit_epoch_offset_s\":%.17g,"
               "\"position_ecef_m\":",
               satellite_index ? "," : "", satellite_name,
               timediff(clock_epoch, receive_epoch), product_clock_s,
               timediff(transmit_epoch, receive_epoch));
        print_vec3(state);
        printf(",\"velocity_ecef_m_s\":[%.17g,%.17g,%.17g]}",
               state[3], state[4], state[5]);
    }
    putchar(']');
    return 1;
}

static int emit_static(const char *path, const char *navigation_path,
                       const char *iono_mode, const char *tropo_mode)
{
    nav_t navigation = {0};
    obsd_t observations[sizeof(observation_prns) / sizeof(observation_prns[0])] = {{0}};
    prcopt_t options = prcopt_default;
    sol_t solution = {{0}};
    ssat_t satellite_status[MAXSAT] = {{0}};
    char message[256] = "";
    int observation_count, satellite_index, used_count = 0, status, navigation_read_status;

    if (!parse_iono_mode(iono_mode, &options.ionoopt) ||
        !parse_tropo_mode(tropo_mode, &options.tropopt)) {
        fprintf(stderr, "unsupported RTKLIB ionosphere/troposphere mode: %s/%s\n",
                iono_mode, tropo_mode);
        return 0;
    }
    readsp3(path, &navigation, 0);
    if (navigation.ne < 11) {
        fprintf(stderr, "RTKLIB readsp3 loaded only %d epochs from %s\n", navigation.ne, path);
        freenav(&navigation, 0xFF);
        return 0;
    }
    navigation_read_status = readrnx(navigation_path, 1, "", NULL, &navigation, NULL);
    if (navigation_read_status <= 0 || navigation.n <= 0) {
        fprintf(stderr,
                "RTKLIB pntpos requires broadcast ephemerides for satposs transmit-time estimation; readrnx status=%d, broadcast ephemeris records=%d, SP3 epochs=%d, nav=%s\n",
                navigation_read_status, navigation.n, navigation.ne, navigation_path);
        freenav(&navigation, 0xFF);
        return 0;
    }
    observation_count = (int)(sizeof(observations) / sizeof(observations[0]));
    for (satellite_index = 0; satellite_index < observation_count; satellite_index++) {
        observations[satellite_index].time = epoch2time((double[]){2020, 6, 24, 12, 0, 0});
        observations[satellite_index].sat = satno(SYS_GPS, observation_prns[satellite_index]);
        observations[satellite_index].P[0] = pseudoranges_m[satellite_index];
        observations[satellite_index].code[0] = CODE_L1C;
    }
    options.mode = PMODE_SINGLE;
    options.navsys = SYS_GPS;
    options.nf = 1;
    options.elmin = -90.0 * D2R;
    options.sateph = EPHOPT_PREC;
    memcpy(solution.rr, (double[]){4.5e6, 0.5e6, 4.5e6}, 3 * sizeof(double));
    status = pntpos(observations, observation_count, &navigation, &options,
                    &solution, NULL, satellite_status, message);
    for (satellite_index = 0; satellite_index < MAXSAT; satellite_index++)
        if (satellite_status[satellite_index].vs) used_count++;
    if (status != 1) {
        fprintf(stderr,
                "RTKLIB pntpos failed: status=%d, used=%d/%d, iono=%s, tropo=%s, elmin=%.17g degrees, SP3 epochs=%d, broadcast ephemeris records=%d, message=%s\n",
                status, used_count, observation_count, iono_mode, tropo_mode,
                options.elmin * R2D, navigation.ne, navigation.n,
                message[0] ? message : "(empty)");
        freenav(&navigation, 0xFF);
        return 0;
    }
    used_count = 0;
    printf("{\"epoch_gps\":[2020,6,24,12,0,0],\"options\":{\"iono\":");
    print_json_string(iono_mode);
    printf(",\"tropo\":");
    print_json_string(tropo_mode);
    printf(",\"ephemeris\":\"precise\",\"elevation_mask_deg\":-90},"
           "\"initial_position_ecef_m\":[4500000,500000,4500000],\"input_satellites\":[");
    for (satellite_index = 0; satellite_index < observation_count; satellite_index++) {
        char satellite_name[8];
        satno2id(satno(SYS_GPS, observation_prns[satellite_index]), satellite_name);
        printf("%s", satellite_index ? "," : "");
        print_json_string(satellite_name);
    }
    printf("],\"pseudoranges_m\":[");
    for (satellite_index = 0; satellite_index < observation_count; satellite_index++)
        printf("%s%.17g", satellite_index ? "," : "", pseudoranges_m[satellite_index]);
    printf("],\"sp3_tx_states\":");
    if (!emit_static_sp3_tx_states(&navigation)) {
        freenav(&navigation, 0xFF);
        return 0;
    }
    printf(",\"status\":%d,\"position_ecef_m\":", status);
    print_vec3(solution.rr);
    printf(",\"clock_s\":%.17g,\"used_satellites\":[", solution.dtr[0]);
    for (satellite_index = 0; satellite_index < MAXSAT; satellite_index++) {
        char satellite_name[8];
        if (!satellite_status[satellite_index].vs) continue;
        satno2id(satellite_index + 1, satellite_name);
        printf("%s\"%s\"", used_count++ ? "," : "", satellite_name);
    }
    printf("],\"message\":");
    print_json_string(message);
    printf("}");
    freenav(&navigation, 0xFF);
    return 1;
}

int main(int argument_count, char **arguments)
{
    if (argument_count != 12) {
        fprintf(stderr, "usage: %s <revision> <geometry-sha256> <static-sha256> "
                "<navigation-sha256> <navigation-name> <navigation-source-url> "
                "<iono:off|brdc> <tropo:off|saas> <geometry-sp3> <static-sp3> <navigation-rinex>\n",
                arguments[0]);
        return 2;
    }
    printf("{\"schema\":\"sidereon-rtklib-sp3-oracle/v1\",\"provenance\":{"
           "\"repository\":\"rtklibexplorer/RTKLIB\",\"branch\":\"demo5\","
           "\"revision\":\"%s\",\"source_url\":"
           "\"https://github.com/rtklibexplorer/RTKLIB/tree/%s\","
           "\"generator\":\"fixtures-generators/rtklib_sp3_oracle/generate.sh\","
           "\"geometry_fixture\":\"sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3\","
           "\"geometry_sha256\":\"%s\",\"static_fixture\":\"sp3/trimmed_go_static.sp3\","
           "\"static_sha256\":\"%s\",\"navigation_file\":",
           arguments[1], arguments[1], arguments[2], arguments[3]);
    print_json_string(arguments[5]);
    printf(",\"navigation_sha256\":\"%s\",\"navigation_source_url\":",
           arguments[4]);
    print_json_string(arguments[6]);
    printf(",\"source_functions\":[\"preceph.c:peph2pos/pephpos\","
           "\"rtkcmn.c:satazel/dops\",\"pntpos.c:pntpos\"]},\"geometry\":");
    if (!emit_geometry(arguments[9])) {
        fprintf(stderr, "RTKLIB SP3 oracle geometry stage failed\n");
        return 1;
    }
    printf(",\"go_fixture_static\":");
    if (!emit_static(arguments[10], arguments[11], arguments[7], arguments[8])) {
        fprintf(stderr, "RTKLIB SP3 oracle static-position stage failed\n");
        return 1;
    }
    printf("}\n");
    return 0;
}
