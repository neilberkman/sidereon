// BNC SSR oracle: decodes RTCM SSR and IGS SSR (4076) frames with the SSR codec
// of BNC, the BKG NTRIP Client, and prints what BNC reads from each frame, and
// writes SSR frames with BNC's encoder. `generate.sh` builds it against the BNC
// sources and writes the fixtures `tests/rtcm_ssr_bnc_oracle.rs` reads.
//
// BNC is used unmodified (BNC 2.13.7, bnc-2.13.7-sources.zip from
// https://igs.bkg.bund.de/root_ftp/NTRIP/software/BNC/); only its SSR codec
// (src/RTCM3/clock_and_orbit) is compiled, with a stand-in header for the part
// of BNC it does not use (see generate.sh).
//
// usage:
//   bnc_ssr_oracle decode <stream>
//       One JSON object per CRC-valid frame, one per line, in stream order:
//       the return code of GetSSR and every value it stores. Floating values
//       are printed as the hex digits of their IEEE 754 bits.
//   bnc_ssr_oracle encode <out>
//       Writes, with BNC's encoder, every RTCM SSR message of every system
//       BNC writes (orbit, clock, combined, URA, high-rate clock, code bias,
//       phase bias, 1057..1270) and the RTCM VTEC message 1264, then the same
//       groups in the IGS SSR format (4076 subtypes and 201), from values a
//       fixed-seed generator draws over each field's range.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "clock_and_orbit/clock_orbit_igs.h"
#include "clock_and_orbit/clock_orbit_rtcm.h"

// The CRC-24Q BNC's encoder and decoder call; BNC defines it in bncutils.cpp.
unsigned long CRC24(long size, const unsigned char *buf) {
  unsigned long crc = 0;
  for (long i = 0; i < size; i++) {
    crc ^= ((unsigned long)buf[i]) << 16;
    for (int j = 0; j < 8; j++) {
      crc <<= 1;
      if (crc & 0x1000000) crc ^= 0x01864CFB;
    }
  }
  return crc & 0xFFFFFF;
}

static unsigned long long dbits(double x) {
  uint64_t u;
  memcpy(&u, &x, sizeof u);
  return (unsigned long long)u;
}

static unsigned int fbits(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof u);
  return (unsigned int)u;
}

static void print_clock_orbit(SsrCorr &c, const SsrCorr::ClockOrbit &co) {
  printf(",\"co\":{\"type\":%u,\"iod\":%u,\"provider\":%u,\"solution\":%u,\"udi\":%u,"
         "\"datum\":%u,\"systems\":[",
         co.messageType, co.SSRIOD, co.SSRProviderID, co.SSRSolutionID, co.UpdateInterval,
         co.SatRefDatum);
  for (int s = 0; s < CLOCKORBIT_SATNUM; s++) {
    printf("%s{\"epoch\":%u,\"sats\":[", s ? "," : "", co.EpochTime[s]);
    for (unsigned k = 0; k < co.NumberOfSat[s]; k++) {
      const SsrCorr::ClockOrbit::SatData &d = co.Sat[c.satoffset[s] + k];
      printf("%s{\"id\":%u,\"iod\":%u,\"toe\":%u,\"ura\":\"%016llx\",\"hr\":\"%016llx\","
             "\"orbit\":[\"%016llx\",\"%016llx\",\"%016llx\",\"%016llx\",\"%016llx\",\"%016llx\"],"
             "\"clock\":[\"%016llx\",\"%016llx\",\"%016llx\"]}",
             k ? "," : "", d.ID, d.IOD, d.toe, dbits(d.UserRangeAccuracy), dbits(d.hrclock),
             dbits(d.Orbit.DeltaRadial), dbits(d.Orbit.DeltaAlongTrack),
             dbits(d.Orbit.DeltaCrossTrack), dbits(d.Orbit.DotDeltaRadial),
             dbits(d.Orbit.DotDeltaAlongTrack), dbits(d.Orbit.DotDeltaCrossTrack),
             dbits(d.Clock.DeltaA0), dbits(d.Clock.DeltaA1), dbits(d.Clock.DeltaA2));
    }
    printf("]}");
  }
  printf("]}");
}

static void print_code_bias(SsrCorr &c, const SsrCorr::CodeBias &b) {
  printf(",\"cb\":{\"type\":%u,\"iod\":%u,\"provider\":%u,\"solution\":%u,\"udi\":%u,"
         "\"systems\":[",
         b.messageType, b.SSRIOD, b.SSRProviderID, b.SSRSolutionID, b.UpdateInterval);
  for (int s = 0; s < CLOCKORBIT_SATNUM; s++) {
    printf("%s{\"epoch\":%u,\"sats\":[", s ? "," : "", b.EpochTime[s]);
    for (unsigned k = 0; k < b.NumberOfSat[s]; k++) {
      const SsrCorr::CodeBias::BiasSat &d = b.Sat[c.satoffset[s] + k];
      printf("%s{\"id\":%u,\"biases\":[", k ? "," : "", d.ID);
      for (unsigned j = 0; j < d.NumberOfCodeBiases; j++) {
        printf("%s[%u,\"%08x\"]", j ? "," : "", d.Biases[j].Type, fbits(d.Biases[j].Bias));
      }
      printf("]}");
    }
    printf("]}");
  }
  printf("]}");
}

static void print_phase_bias(SsrCorr &c, const SsrCorr::PhaseBias &p) {
  printf(",\"pb\":{\"type\":%u,\"iod\":%u,\"provider\":%u,\"solution\":%u,\"udi\":%u,"
         "\"dispersive\":%u,\"mw\":%u,\"systems\":[",
         p.messageType, p.SSRIOD, p.SSRProviderID, p.SSRSolutionID, p.UpdateInterval,
         p.DispersiveBiasConsistencyIndicator, p.MWConsistencyIndicator);
  for (int s = 0; s < CLOCKORBIT_SATNUM; s++) {
    printf("%s{\"epoch\":%u,\"sats\":[", s ? "," : "", p.EpochTime[s]);
    for (unsigned k = 0; k < p.NumberOfSat[s]; k++) {
      const SsrCorr::PhaseBias::PhaseBiasSat &d = p.Sat[c.satoffset[s] + k];
      printf("%s{\"id\":%u,\"yaw\":\"%016llx\",\"yaw_rate\":\"%016llx\",\"biases\":[",
             k ? "," : "", d.ID, dbits(d.YawAngle), dbits(d.YawRate));
      for (unsigned j = 0; j < d.NumberOfPhaseBiases; j++) {
        printf("%s[%u,%u,%u,%u,\"%08x\"]", j ? "," : "", d.Biases[j].Type,
               d.Biases[j].SignalIntegerIndicator, d.Biases[j].SignalsWideLaneIntegerIndicator,
               d.Biases[j].SignalDiscontinuityCounter, fbits(d.Biases[j].Bias));
      }
      printf("]}");
    }
    printf("]}");
  }
  printf("]}");
}

static void print_vtec(const SsrCorr::VTEC &v) {
  printf(",\"vtec\":{\"epoch\":%u,\"udi\":%u,\"iod\":%u,\"provider\":%u,\"solution\":%u,"
         "\"quality\":\"%016llx\",\"layers\":[",
         v.EpochTime, v.UpdateInterval, v.SSRIOD, v.SSRProviderID, v.SSRSolutionID,
         dbits(v.Quality));
  for (unsigned l = 0; l < v.NumLayers; l++) {
    const SsrCorr::VTEC::IonoLayers &L = v.Layers[l];
    printf("%s{\"height\":\"%016llx\",\"degree\":%u,\"order\":%u,\"c\":[", l ? "," : "",
           dbits(L.Height), L.Degree, L.Order);
    int first = 1;
    for (unsigned o = 0; o <= L.Order; o++) {
      for (unsigned d = o; d <= L.Degree; d++) {
        printf("%s\"%016llx\"", first ? "" : ",", dbits(L.Cosinus[d][o]));
        first = 0;
      }
    }
    printf("],\"s\":[");
    first = 1;
    for (unsigned o = 1; o <= L.Order; o++) {
      for (unsigned d = o; d <= L.Degree; d++) {
        printf("%s\"%016llx\"", first ? "" : ",", dbits(L.Sinus[d][o]));
        first = 0;
      }
    }
    printf("]}");
  }
  printf("]}");
}

static std::vector<unsigned char> read_file(const char *path) {
  std::vector<unsigned char> data;
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(2);
  }
  int ch;
  while ((ch = fgetc(fp)) != EOF) data.push_back((unsigned char)ch);
  fclose(fp);
  return data;
}

static int decode(const char *path) {
  std::vector<unsigned char> data = read_file(path);
  SsrCorrRtcm rtcm;
  SsrCorrIgs igs;
  static SsrCorr::ClockOrbit co;
  static SsrCorr::CodeBias cb;
  static SsrCorr::PhaseBias pb;
  static SsrCorr::VTEC vtec;
  size_t pos = 0;
  while (pos + 6 <= data.size()) {
    if (data[pos] != 0xD3) {
      pos++;
      continue;
    }
    size_t len = ((data[pos + 1] & 3) << 8) | data[pos + 2];
    if (pos + 6 + len > data.size()) break;
    unsigned long crc = ((unsigned long)data[pos + 3 + len] << 16) |
                        ((unsigned long)data[pos + 4 + len] << 8) | data[pos + 5 + len];
    if (CRC24((long)(3 + len), &data[pos]) != crc) {
      pos++;
      continue;
    }
    unsigned type = ((unsigned)data[pos + 3] << 4) | (data[pos + 4] >> 4);
    SsrCorr &codec = type == 4076 ? (SsrCorr &)igs : (SsrCorr &)rtcm;
    memset(&co, 0, sizeof co);
    memset(&cb, 0, sizeof cb);
    memset(&pb, 0, sizeof pb);
    memset(&vtec, 0, sizeof vtec);
    int used = 0;
    int ret = codec.GetSSR(&co, &cb, &vtec, &pb, (const char *)&data[pos], len + 6, &used);
    printf("{\"offset\":%zu,\"type\":%u,\"ret\":%d", pos, type, ret);
    if (co.messageType) print_clock_orbit(codec, co);
    if (cb.messageType) print_code_bias(codec, cb);
    if (pb.messageType) print_phase_bias(codec, pb);
    if (vtec.NumLayers) print_vtec(vtec);
    printf("}\n");
    pos += 6 + len;
  }
  return 0;
}

// A fixed-seed generator (the 64-bit LCG of Knuth's MMIX).
static uint64_t state = 20260924;
static uint64_t next_u64() {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return state >> 11;
}
static long draw(long low, long high) {
  return low + (long)(next_u64() % (uint64_t)(high - low + 1));
}

// Satellite IDs each system's RTCM layout can carry (QZSS four bits, GLONASS
// five, the others six); IGS SSR carries six for all.
static const unsigned ID_LIMIT_RTCM[CLOCKORBIT_SATNUM] = {63, 31, 63, 15, 63, 63};

static void fill(SsrCorr &c, bool igs, SsrCorr::ClockOrbit &co, SsrCorr::CodeBias &cb,
                 SsrCorr::PhaseBias &pb) {
  memset(&co, 0, sizeof co);
  memset(&cb, 0, sizeof cb);
  memset(&pb, 0, sizeof pb);
  co.SSRIOD = cb.SSRIOD = pb.SSRIOD = (unsigned)draw(0, 15);
  co.SSRProviderID = cb.SSRProviderID = pb.SSRProviderID = (unsigned)draw(0, 65535);
  co.SSRSolutionID = cb.SSRSolutionID = pb.SSRSolutionID = (unsigned)draw(0, 15);
  co.UpdateInterval = cb.UpdateInterval = pb.UpdateInterval = (unsigned)draw(0, 15);
  co.SatRefDatum = (unsigned)draw(0, 1);
  pb.DispersiveBiasConsistencyIndicator = (unsigned)draw(0, 1);
  pb.MWConsistencyIndicator = (unsigned)draw(0, 1);
  for (int i = 0; i < 7; i++) co.Supplied[i] = 1;
  for (int s = 0; s < CLOCKORBIT_SATNUM; s++) {
    unsigned epoch = s == CLOCKORBIT_SATGLONASS && !igs ? (unsigned)draw(0, 86399)
                                                        : (unsigned)draw(0, 604799);
    co.EpochTime[s] = cb.EpochTime[s] = pb.EpochTime[s] = epoch;
    unsigned n = 3;
    unsigned limit = igs ? 63 : ID_LIMIT_RTCM[s];
    co.NumberOfSat[s] = cb.NumberOfSat[s] = pb.NumberOfSat[s] = n;
    for (unsigned k = 0; k < n; k++) {
      unsigned at = c.satoffset[s] + k;
      unsigned id = k == 0 ? 1 : k == 1 ? limit : (unsigned)draw(2, limit - 1);
      SsrCorr::ClockOrbit::SatData &d = co.Sat[at];
      d.ID = id;
      // The issue: eight bits (IGS; RTCM GPS, GLONASS, QZSS), ten (RTCM
      // Galileo), the SBAS IOD CRC (24 bits) or the BeiDou IOD (8 bits).
      d.IOD = (unsigned)draw(0, igs ? 255
                                     : s == CLOCKORBIT_SATGALILEO ? 1023
                                     : s == CLOCKORBIT_SATSBAS ? 0xFFFFFF : 255);
      d.toe = s == CLOCKORBIT_SATSBAS ? 16 * (unsigned)draw(0, 511)
            : s == CLOCKORBIT_SATBDS ? 8 * (unsigned)draw(0, 1023) : 0;
      d.UserRangeAccuracy = draw(0, 5500) / 1000.0;
      d.hrclock = draw(-2097151, 2097151) / 10000.0;
      d.Orbit.DeltaRadial = draw(-2097151, 2097151) / 10000.0;
      d.Orbit.DeltaAlongTrack = draw(-524287, 524287) / 2500.0;
      d.Orbit.DeltaCrossTrack = draw(-524287, 524287) / 2500.0;
      d.Orbit.DotDeltaRadial = draw(-1048575, 1048575) / 1000000.0;
      d.Orbit.DotDeltaAlongTrack = draw(-262143, 262143) / 250000.0;
      d.Orbit.DotDeltaCrossTrack = draw(-262143, 262143) / 250000.0;
      d.Clock.DeltaA0 = draw(-2097151, 2097151) / 10000.0;
      d.Clock.DeltaA1 = draw(-1048575, 1048575) / 1000000.0;
      d.Clock.DeltaA2 = draw(-67108863, 67108863) / 50000000.0;
      SsrCorr::CodeBias::BiasSat &b = cb.Sat[at];
      b.ID = id;
      b.NumberOfCodeBiases = 3;
      SsrCorr::PhaseBias::PhaseBiasSat &p = pb.Sat[at];
      p.ID = id;
      p.NumberOfPhaseBiases = 2;
      p.YawAngle = draw(0, 511) * MPI / 256.0;
      p.YawRate = draw(-127, 127) * MPI / 8192.0;
      for (unsigned j = 0; j < 3; j++) {
        b.Biases[j].Type = j;
        b.Biases[j].Bias = (float)(draw(-8191, 8191) / 100.0);
      }
      for (unsigned j = 0; j < 2; j++) {
        p.Biases[j].Type = j;
        p.Biases[j].SignalIntegerIndicator = (unsigned)draw(0, 1);
        p.Biases[j].SignalsWideLaneIntegerIndicator = (unsigned)draw(0, 3);
        p.Biases[j].SignalDiscontinuityCounter = (unsigned)draw(0, 15);
        p.Biases[j].Bias = (float)(draw(-524287, 524287) / 10000.0);
      }
    }
  }
}

static void fill_vtec(SsrCorr::VTEC &v) {
  memset(&v, 0, sizeof v);
  v.EpochTime = (unsigned)draw(0, 604799);
  v.UpdateInterval = (unsigned)draw(0, 15);
  v.SSRIOD = (unsigned)draw(0, 15);
  v.SSRProviderID = (unsigned)draw(0, 65535);
  v.SSRSolutionID = (unsigned)draw(0, 15);
  v.Quality = draw(0, 511) / 20.0;
  v.NumLayers = 2;
  const unsigned degree[2] = {15, 3}, order[2] = {15, 2};
  for (unsigned l = 0; l < v.NumLayers; l++) {
    SsrCorr::VTEC::IonoLayers &L = v.Layers[l];
    L.Height = draw(0, 255) * 10000.0;
    L.Degree = degree[l];
    L.Order = order[l];
    for (unsigned o = 0; o <= L.Order; o++) {
      for (unsigned d = o; d <= L.Degree; d++) {
        L.Cosinus[d][o] = draw(-32767, 32767) / 200.0;
        if (o) L.Sinus[d][o] = draw(-32767, 32767) / 200.0;
      }
    }
  }
}

static void put(FILE *out, const char *buffer, size_t n) {
  if (n == 0) {
    fprintf(stderr, "BNC encoder wrote nothing\n");
    exit(2);
  }
  fwrite(buffer, 1, n, out);
}

static int encode(const char *path) {
  FILE *out = fopen(path, "wb");
  static char buffer[CLOCKORBIT_BUFFERSIZE];
  static SsrCorr::ClockOrbit co;
  static SsrCorr::CodeBias cb;
  static SsrCorr::PhaseBias pb;
  static SsrCorr::VTEC vtec;
  SsrCorrRtcm rtcm;
  SsrCorrIgs igs;
  for (int format = 0; format < 2; format++) {
    SsrCorr &c = format ? (SsrCorr &)igs : (SsrCorr &)rtcm;
    fill(c, format == 1, co, cb, pb);
    const unsigned co_types[CLOCKORBIT_SATNUM][5] = {
        {c.COTYPE_GPSORBIT, c.COTYPE_GPSCLOCK, c.COTYPE_GPSCOMBINED, c.COTYPE_GPSURA,
         c.COTYPE_GPSHR},
        {c.COTYPE_GLONASSORBIT, c.COTYPE_GLONASSCLOCK, c.COTYPE_GLONASSCOMBINED,
         c.COTYPE_GLONASSURA, c.COTYPE_GLONASSHR},
        {c.COTYPE_GALILEOORBIT, c.COTYPE_GALILEOCLOCK, c.COTYPE_GALILEOCOMBINED,
         c.COTYPE_GALILEOURA, c.COTYPE_GALILEOHR},
        {c.COTYPE_QZSSORBIT, c.COTYPE_QZSSCLOCK, c.COTYPE_QZSSCOMBINED, c.COTYPE_QZSSURA,
         c.COTYPE_QZSSHR},
        {c.COTYPE_SBASORBIT, c.COTYPE_SBASCLOCK, c.COTYPE_SBASCOMBINED, c.COTYPE_SBASURA,
         c.COTYPE_SBASHR},
        {c.COTYPE_BDSORBIT, c.COTYPE_BDSCLOCK, c.COTYPE_BDSCOMBINED, c.COTYPE_BDSURA,
         c.COTYPE_BDSHR}};
    const unsigned cb_types[CLOCKORBIT_SATNUM] = {c.CBTYPE_GPS, c.CBTYPE_GLONASS,
                                                  c.CBTYPE_GALILEO, c.CBTYPE_QZSS,
                                                  c.CBTYPE_SBAS, c.CBTYPE_BDS};
    // BNC's RTCM codec sets its PBTYPE_* constants with post-increments, which
    // leaves them one system off; its phase-bias writer takes 1265 + the
    // system's index, which is what is passed for the RTCM format here.
    unsigned pb_types[CLOCKORBIT_SATNUM] = {c.PBTYPE_GPS, c.PBTYPE_GLONASS,
                                            c.PBTYPE_GALILEO, c.PBTYPE_QZSS,
                                            c.PBTYPE_SBAS, c.PBTYPE_BDS};
    if (!format) {
      for (int s = 0; s < CLOCKORBIT_SATNUM; s++) pb_types[s] = 1265 + s;
    }
    for (int s = 0; s < CLOCKORBIT_SATNUM; s++) {
      for (int k = 0; k < 5; k++) {
        put(out, buffer, c.MakeClockOrbit(&co, co_types[s][k], 0, buffer, sizeof buffer));
      }
      put(out, buffer, c.MakeCodeBias(&cb, cb_types[s], 0, buffer, sizeof buffer));
      put(out, buffer, c.MakePhaseBias(&pb, pb_types[s], 0, buffer, sizeof buffer));
    }
    fill_vtec(vtec);
    put(out, buffer, c.MakeVTEC(&vtec, 0, buffer, sizeof buffer));
  }
  fclose(out);
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 3 && !strcmp(argv[1], "decode")) return decode(argv[2]);
  if (argc == 3 && !strcmp(argv[1], "encode")) return encode(argv[2]);
  fprintf(stderr, "usage: see the comment at the top of %s\n", __FILE__);
  return 2;
}
