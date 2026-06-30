// Test-only oracle wrapper. Calls Vallado C++ sgp4init with the supplied
// element bits, then flattens the resulting elsetrec into two arrays so
// the Rust side can field-by-field diff against its own ElsetRec.
//
// The double_out / int_out layouts MUST stay in lockstep with the field
// orderings defined by the FIELDS_F64 / FIELDS_I32 constants on the Rust side.

#include "SGP4.h"
#include <cstring>

extern "C" {

// Field counts — keep in sync with the Rust side.
const int CPP_DUMP_DOUBLE_COUNT = 112;
const int CPP_DUMP_INT_COUNT    = 5;

// Returns 0 on success, nonzero on sgp4init error (error code is also
// written to int_out[2]).
int cpp_sgp4init_dump(
    const char* satnum,         // null-terminated, used for satrec.satnum
    char    opsmode,            // 'i' (improved) or 'a' (AFSPC)
    double  epoch_sgp4,
    double  bstar,
    double  ndot,
    double  nddot,
    double  ecco,
    double  argpo,              // radians
    double  inclo,              // radians
    double  mo,                 // radians
    double  no_kozai,           // rad/min
    double  nodeo,              // radians
    int     epochyr,
    double  epochdays,
    double  jdsatepoch,
    double  jdsatepoch_frac,
    double* double_out,         // length CPP_DUMP_DOUBLE_COUNT
    int*    int_out             // length CPP_DUMP_INT_COUNT
) {
    elsetrec satrec = {};

    // Set the pre-init fields the same way our Rust path does.
    satrec.epochyr     = epochyr;
    satrec.epochdays   = epochdays;
    satrec.jdsatepoch  = jdsatepoch;
    satrec.jdsatepochF = jdsatepoch_frac;

    SGP4Funcs::sgp4init(
        wgs72, opsmode, satnum, epoch_sgp4,
        bstar, ndot, nddot, ecco, argpo,
        inclo, mo, no_kozai, nodeo, satrec
    );

    // Restore the split epoch the way our Rust caller does — initl may have
    // overwritten satrec.jdsatepoch with a derived value.
    satrec.jdsatepoch  = jdsatepoch;
    satrec.jdsatepochF = jdsatepoch_frac;

    // ── Dump f64 fields ──────────────────────────────────────────────
    int k = 0;

    // Near-Earth block (25)
    double_out[k++] = satrec.aycof;
    double_out[k++] = satrec.con41;
    double_out[k++] = satrec.cc1;
    double_out[k++] = satrec.cc4;
    double_out[k++] = satrec.cc5;
    double_out[k++] = satrec.d2;
    double_out[k++] = satrec.d3;
    double_out[k++] = satrec.d4;
    double_out[k++] = satrec.delmo;
    double_out[k++] = satrec.eta;
    double_out[k++] = satrec.argpdot;
    double_out[k++] = satrec.omgcof;
    double_out[k++] = satrec.sinmao;
    double_out[k++] = satrec.t;
    double_out[k++] = satrec.t2cof;
    double_out[k++] = satrec.t3cof;
    double_out[k++] = satrec.t4cof;
    double_out[k++] = satrec.t5cof;
    double_out[k++] = satrec.x1mth2;
    double_out[k++] = satrec.x7thm1;
    double_out[k++] = satrec.mdot;
    double_out[k++] = satrec.nodedot;
    double_out[k++] = satrec.xlcof;
    double_out[k++] = satrec.xmcof;
    double_out[k++] = satrec.nodecf;

    // Deep-space block (60)
    double_out[k++] = satrec.d2201;
    double_out[k++] = satrec.d2211;
    double_out[k++] = satrec.d3210;
    double_out[k++] = satrec.d3222;
    double_out[k++] = satrec.d4410;
    double_out[k++] = satrec.d4422;
    double_out[k++] = satrec.d5220;
    double_out[k++] = satrec.d5232;
    double_out[k++] = satrec.d5421;
    double_out[k++] = satrec.d5433;
    double_out[k++] = satrec.dedt;
    double_out[k++] = satrec.del1;
    double_out[k++] = satrec.del2;
    double_out[k++] = satrec.del3;
    double_out[k++] = satrec.didt;
    double_out[k++] = satrec.dmdt;
    double_out[k++] = satrec.dnodt;
    double_out[k++] = satrec.domdt;
    double_out[k++] = satrec.e3;
    double_out[k++] = satrec.ee2;
    double_out[k++] = satrec.peo;
    double_out[k++] = satrec.pgho;
    double_out[k++] = satrec.pho;
    double_out[k++] = satrec.pinco;
    double_out[k++] = satrec.plo;
    double_out[k++] = satrec.se2;
    double_out[k++] = satrec.se3;
    double_out[k++] = satrec.sgh2;
    double_out[k++] = satrec.sgh3;
    double_out[k++] = satrec.sgh4;
    double_out[k++] = satrec.sh2;
    double_out[k++] = satrec.sh3;
    double_out[k++] = satrec.si2;
    double_out[k++] = satrec.si3;
    double_out[k++] = satrec.sl2;
    double_out[k++] = satrec.sl3;
    double_out[k++] = satrec.sl4;
    double_out[k++] = satrec.gsto;
    double_out[k++] = satrec.xfact;
    double_out[k++] = satrec.xgh2;
    double_out[k++] = satrec.xgh3;
    double_out[k++] = satrec.xgh4;
    double_out[k++] = satrec.xh2;
    double_out[k++] = satrec.xh3;
    double_out[k++] = satrec.xi2;
    double_out[k++] = satrec.xi3;
    double_out[k++] = satrec.xl2;
    double_out[k++] = satrec.xl3;
    double_out[k++] = satrec.xl4;
    double_out[k++] = satrec.xlamo;
    double_out[k++] = satrec.zmol;
    double_out[k++] = satrec.zmos;
    double_out[k++] = satrec.atime;
    double_out[k++] = satrec.xli;
    double_out[k++] = satrec.xni;

    // Element / state block (16)
    double_out[k++] = satrec.a;
    double_out[k++] = satrec.altp;
    double_out[k++] = satrec.alta;
    double_out[k++] = satrec.epochdays;
    double_out[k++] = satrec.jdsatepoch;
    double_out[k++] = satrec.jdsatepochF;
    double_out[k++] = satrec.nddot;
    double_out[k++] = satrec.ndot;
    double_out[k++] = satrec.bstar;
    double_out[k++] = satrec.rcse;
    double_out[k++] = satrec.inclo;
    double_out[k++] = satrec.nodeo;
    double_out[k++] = satrec.ecco;
    double_out[k++] = satrec.argpo;
    double_out[k++] = satrec.mo;
    double_out[k++] = satrec.no_kozai;

    // Singly-averaged + unkozai (8)
    double_out[k++] = satrec.no_unkozai;
    double_out[k++] = satrec.am;
    double_out[k++] = satrec.em;
    double_out[k++] = satrec.im;
    double_out[k++] = satrec.Om;
    double_out[k++] = satrec.om;
    double_out[k++] = satrec.mm;
    double_out[k++] = satrec.nm;

    // Constants (8)
    double_out[k++] = satrec.tumin;
    double_out[k++] = satrec.mus;
    double_out[k++] = satrec.radiusearthkm;
    double_out[k++] = satrec.xke;
    double_out[k++] = satrec.j2;
    double_out[k++] = satrec.j3;
    double_out[k++] = satrec.j4;
    double_out[k++] = satrec.j3oj2;

    // Running total: 25 (near-Earth) + 55 (deep-space) + 16 (elements) +
    // 8 (singly-averaged) + 8 (constants) = 112. Must equal CPP_DUMP_DOUBLE_COUNT.

    // ── Dump i32 fields ──────────────────────────────────────────────
    int_out[0] = satrec.epochyr;
    int_out[1] = satrec.epochtynumrev;
    int_out[2] = satrec.error;
    int_out[3] = satrec.isimp;
    int_out[4] = satrec.irez;

    return satrec.error;
}

// Also expose a propagation entry so we can compare per-tsince r/v if needed.
int cpp_sgp4_step(
    const char* satnum,
    char    opsmode,            // 'i' (improved) or 'a' (AFSPC)
    double  epoch_sgp4,
    double  bstar,
    double  ndot,
    double  nddot,
    double  ecco,
    double  argpo,
    double  inclo,
    double  mo,
    double  no_kozai,
    double  nodeo,
    int     epochyr,
    double  epochdays,
    double  jdsatepoch,
    double  jdsatepoch_frac,
    double  tsince,
    double* r_out,
    double* v_out
) {
    elsetrec satrec = {};
    satrec.epochyr     = epochyr;
    satrec.epochdays   = epochdays;
    satrec.jdsatepoch  = jdsatepoch;
    satrec.jdsatepochF = jdsatepoch_frac;

    SGP4Funcs::sgp4init(
        wgs72, opsmode, satnum, epoch_sgp4,
        bstar, ndot, nddot, ecco, argpo,
        inclo, mo, no_kozai, nodeo, satrec
    );
    if (satrec.error != 0) return satrec.error;

    satrec.jdsatepoch  = jdsatepoch;
    satrec.jdsatepochF = jdsatepoch_frac;

    SGP4Funcs::sgp4(satrec, tsince, r_out, v_out);
    return satrec.error;
}

}  // extern "C"
