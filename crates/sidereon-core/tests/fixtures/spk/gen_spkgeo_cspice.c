/*
 * Generates observe_de_spkgeo_cspice.json, the CSPICE spkgeo_c oracle for
 * crates/sidereon-core/tests/spk_cspice_oracle.rs.
 *
 * Build CSPICE N0067 from the NAIF source with floating-point contraction
 * off, so every multiply and add rounds separately as the Fortran source
 * states (a build that fuses multiply-adds differs in the last bits):
 *
 *   cd cspice/src/cspice
 *   setenv TKCOMPILEOPTIONS "-m64 -c -ansi -O2 -fPIC -ffp-contract=off -DNON_UNIX_STDIO -Wno-shift-op-parentheses -Wno-logical-op-parentheses -Wno-parentheses"
 *   csh mkprodct.csh
 *
 * (Keep the platform's other TKCOMPILEOPTIONS flags from mkprodct.csh and
 * add -ffp-contract=off.) Then compile and run this program against the full
 * de421.bsp, the kernel observe_de.bsp was excerpted from:
 *
 *   clang -O2 -ffp-contract=off -I cspice/include gen_spkgeo_cspice.c \
 *       cspice/lib/cspice.a -lm -o gen_spkgeo_cspice
 *   ./gen_spkgeo_cspice path/to/de421.bsp > observe_de_spkgeo_cspice.json
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "SpiceUsr.h"
static void hx(double v){uint64_t u;memcpy(&u,&v,8);printf("\"0x%016llx\"",(unsigned long long)u);}
int main(int argc, char **argv){
  if(argc!=2){fprintf(stderr,"usage: %s de421.bsp\n",argv[0]);return 2;}
  furnsh_c(argv[1]);
  if(failed_c()){fprintf(stderr,"cannot load %s\n",argv[1]);return 1;}
  double epochs[4]={750600000.0,760017600.0,751291200.0,755000000.5};
  int pairs[6][2]={{399,0},{3,399},{4,399},{10,399},{399,3},{5,4}};
  const char*frs[3]={"ECLIPJ2000","B1950","GALACTIC"};
  double rep[2]={751291200.0,755000000.5};
  int rot[2][2]={{4,399},{399,0}};
  int first=1; SpiceDouble st[6],lt;
  printf("{\"source\":\"CSPICE N0067 built from source with -ffp-contract=off, spkgeo_c on the full de421.bsp (NAIF generic_kernels/spk/planets/a_old_versions), the kernel observe_de.bsp was excerpted from by jplephem 2.24; CSPICE refuses the excerpt itself (SPICE(DAFBEGGTEND))\",\"cases\":[\n");
  #define EMIT(T,O,F,ET) do{ spkgeo_c(T,ET,F,O,st,&lt); if(failed_c()){fprintf(stderr,"fail\n");return 1;} \
    if(!first)printf(",\n"); first=0; printf("{\"target\":%d,\"observer\":%d,\"frame\":\"%s\",\"et\":%.17g,\"et_bits\":",T,O,F,ET); hx(ET); \
    printf(",\"state_bits\":["); for(int k=0;k<6;k++){ if(k)printf(","); hx(st[k]); } printf("],\"state\":["); \
    for(int k=0;k<6;k++){ if(k)printf(","); printf("%.17g",st[k]); } printf("]}"); }while(0)
  for(int p=0;p<6;p++) for(int e=0;e<4;e++) EMIT(pairs[p][0],pairs[p][1],"J2000",epochs[e]);
  for(int p=0;p<2;p++) for(int f=0;f<3;f++) for(int e=0;e<2;e++) EMIT(rot[p][0],rot[p][1],frs[f],rep[e]);
  printf("\n]}\n"); return 0; }
