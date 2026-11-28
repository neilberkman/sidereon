#include "SGP4.h"

extern "C" double sgp4_oracle_sin(double);
extern "C" double sgp4_oracle_cos(double);
extern "C" double sgp4_oracle_atan2(double, double);
extern "C" double sgp4_oracle_pow(double, double);

#define sin sgp4_oracle_sin
#define cos sgp4_oracle_cos
#define atan2 sgp4_oracle_atan2
#define pow sgp4_oracle_pow
#include "SGP4.cpp"
