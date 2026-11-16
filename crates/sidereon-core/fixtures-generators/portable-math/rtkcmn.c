/* Compile the unmodified RTKLIB algorithms with alternate math entry points. */
#include <math.h>
#include "rtklib.h"
extern double portable_sin(double), portable_cos(double), portable_sqrt(double);
extern double portable_asin(double), portable_atan(double), portable_atan2(double,double);
#define sin portable_sin
#define cos portable_cos
#define sqrt portable_sqrt
#define asin portable_asin
#define atan portable_atan
#define atan2 portable_atan2
#include <rtkcmn.c>
