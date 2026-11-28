double portable_sin(double);
double portable_cos(double);
double portable_atan2(double, double);

double __wrap_sin(double x) { return portable_sin(x); }
double __wrap_cos(double x) { return portable_cos(x); }
double __wrap_atan2(double y, double x) { return portable_atan2(y, x); }
