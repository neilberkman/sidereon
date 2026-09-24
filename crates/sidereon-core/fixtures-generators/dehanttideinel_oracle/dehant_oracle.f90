! Oracle driver for the IERS Conventions (2010) Chapter 7 routine
! DEHANTTIDEINEL. It calls the unmodified IERS source for every case and writes
! the inputs and the displacement it returns as the elements of a JSON array on
! standard output; generate.sh wraps them into the fixture.
!
! Every real number is written with 17 significant digits, so a correctly
! rounded reader recovers the exact binary64 value the routine saw.
!
! The cases are the four test cases printed in the DEHANTTIDEINEL.F header,
! followed by a grid of stations and UTC dates. The grid spans the pre-UTC
! years (before 1960, where the SOFA DAT routine returns 0 s), the 1960-1971
! drift era, the leap-second era, and dates after the last table entry. The
! grid's Sun and Moon positions come from the low-precision formulas of the
! Astronomical Almanac and are rotated to Earth-fixed axes by the mean sidereal
! angle; they only have to be realistic, because the test passes the same
! numbers to both implementations.

program dehant_oracle
  implicit none
  double precision, parameter :: d2r = 3.14159265358979323846d0 / 180d0
  double precision, parameter :: au_m = 1.495978707d11
  integer, parameter :: nsta = 8, ndate = 13
  double precision :: stations(3, nsta)
  integer :: dates(3, ndate)
  double precision :: hours(ndate)
  double precision :: xsta(3), xsun(3), xmon(3)
  integer :: i, j, ncase
  character(len=64) :: id

  data stations / &
     4075578.385d0,   931852.890d0,  4801570.154d0, &
    -3950072.000d0,  2522415.000d0, -4311637.000d0, &
     1769693.000d0, -5044574.000d0, -3468321.000d0, &
    -1508022.000d0,  6195576.000d0,   148797.000d0, &
     1202433.000d0,   252632.000d0,  6237772.000d0, &
    -1311703.000d0,   310815.000d0, -6213255.000d0, &
     3839591.000d0, -5059567.000d0,   579957.000d0, &
    -3946713.000d0,  3366290.000d0,  3698912.000d0 /
  data dates / &
    1958,  6, 15, &
    1961,  8,  1, &
    1965,  3, 10, &
    1968,  2,  1, &
    1971, 12, 31, &
    1972,  1,  1, &
    1985,  7,  1, &
    1999,  1,  1, &
    2009,  4, 13, &
    2016, 12, 31, &
    2017,  1,  1, &
    2024,  2, 29, &
    2040,  6, 30 /
  data hours / 6.5d0, 0d0, 13.25d0, 22d0, 23.999d0, 0d0, 12d0, 3.75d0, &
               0d0, 23.5d0, 0.25d0, 18d0, 9d0 /

  ncase = 0

  ! The four header test cases of DEHANTTIDEINEL.F.
  call emit('header_case_1', &
    [4075578.385d0, 931852.890d0, 4801570.154d0], &
    [137859926952.015d0, 54228127881.4350d0, 23509422341.6960d0], &
    [-179996231.920342d0, -312468450.131567d0, -169288918.592160d0], &
    2009, 4, 13, 0.00d0)
  call emit('header_case_2', &
    [1112189.660d0, -4842955.026d0, 3985352.284d0], &
    [-54537460436.2357d0, 130244288385.279d0, 56463429031.5996d0], &
    [300396716.912d0, 243238281.451d0, 120548075.939d0], &
    2012, 7, 13, 0.00d0)
  call emit('header_case_3', &
    [1112200.5696d0, -4842957.8511d0, 3985345.9122d0], &
    [100210282451.6279d0, 103055630398.3160d0, 56855096480.4475d0], &
    [369817604.4348d0, 1897917.5258d0, 120804980.8284d0], &
    2015, 7, 15, 0.00d0)
  call emit('header_case_4', &
    [1112152.8166d0, -4842857.5435d0, 3985496.1783d0], &
    [8382471154.1312895d0, 10512408445.356153d0, -5360583240.3763866d0], &
    [380934092.93550891d0, 2871428.1904491195d0, 79015680.553570181d0], &
    2017, 1, 15, 0.00d0)

  do j = 1, ndate
    call sun_moon(dates(1, j), dates(2, j), dates(3, j), hours(j), xsun, xmon)
    do i = 1, nsta
      xsta = stations(:, i)
      write (id, '(a,i4.4,a,i2.2,a,i2.2,a,i1)') 'grid_', dates(1, j), '_', &
        dates(2, j), '_', dates(3, j), '_station_', i
      call emit(trim(id), xsta, xsun, xmon, dates(1, j), dates(2, j), &
        dates(3, j), hours(j))
    end do
  end do

  write (*, '(a)') ''

contains

  subroutine emit(name, sta, sun, mon, yr, mo, dy, fhr)
    character(len=*), intent(in) :: name
    double precision, intent(in) :: sta(3), sun(3), mon(3), fhr
    integer, intent(in) :: yr, mo, dy
    double precision :: s(3), su(3), mn(3), out(3), h

    s = sta
    su = sun
    mn = mon
    h = fhr
    call DEHANTTIDEINEL(s, yr, mo, dy, h, su, mn, out)
    if (ncase > 0) write (*, '(a)') ','
    ncase = ncase + 1
    write (*, '(a)', advance='no') '    {"id": "' // name // '", "xsta_m": ' // &
      vec(sta) // ', "xsun_m": ' // vec(sun) // ', "xmon_m": ' // vec(mon)
    write (*, '(a,i0,a,i0,a,i0,a)', advance='no') ', "year": ', yr, &
      ', "month": ', mo, ', "day": ', dy, ', "fhr_hours": '
    write (*, '(a)', advance='no') num(fhr) // ', "dxtide_m": ' // vec(out) // '}'
  end subroutine emit

  function num(x) result(text)
    double precision, intent(in) :: x
    character(len=:), allocatable :: text
    character(len=32) :: buf
    write (buf, '(es24.16e3)') x
    text = trim(adjustl(buf))
  end function num

  function vec(v) result(text)
    double precision, intent(in) :: v(3)
    character(len=:), allocatable :: text
    text = '[' // num(v(1)) // ', ' // num(v(2)) // ', ' // num(v(3)) // ']'
  end function vec

  ! Low-precision Sun and Moon (Astronomical Almanac, section C and D) in
  ! Earth-fixed metres, rotated by the mean sidereal angle only.
  subroutine sun_moon(yr, mo, dy, fhr, sun, mon)
    integer, intent(in) :: yr, mo, dy
    double precision, intent(in) :: fhr
    double precision, intent(out) :: sun(3), mon(3)
    double precision :: djm0, djm, d, g, q, lam, r, eps, gmst
    double precision :: lm, mm, fm, lamm, betm, dist, v(3)
    integer :: status

    call iau_CAL2JD(yr, mo, dy, djm0, djm, status)
    d = (djm0 - 2451545d0) + djm + fhr / 24d0
    eps = (23.439d0 - 0.00000036d0 * d) * d2r

    g = (357.529d0 + 0.98560028d0 * d) * d2r
    q = 280.459d0 + 0.98564736d0 * d
    lam = (q + 1.915d0 * sin(g) + 0.020d0 * sin(2d0 * g)) * d2r
    r = (1.00014d0 - 0.01671d0 * cos(g) - 0.00014d0 * cos(2d0 * g)) * au_m
    v = [r * cos(lam), r * cos(eps) * sin(lam), r * sin(eps) * sin(lam)]
    gmst = mod(280.46061837d0 + 360.98564736629d0 * d, 360d0) * d2r
    sun = [cos(gmst) * v(1) + sin(gmst) * v(2), -sin(gmst) * v(1) + cos(gmst) * v(2), v(3)]

    lm = 218.316d0 + 13.176396d0 * d
    mm = (134.963d0 + 13.064993d0 * d) * d2r
    fm = (93.272d0 + 13.229350d0 * d) * d2r
    lamm = (lm + 6.289d0 * sin(mm)) * d2r
    betm = 5.128d0 * sin(fm) * d2r
    dist = (385001d0 - 20905d0 * cos(mm)) * 1000d0
    v = [dist * cos(betm) * cos(lamm), &
         dist * (cos(eps) * cos(betm) * sin(lamm) - sin(eps) * sin(betm)), &
         dist * (sin(eps) * cos(betm) * sin(lamm) + cos(eps) * sin(betm))]
    mon = [cos(gmst) * v(1) + sin(gmst) * v(2), -sin(gmst) * v(1) + cos(gmst) * v(2), v(3)]
  end subroutine sun_moon

end program dehant_oracle

! DEHANTTIDEINEL.F calls CAL2JD and DAT, while the SOFA files distributed with
! it define iau_CAL2JD and iau_DAT. These two shims link the names unchanged.
subroutine CAL2JD(iy, im, id, djm0, djm, j)
  implicit none
  integer, intent(in) :: iy, im, id
  double precision, intent(out) :: djm0, djm
  integer, intent(out) :: j
  call iau_CAL2JD(iy, im, id, djm0, djm, j)
end subroutine CAL2JD

subroutine DAT(iy, im, id, fd, deltat, j)
  implicit none
  integer, intent(in) :: iy, im, id
  double precision, intent(in) :: fd
  double precision, intent(out) :: deltat
  integer, intent(out) :: j
  call iau_DAT(iy, im, id, fd, deltat, j)
end subroutine DAT
