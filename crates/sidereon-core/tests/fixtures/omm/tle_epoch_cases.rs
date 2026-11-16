use std::convert::TryFrom;

fn leap(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn main() {
    println!("# epoch_year\tepoch_day\tyear\tmonth\tday\thour\tminute\tsecond");
    for (year, units) in [
        (2019, 36_600_000_000_u64),
        (2025, 36_650_000_000),
        (2024, 36_699_999_999),
        (2025, 0),
        (2025, 50_000_000),
    ] {
        let epoch_year = year;
        let mut year = year;
        let mut day = i64::try_from(units / 100_000_000).unwrap() - 1;
        while day < 0 {
            year -= 1;
            day += if leap(year) { 366 } else { 365 };
        }
        while day >= if leap(year) { 366 } else { 365 } {
            day -= if leap(year) { 366 } else { 365 };
            year += 1;
        }
        let mut month = 1;
        let lengths = [31, 28 + i32::from(leap(year)), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        while day >= i64::from(lengths[month - 1]) {
            day -= i64::from(lengths[month - 1]);
            month += 1;
        }
        let micros = (units % 100_000_000) * 864;
        println!(
            "{epoch_year}\t{}.{:08}\t{year}\t{month}\t{}\t{}\t{}\t{}",
            units / 100_000_000,
            units % 100_000_000,
            day + 1,
            micros / 3_600_000_000,
            micros / 60_000_000 % 60,
            micros / 1_000_000 % 60
        );
    }
}
