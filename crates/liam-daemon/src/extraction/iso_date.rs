// SPDX-License-Identifier: Apache-2.0
//! Strict `YYYY-MM-DD` parsing for model output.

use super::rules::is_calendar_date;

const ISO_DATE_LEN: usize = 10;
const SEPARATOR_POSITIONS: [usize; 2] = [4, 7];
const SEPARATOR: u8 = b'-';

/// Ordered by calendar position: year, then month, then day.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct IsoDate {
    pub year: u32,
    pub month: u32,
    pub day: u32,
}

/// A real calendar date in exactly `YYYY-MM-DD` form. The shape is checked
/// first because the field parsers accept signs and the calendar check slices
/// by byte.
pub fn parse_iso_date(text: &str) -> Option<IsoDate> {
    let bytes = text.as_bytes();
    let well_formed = bytes.len() == ISO_DATE_LEN
        && bytes.iter().enumerate().all(|(position, byte)| {
            if SEPARATOR_POSITIONS.contains(&position) {
                *byte == SEPARATOR
            } else {
                byte.is_ascii_digit()
            }
        });
    if !well_formed || !is_calendar_date(text) {
        return None;
    }
    let mut fields = text.split('-').map(|field| field.parse::<u32>().ok());
    Some(IsoDate {
        year: fields.next()??,
        month: fields.next()??,
        day: fields.next()??,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_real_iso_dates() {
        for (text, expected) in [
            ("2026-03-04", true),
            ("2024-02-29", true),
            ("2026-02-29", false),
            ("2026-13-45", false),
            ("2026-3-4", false),
            ("2026-+3-04", false),
            ("20260304", false),
            ("2026-03-04T10:00", false),
            ("é026-03-04", false),
            ("", false),
        ] {
            // Act
            let parsed = parse_iso_date(text);

            // Assert
            assert_eq!(parsed.is_some(), expected, "{text:?}");
        }
    }

    #[test]
    fn splits_the_fields() {
        // Act
        let parsed = parse_iso_date("1990-03-04");

        // Assert
        assert_eq!(
            parsed,
            Some(IsoDate {
                year: 1990,
                month: 3,
                day: 4
            })
        );
    }
}
