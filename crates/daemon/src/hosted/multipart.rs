// SPDX-License-Identifier: Apache-2.0
//! A `multipart/form-data` body, built in memory. Only OpenAI's file upload
//! needs one.

use sha2::{Digest, Sha256};

/// Hex characters of the content digest used in the boundary.
const BOUNDARY_CHARS: usize = 32;
const CRLF: &[u8] = b"\r\n";

struct Part {
    disposition: String,
    content_type: Option<&'static str>,
    data: Vec<u8>,
}

/// A form being built. Names and file names are fixed strings in the caller,
/// so none needs quoting.
#[derive(Default)]
pub struct Form {
    parts: Vec<Part>,
}

/// A finished form: the header value and the body.
pub struct Encoded {
    pub content_type: String,
    pub body: Vec<u8>,
}

impl Form {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(mut self, name: &'static str, value: &str) -> Self {
        self.parts.push(Part {
            disposition: format!("form-data; name=\"{name}\""),
            content_type: None,
            data: value.as_bytes().to_vec(),
        });
        self
    }

    pub fn file(
        mut self,
        name: &'static str,
        file_name: &'static str,
        content_type: &'static str,
        data: Vec<u8>,
    ) -> Self {
        self.parts.push(Part {
            disposition: format!("form-data; name=\"{name}\"; filename=\"{file_name}\""),
            content_type: Some(content_type),
            data,
        });
        self
    }

    /// The boundary is a digest of the parts, extended until no part holds it,
    /// so the same form encodes the same way and the body can never end early.
    pub fn encode(self) -> Encoded {
        let mut digest = Sha256::new();
        for part in &self.parts {
            digest.update(part.disposition.as_bytes());
            digest.update(&part.data);
        }
        let hex: String = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let boundary = unused_in(&self.parts, format!("liam-{}", &hex[..BOUNDARY_CHARS]));
        let mut body = Vec::new();
        for part in &self.parts {
            body.extend_from_slice(format!("--{boundary}").as_bytes());
            body.extend_from_slice(CRLF);
            body.extend_from_slice(format!("Content-Disposition: {}", part.disposition).as_bytes());
            body.extend_from_slice(CRLF);
            if let Some(content_type) = part.content_type {
                body.extend_from_slice(format!("Content-Type: {content_type}").as_bytes());
                body.extend_from_slice(CRLF);
            }
            body.extend_from_slice(CRLF);
            body.extend_from_slice(&part.data);
            body.extend_from_slice(CRLF);
        }
        body.extend_from_slice(format!("--{boundary}--").as_bytes());
        body.extend_from_slice(CRLF);
        Encoded {
            content_type: format!("multipart/form-data; boundary={boundary}"),
            body,
        }
    }
}

/// `boundary`, lengthened until no part's data holds it.
fn unused_in(parts: &[Part], mut boundary: String) -> String {
    while parts.iter().any(|part| contains(&part.data, &boundary)) {
        boundary.push('x');
    }
    boundary
}

fn contains(data: &[u8], needle: &str) -> bool {
    let needle = needle.as_bytes();
    data.windows(needle.len()).any(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boundary_of(encoded: &Encoded) -> String {
        encoded
            .content_type
            .strip_prefix("multipart/form-data; boundary=")
            .unwrap()
            .to_owned()
    }

    #[test]
    fn a_form_encodes_its_fields_and_file_between_the_boundary_lines() {
        // Arrange
        let form = Form::new().text("purpose", "batch").file(
            "file",
            "requests.jsonl",
            "application/jsonl",
            b"{\"a\":1}\n".to_vec(),
        );

        // Act
        let encoded = form.encode();

        // Assert
        let boundary = boundary_of(&encoded);
        let expected = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"requests.jsonl\"\r\n\
             Content-Type: application/jsonl\r\n\r\n{{\"a\":1}}\n\r\n--{boundary}--\r\n"
        );
        assert_eq!(String::from_utf8(encoded.body).unwrap(), expected);
    }

    #[test]
    fn the_same_form_encodes_the_same_way_and_a_different_one_does_not() {
        // Arrange
        let build = |value: &str| Form::new().text("purpose", value).encode();

        // Act
        let first = build("batch");
        let again = build("batch");
        let other = build("other");

        // Assert
        assert_eq!(first.body, again.body);
        assert_ne!(boundary_of(&first), boundary_of(&other));
    }

    #[test]
    fn a_boundary_that_a_part_holds_is_lengthened_until_none_does() {
        // Arrange
        let parts = vec![Part {
            disposition: String::new(),
            content_type: None,
            data: b"has liam-abc and liam-abcx inside".to_vec(),
        }];

        // Act
        let free = unused_in(&parts, "liam-zzz".to_owned());
        let lengthened = unused_in(&parts, "liam-abc".to_owned());

        // Assert
        assert_eq!(free, "liam-zzz");
        assert_eq!(lengthened, "liam-abcxx");
    }
}
