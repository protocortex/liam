// SPDX-License-Identifier: Apache-2.0
//! A minimal HTTP/1.1 client for the loopback sidecar. It speaks plain HTTP
//! only, sends `Connection: close`, never follows a redirect, and bounds every
//! read, so a misbehaving local server cannot stall or exhaust the daemon.
//! Hand-written because the only peer is a process on this machine: a general
//! client would bring a TLS stack and redirect logic this path must not have.

use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::time::timeout;

use super::client::ExtractError;

/// Longest status or header line accepted.
const MAX_LINE_BYTES: usize = 8 * 1024;
const MAX_HEADERS: usize = 64;
const HEX_RADIX: u32 = 16;
const CHUNK_TERMINATOR: &[u8; 2] = b"\r\n";

pub struct Request<'a> {
    pub method: &'static str,
    pub path: &'a str,
    pub bearer: Option<&'a str>,
    pub body: &'a [u8],
}

pub struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

#[derive(Clone, Copy)]
pub struct Limits {
    /// Covers connect, send, and the whole read.
    pub timeout: Duration,
    pub max_body_bytes: usize,
}

enum Framing {
    Length(usize),
    Chunked,
    UntilClose,
}

/// Rejects a value that could add or split a header line.
pub fn check_header_value(value: &str) -> Result<(), ExtractError> {
    if value.bytes().any(|byte| byte.is_ascii_control()) {
        return Err(ExtractError::InvalidRequest(
            "a header value has a control byte",
        ));
    }
    Ok(())
}

pub async fn send(
    addr: SocketAddr,
    request: &Request<'_>,
    limits: Limits,
) -> Result<Response, ExtractError> {
    check_header_value(request.path)?;
    if let Some(key) = request.bearer {
        check_header_value(key)?;
    }
    timeout(
        limits.timeout,
        exchange(addr, request, limits.max_body_bytes),
    )
    .await
    .map_err(|_| ExtractError::Timeout)?
}

async fn exchange(
    addr: SocketAddr,
    request: &Request<'_>,
    max_body_bytes: usize,
) -> Result<Response, ExtractError> {
    let mut stream = TcpStream::connect(addr)
        .await
        .map_err(|error| ExtractError::Connect(error.kind()))?;
    stream
        .write_all(&encode(addr, request))
        .await
        .map_err(io_error)?;
    let mut reader = BufReader::new(stream);
    let status = read_status(&mut reader).await?;
    let framing = read_framing(&mut reader).await?;
    let body = read_body(&mut reader, framing, max_body_bytes).await?;
    Ok(Response { status, body })
}

fn encode(addr: SocketAddr, request: &Request<'_>) -> Vec<u8> {
    let mut head = format!(
        "{} {} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n",
        request.method, request.path
    );
    if let Some(key) = request.bearer {
        head.push_str(&format!("Authorization: Bearer {key}\r\n"));
    }
    if !request.body.is_empty() {
        head.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            request.body.len()
        ));
    }
    head.push_str("\r\n");
    let mut bytes = head.into_bytes();
    bytes.extend_from_slice(request.body);
    bytes
}

fn io_error(error: io::Error) -> ExtractError {
    match error.kind() {
        io::ErrorKind::UnexpectedEof => ExtractError::Truncated,
        kind => ExtractError::Io(kind),
    }
}

async fn read_line<R: AsyncBufRead + Unpin>(reader: &mut R) -> Result<String, ExtractError> {
    let mut line = Vec::new();
    (&mut *reader)
        .take(MAX_LINE_BYTES as u64)
        .read_until(b'\n', &mut line)
        .await
        .map_err(io_error)?;
    if !line.ends_with(b"\n") {
        return Err(if line.len() >= MAX_LINE_BYTES {
            ExtractError::Malformed("a header line is too long")
        } else {
            ExtractError::Truncated
        });
    }
    let text =
        String::from_utf8(line).map_err(|_| ExtractError::Malformed("a header is not UTF-8"))?;
    Ok(text.trim_end_matches(['\r', '\n']).to_owned())
}

async fn read_status<R: AsyncBufRead + Unpin>(reader: &mut R) -> Result<u16, ExtractError> {
    let line = read_line(reader).await?;
    let mut parts = line.split_whitespace();
    let version = parts.next().unwrap_or_default();
    let status = parts.next().and_then(|code| code.parse().ok());
    match status {
        Some(code) if version.starts_with("HTTP/1.") => Ok(code),
        _ => Err(ExtractError::Malformed("the status line is invalid")),
    }
}

async fn read_framing<R: AsyncBufRead + Unpin>(reader: &mut R) -> Result<Framing, ExtractError> {
    let mut length: Option<usize> = None;
    let mut chunked = false;
    for _ in 0..=MAX_HEADERS {
        let line = read_line(reader).await?;
        if line.is_empty() {
            return Ok(match (chunked, length) {
                (true, _) => Framing::Chunked,
                (false, Some(length)) => Framing::Length(length),
                (false, None) => Framing::UntilClose,
            });
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(ExtractError::Malformed("a header has no colon"));
        };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => {
                let parsed = value
                    .parse()
                    .map_err(|_| ExtractError::Malformed("the content length is invalid"))?;
                if length.is_some_and(|seen| seen != parsed) {
                    return Err(ExtractError::Malformed("the content lengths disagree"));
                }
                length = Some(parsed);
            }
            "transfer-encoding" => chunked |= value.to_ascii_lowercase().contains("chunked"),
            _ => {}
        }
    }
    Err(ExtractError::Malformed("the response has too many headers"))
}

async fn read_body<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    framing: Framing,
    max: usize,
) -> Result<Vec<u8>, ExtractError> {
    match framing {
        Framing::Length(length) => {
            check_size(length, max)?;
            read_exact_vec(reader, length).await
        }
        Framing::UntilClose => {
            let mut body = Vec::new();
            (&mut *reader)
                .take(max as u64 + 1)
                .read_to_end(&mut body)
                .await
                .map_err(io_error)?;
            check_size(body.len(), max)?;
            Ok(body)
        }
        Framing::Chunked => read_chunked(reader, max).await,
    }
}

async fn read_chunked<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    max: usize,
) -> Result<Vec<u8>, ExtractError> {
    let mut body = Vec::new();
    loop {
        let line = read_line(reader).await?;
        let size_text = line.split(';').next().unwrap_or_default().trim();
        let size = usize::from_str_radix(size_text, HEX_RADIX)
            .map_err(|_| ExtractError::Malformed("a chunk size is invalid"))?;
        if size == 0 {
            skip_trailers(reader).await?;
            return Ok(body);
        }
        check_size(body.len().saturating_add(size), max)?;
        body.extend(read_exact_vec(reader, size).await?);
        if read_exact_vec(reader, CHUNK_TERMINATOR.len()).await? != CHUNK_TERMINATOR {
            return Err(ExtractError::Malformed("a chunk is not terminated"));
        }
    }
}

async fn skip_trailers<R: AsyncBufRead + Unpin>(reader: &mut R) -> Result<(), ExtractError> {
    for _ in 0..=MAX_HEADERS {
        if read_line(reader).await?.is_empty() {
            return Ok(());
        }
    }
    Err(ExtractError::Malformed(
        "the response has too many trailers",
    ))
}

fn check_size(size: usize, limit: usize) -> Result<(), ExtractError> {
    if size > limit {
        return Err(ExtractError::TooLarge { limit });
    }
    Ok(())
}

async fn read_exact_vec<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    len: usize,
) -> Result<Vec<u8>, ExtractError> {
    let mut bytes = vec![0; len];
    reader.read_exact(&mut bytes).await.map_err(io_error)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use tokio::io::AsyncWriteExt;
    use tokio::net::{TcpListener, TcpStream};

    use super::super::test_double::{ok, raw, spawn_double, Reply};
    use super::*;

    const REQUEST_TIMEOUT: Duration = Duration::from_millis(200);
    const TIMEOUT_WINDOW_MAX: Duration = Duration::from_secs(2);
    const STALL: Duration = Duration::from_secs(10);
    const SMALL_CAP: usize = 16;
    const GENEROUS_CAP: usize = 1024;
    const BODY: &str = r#"{"content":"hi"}"#;

    /// An address nothing listens on. The port is probed once and re-picked
    /// if another process took it between the drop and the probe.
    async fn closed_address() -> SocketAddr {
        for _ in 0..2 {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            drop(listener);
            if TcpStream::connect(addr).await.is_err() {
                return addr;
            }
        }
        panic!("no unused loopback port found");
    }

    fn get(path: &str) -> Request<'_> {
        Request {
            method: "GET",
            path,
            bearer: None,
            body: &[],
        }
    }

    async fn fetch(reply: Reply, max_body_bytes: usize) -> Result<Response, ExtractError> {
        let double = spawn_double(vec![reply]).await;
        let limits = Limits {
            timeout: REQUEST_TIMEOUT,
            max_body_bytes,
        };
        send(double.addr, &get("/x"), limits).await
    }

    #[tokio::test]
    async fn a_post_carries_the_bearer_key_the_length_and_the_body() {
        // Arrange
        let double = spawn_double(vec![ok(BODY)]).await;
        let request = Request {
            method: "POST",
            path: "/completion",
            bearer: Some("secret"),
            body: b"{}",
        };
        let limits = Limits {
            timeout: REQUEST_TIMEOUT,
            max_body_bytes: GENEROUS_CAP,
        };

        // Act
        let response = send(double.addr, &request, limits).await.unwrap();

        // Assert
        assert_eq!((response.status, response.body), (200, BODY.into()));
        let sent = &double.requests()[0];
        assert!(sent.starts_with("POST /completion HTTP/1.1\r\n"));
        assert!(sent.contains("Authorization: Bearer secret\r\n"));
        assert!(sent.contains("Content-Length: 2\r\n"));
        assert!(sent.contains("Connection: close\r\n"));
        assert!(sent.ends_with("\r\n\r\n{}"));
    }

    #[tokio::test]
    async fn chunked_and_fixed_length_bodies_are_read_in_full() {
        let (head, tail) = BODY.split_at(5);
        let chunked = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{head}\r\n{:x};ext=1\r\n{tail}\r\n0\r\nX-Trailer: 1\r\n\r\n",
            head.len(),
            tail.len()
        );
        for reply in [raw(chunked), ok(BODY)] {
            // Act
            let response = fetch(reply, GENEROUS_CAP).await.unwrap();

            // Assert
            assert_eq!(response.body, BODY.as_bytes());
        }
    }

    #[tokio::test]
    async fn a_body_without_framing_is_read_until_the_connection_closes() {
        // Act
        let response = fetch(raw(format!("HTTP/1.1 200 OK\r\n\r\n{BODY}")), GENEROUS_CAP)
            .await
            .unwrap();

        // Assert
        assert_eq!(response.body, BODY.as_bytes());
    }

    #[tokio::test]
    async fn a_cut_or_malformed_response_is_a_typed_error() {
        let cases = [
            (
                "HTTP/1.1 200 OK\r\nContent-Length: 50\r\n\r\n{}",
                ExtractError::Truncated,
            ),
            (
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\na\r\nabc",
                ExtractError::Truncated,
            ),
            ("HTTP/1.1 200 OK\r\nContent-Le", ExtractError::Truncated),
            (
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n",
                ExtractError::Malformed("a chunk size is invalid"),
            ),
            (
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nabXX",
                ExtractError::Malformed("a chunk is not terminated"),
            ),
            (
                "HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n{}",
                ExtractError::Malformed("the content lengths disagree"),
            ),
            (
                "HTTP/1.1 200 OK\r\nContent-Length: x\r\n\r\n",
                ExtractError::Malformed("the content length is invalid"),
            ),
            (
                "HTTP/1.1 200 OK\r\nno colon here\r\n\r\n",
                ExtractError::Malformed("a header has no colon"),
            ),
            (
                "garbage\r\n\r\n",
                ExtractError::Malformed("the status line is invalid"),
            ),
        ];
        for (reply, expected) in cases {
            // Act
            let result = fetch(raw(reply), GENEROUS_CAP).await;

            // Assert
            assert_eq!(result.err(), Some(expected), "reply {reply:?}");
        }
    }

    #[tokio::test]
    async fn a_response_over_the_size_cap_is_rejected() {
        let over = "x".repeat(SMALL_CAP + 1);
        let replies = [
            format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", over.len()),
            format!(
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{over}\r\n0\r\n\r\n",
                over.len()
            ),
            format!("HTTP/1.1 200 OK\r\n\r\n{over}"),
        ];
        for reply in replies {
            // Act
            let result = fetch(raw(reply.clone()), SMALL_CAP).await;

            // Assert
            assert_eq!(
                result.err(),
                Some(ExtractError::TooLarge { limit: SMALL_CAP }),
                "reply {reply:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_redirect_is_returned_as_is_and_not_followed() {
        // Arrange
        let target = spawn_double(vec![ok(BODY)]).await;
        let redirect = format!(
            "HTTP/1.1 302 Found\r\nLocation: http://{}/x\r\nContent-Length: 0\r\n\r\n",
            target.addr
        );

        // Act
        let response = fetch(raw(redirect), GENEROUS_CAP).await.unwrap();

        // Assert
        assert_eq!(response.status, 302);
        assert!(target.requests().is_empty());
    }

    #[tokio::test]
    async fn a_slow_response_times_out_inside_the_window() {
        // Arrange
        let slow = Reply {
            delay: STALL,
            ..ok(BODY)
        };

        // Act
        let started = Instant::now();
        let result = fetch(slow, GENEROUS_CAP).await;
        let elapsed = started.elapsed();

        // Assert
        assert_eq!(result.err(), Some(ExtractError::Timeout));
        assert!(elapsed >= REQUEST_TIMEOUT && elapsed < TIMEOUT_WINDOW_MAX);
    }

    #[tokio::test]
    async fn a_refused_connection_is_a_typed_error() {
        // Arrange
        let addr = closed_address().await;
        let limits = Limits {
            timeout: REQUEST_TIMEOUT,
            max_body_bytes: GENEROUS_CAP,
        };

        // Act
        let result = send(addr, &get("/x"), limits).await;

        // Assert
        assert_eq!(
            result.err(),
            Some(ExtractError::Connect(io::ErrorKind::ConnectionRefused))
        );
    }

    #[tokio::test]
    async fn two_transfer_encoding_headers_still_mean_chunked() {
        // Arrange
        let reply = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nTransfer-Encoding: identity\r\n\r\n2\r\nhi\r\n0\r\n\r\n";

        // Act
        let response = fetch(raw(reply), GENEROUS_CAP).await.unwrap();

        // Assert
        assert_eq!(response.body, b"hi");
    }

    #[tokio::test]
    async fn a_header_line_over_the_limit_is_rejected() {
        // Arrange
        let reply = format!(
            "HTTP/1.1 200 OK\r\nX-Pad: {}\r\n\r\n",
            "x".repeat(MAX_LINE_BYTES)
        );

        // Act
        let result = fetch(raw(reply), GENEROUS_CAP).await;

        // Assert
        assert_eq!(
            result.err(),
            Some(ExtractError::Malformed("a header line is too long"))
        );
    }

    #[tokio::test]
    async fn the_header_count_is_capped() {
        let headers = |count: usize| "X-H: 1\r\n".repeat(count);
        let reply = |count| {
            format!(
                "HTTP/1.1 200 OK\r\n{}Content-Length: 0\r\n\r\n",
                headers(count)
            )
        };

        // Act
        let at_cap = fetch(raw(reply(MAX_HEADERS - 1)), GENEROUS_CAP).await;
        let over_cap = fetch(raw(reply(MAX_HEADERS)), GENEROUS_CAP).await;

        // Assert
        assert!(at_cap.is_ok());
        assert_eq!(
            over_cap.err(),
            Some(ExtractError::Malformed("the response has too many headers"))
        );
    }

    #[tokio::test]
    async fn the_trailer_count_is_capped() {
        // Arrange
        let trailers = "X-T: 1\r\n".repeat(MAX_HEADERS + 1);
        let reply =
            format!("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n{trailers}\r\n");

        // Act
        let result = fetch(raw(reply), GENEROUS_CAP).await;

        // Assert
        assert_eq!(
            result.err(),
            Some(ExtractError::Malformed(
                "the response has too many trailers"
            ))
        );
    }

    #[tokio::test]
    async fn an_endless_unframed_body_is_cut_at_the_cap_before_the_deadline() {
        // Arrange
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = stream.write_all(b"HTTP/1.1 200 OK\r\n\r\n").await;
            let filler = [b'x'; 4096];
            while stream.write_all(&filler).await.is_ok() {}
        });
        let limits = Limits {
            timeout: REQUEST_TIMEOUT,
            max_body_bytes: SMALL_CAP,
        };

        // Act
        let result = send(addr, &get("/x"), limits).await;

        // Assert
        assert_eq!(
            result.err(),
            Some(ExtractError::TooLarge { limit: SMALL_CAP })
        );
    }

    #[tokio::test]
    async fn a_control_byte_in_the_key_or_path_is_rejected_before_connecting() {
        let limits = Limits {
            timeout: REQUEST_TIMEOUT,
            max_body_bytes: GENEROUS_CAP,
        };
        for (path, bearer) in [
            ("/x", Some("key\r\nX-Evil: 1")),
            ("/x", Some("key\0")),
            ("/x\r\nX-Evil: 1", None),
        ] {
            // Arrange
            let double = spawn_double(vec![ok(BODY)]).await;
            let request = Request {
                method: "GET",
                path,
                bearer,
                body: &[],
            };

            // Act
            let result = send(double.addr, &request, limits).await;

            // Assert
            assert!(matches!(result, Err(ExtractError::InvalidRequest(_))));
            assert!(double.requests().is_empty());
        }
    }
}
