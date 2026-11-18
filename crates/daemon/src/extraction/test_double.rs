// SPDX-License-Identifier: Apache-2.0
//! A loopback HTTP server for tests: it records each request and answers with
//! scripted bytes, so the client is exercised against a real socket.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

pub struct Reply {
    pub delay: Duration,
    pub bytes: Vec<u8>,
}

pub struct Double {
    pub addr: SocketAddr,
    requests: Arc<Mutex<Vec<String>>>,
}

impl Double {
    pub fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

/// A 200 response with a fixed-length body.
pub fn ok(body: &str) -> Reply {
    raw(format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    ))
}

pub fn raw(text: impl Into<Vec<u8>>) -> Reply {
    Reply {
        delay: Duration::ZERO,
        bytes: text.into(),
    }
}

/// Serves `replies` one per connection, repeating the last.
pub async fn spawn_double(replies: Vec<Reply>) -> Double {
    serve(replies, None).await
}

/// Like `spawn_double`, and creates the file `ready` when the first request
/// has been recorded, so another process can wait for it.
pub async fn spawn_double_signalling(replies: Vec<Reply>, ready: PathBuf) -> Double {
    serve(replies, Some(ready)).await
}

async fn serve(replies: Vec<Reply>, ready: Option<PathBuf>) -> Double {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&requests);
    tokio::spawn(async move {
        let replies = Arc::new(replies);
        for index in 0..=usize::MAX {
            let (stream, _) = listener.accept().await.unwrap();
            let (log, replies, ready) = (Arc::clone(&log), Arc::clone(&replies), ready.clone());
            tokio::spawn(async move {
                let reply = &replies[index.min(replies.len() - 1)];
                let mut stream = BufReader::new(stream);
                let request = read_request(&mut stream).await;
                log.lock().unwrap().push(request);
                if let Some(ready) = &ready {
                    let _ = std::fs::write(ready, b"seen");
                }
                tokio::time::sleep(reply.delay).await;
                let _ = stream.get_mut().write_all(&reply.bytes).await;
                let _ = stream.get_mut().shutdown().await;
            });
        }
    });
    Double { addr, requests }
}

async fn read_request(stream: &mut BufReader<TcpStream>) -> String {
    let mut text = String::new();
    let mut length = 0;
    loop {
        let mut line = String::new();
        stream.read_line(&mut line).await.unwrap();
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap();
        }
        text.push_str(&line);
        if line == "\r\n" {
            break;
        }
    }
    let mut body = vec![0; length];
    stream.read_exact(&mut body).await.unwrap();
    text + &String::from_utf8(body).unwrap()
}
