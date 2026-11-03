// SPDX-License-Identifier: Apache-2.0
//! Calls to hosted model providers. The policy layer (`client`) sits over any
//! transport, so the checks that keep a request on an allowed host, off
//! redirects and inside its limits are tested without a socket.

pub mod client;
pub mod secrets;
pub mod transport;
