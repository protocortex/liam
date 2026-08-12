// SPDX-License-Identifier: AGPL-3.0-only
//! Backend-neutral values and rows. Every backend converts these to and from
//! its own types, so the shared graph logic speaks one vocabulary. Rows are
//! materialized (owned), which sidesteps the sync-versus-async streaming
//! mismatch between rusqlite and libSQL.

use crate::error::{Error, Result};
use crate::ids::Millis;

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Int(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Real(v)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Text(v.to_string())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Text(v)
    }
}
impl From<Vec<u8>> for Value {
    fn from(v: Vec<u8>) -> Self {
        Value::Blob(v)
    }
}
impl From<Millis> for Value {
    fn from(v: Millis) -> Self {
        Value::Int(v.0)
    }
}

/// A materialized row: values addressed by column index.
#[derive(Clone, Debug)]
pub struct Row(pub Vec<Value>);

impl Row {
    pub fn get_i64(&self, i: usize) -> Result<i64> {
        match self.0.get(i) {
            Some(Value::Int(v)) => Ok(*v),
            _ => Err(Error::Backend(format!("column {i} is not an integer"))),
        }
    }
    pub fn get_string(&self, i: usize) -> Result<String> {
        match self.0.get(i) {
            Some(Value::Text(v)) => Ok(v.clone()),
            _ => Err(Error::Backend(format!("column {i} is not text"))),
        }
    }
}
