// safetensors container: [u64 LE header_len][header json][tensor bytes].
//   data_offsets are relative to the end of the JSON header.
use memmap2::Mmap;
use serde_json::{Value, from_slice};
use std::fs::File;
use std::path::Path;

use crate::format::dtype::{Dtype, dtype_from_name};

pub struct Tensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub off: u64,
    pub len: u64,
}

pub struct Shard {
    pub mmap: Mmap,
    /// Byte offset where the tensor data region starts (end of header).
    pub data_start: u64,
    /// Number of bytes in the data region.
    pub data_len: u64,
    pub tensors: Vec<Tensor>,
}

pub fn open(p: &Path) -> Shard {
    let file = File::open(p).expect("open shard");
    let mmap = unsafe { Mmap::map(&file).expect("map shard") };
    let tensors = parse_header(&mmap);
    let header_len: u64 = read_u64_le(&mmap[0..8], 0);
    let data_start: u64 = 8 + header_len;
    let data_len = u64::try_from(mmap.len()).expect("len") - data_start;
    Shard {
        mmap,
        data_start,
        data_len,
        tensors,
    }
}

/// The whole data region as a slice of the mapping.
pub fn data_bytes(shard: &Shard) -> &[u8] {
    let start = usize::try_from(shard.data_start).expect("data_start usize");
    let len = usize::try_from(shard.data_len).expect("data_len usize");
    &shard.mmap[start..start + len]
}

fn read_u64_le(b: &[u8], off: usize) -> u64 {
    let a0: u64 = b[off].into();
    let a1: u64 = b[off + 1].into();
    let a2: u64 = b[off + 2].into();
    let a3: u64 = b[off + 3].into();
    let a4: u64 = b[off + 4].into();
    let a5: u64 = b[off + 5].into();
    let a6: u64 = b[off + 6].into();
    let a7: u64 = b[off + 7].into();
    a0 | (a1 << 8) | (a2 << 16) | (a3 << 24) | (a4 << 32) | (a5 << 40) | (a6 << 48) | (a7 << 56)
}

/// Parse the header JSON and build the tensor table relative to `data_start`.
fn parse_header(mmap: &Mmap) -> Vec<Tensor> {
    let header_len: u64 = read_u64_le(&mmap[0..8], 0);
    let data_start: u64 = 8 + header_len;
    let from: usize = 8;
    let end: usize = usize::try_from(data_start).expect("header end");
    let root: Value = from_slice(&mmap[from..end]).expect("parse safetensors header");

    let mut tensors = Vec::new();
    for (name, meta) in root.as_object().expect("header must be object").iter() {
        // Skip top-level safetensors metadata keys (__metadata__, __header__).
        if name.starts_with("__") {
            continue;
        }
        let dtype = dtype_from_name(meta["dtype"].as_str().expect("dtype str"));
        let offsets = meta["data_offsets"].as_array().expect("data_offsets array");
        let start: u64 = u64::try_from(offsets[0].as_i64().expect("start")).expect("start u64");
        let end: u64 = u64::try_from(offsets[1].as_i64().expect("end")).expect("end u64");
        tensors.push(Tensor {
            name: name.clone(),
            dtype,
            shape: shape_of(meta),
            off: data_start + start,
            len: end - start,
        });
    }
    tensors
}

fn shape_of(meta: &Value) -> Vec<u64> {
    let arr = meta["shape"].as_array().expect("shape array");
    arr.iter()
        .map(|v| u64::try_from(v.as_i64().expect("shape int")).expect("shape u64"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::id;

    #[test]
    fn parses_synthetic_shard() {
        // two tensors tile a 40-byte data region exactly
        let header = r#"{"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b":{"dtype":"BF16","shape":[8],"data_offsets":[24,40]}}"#;
        let mut file = Vec::<u8>::new();
        file.extend_from_slice(&header.len().to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&[0u8; 40]); // data blob
        let path = format!("/tmp/engine_export_{}.safetensors", id());
        fs::write(&path, &file).expect("write shard");
        let shard = open(Path::new(&path));
        let hl: u64 = u64::try_from(header.len()).expect("hl");
        let base: u64 = 8 + hl;
        assert_eq!(shard.tensors.len(), 2, "expected 2 tensors");
        let a = shard
            .tensors
            .iter()
            .find(|t| t.name == "a")
            .expect("tensor a");
        assert_eq!(a.off, base);
        assert_eq!(a.len, 24);
        let b = shard
            .tensors
            .iter()
            .find(|t| t.name == "b")
            .expect("tensor b");
        assert_eq!(b.off, base + 24);
        assert_eq!(b.len, 16);
        fs::remove_file(&path).ok();
    }
}
