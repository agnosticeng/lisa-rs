use std::collections::HashMap;
use std::path::PathBuf;

use crate::format::dtype::Dtype;
use crate::format::safetensor::{self as st, Shard};

#[derive(Clone, Debug)]
pub struct WeightSlot {
    pub shard: usize,
    pub offset: usize,
    pub len: usize,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
}

pub struct WeightIndex {
    shards: Vec<Shard>,
    slots: HashMap<String, WeightSlot>,
}

impl WeightIndex {
    pub fn open(paths: &[PathBuf]) -> Self {
        let shards: Vec<_> = paths.iter().map(|path| st::open(path)).collect();
        let mut slots = HashMap::new();
        for (shard_index, shard) in shards.iter().enumerate() {
            for tensor in &shard.tensors {
                let previous = slots.insert(
                    tensor.name.clone(),
                    WeightSlot {
                        shard: shard_index,
                        offset: usize::try_from(tensor.off - shard.data_start)
                            .expect("tensor offset"),
                        len: usize::try_from(tensor.len).expect("tensor length"),
                        dtype: tensor.dtype,
                        shape: tensor.shape.clone(),
                    },
                );
                assert!(previous.is_none(), "duplicate tensor {}", tensor.name);
            }
        }
        Self { shards, slots }
    }

    pub fn slot(&self, name: &str) -> Option<&WeightSlot> {
        self.slots.get(name)
    }

    pub fn bytes(&self, name: &str) -> Option<&[u8]> {
        let slot = self.slot(name)?;
        let data = st::data_bytes(&self.shards[slot.shard]);
        Some(&data[slot.offset..slot.offset + slot.len])
    }

    pub fn rows(&self, name: &str, first: usize, count: usize) -> Option<&[u8]> {
        let slot = self.slot(name)?;
        let rows = usize::try_from(*slot.shape.first()?).ok()?;
        if first.checked_add(count)? > rows || slot.len % rows != 0 {
            return None;
        }
        let row_bytes = slot.len / rows;
        let start = first * row_bytes;
        let end = start + count * row_bytes;
        self.bytes(name)?.get(start..end)
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn shard_bytes(&self, shard: usize) -> Option<&[u8]> {
        self.shards.get(shard).map(st::data_bytes)
    }
}
