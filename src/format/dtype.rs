// sandbox: pure data-type helpers for the safetensors / Metal stack.

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Dtype {
    Bool,
    U8,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    F64,
}

pub fn dtype_from_name(name: &str) -> Dtype {
    match name {
        "BOOL" => Dtype::Bool,
        "U8" => Dtype::U8,
        "U16" => Dtype::U16,
        "I16" => Dtype::I16,
        "F16" => Dtype::F16,
        "BF16" => Dtype::BF16,
        "U32" => Dtype::U32,
        "I32" => Dtype::I32,
        "F32" => Dtype::F32,
        "U64" => Dtype::U64,
        "F64" => Dtype::F64,
        _ => panic!("unknown dtype {name}"),
    }
}

#[inline(always)]
pub fn bf16_to_f32(bits: u16) -> f32 {
    let u: u32 = bits.into();
    f32::from_bits(u << 16)
}

#[inline(always)]
pub fn f32_to_bf16(v: f32) -> u16 {
    let hi: u32 = v.to_bits() >> 16;
    let lo: u8 = u8::try_from(hi & 0xFF).expect("bf16 lo");
    let hi_b: u8 = u8::try_from(hi >> 8).expect("bf16 hi");
    u16::from(lo) | (u16::from(hi_b) << 8)
}
