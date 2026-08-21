// Minimal Metal host bridge over the Metal C API + Objective-C runtime.
// No external crate: we bind MTLCreateSystemDefaultDevice (a C function) and
// talk to object methods through objc_msgSend with registered selectors.
#![allow(non_upper_case_globals)]
#![allow(clashing_extern_declarations)]

use std::ffi::{CStr, CString, c_char, c_void};

#[cfg(target_os = "macos")]
#[link(kind = "framework", name = "Metal")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *const c_void;
}

// Objective-C runtime (libobjc) + Foundation (NSString etc).
#[cfg(target_os = "macos")]
#[link(name = "objc")]
#[link(kind = "framework", name = "CoreFoundation")]
unsafe extern "C" {
    #[link_name = "objc_msgSend"]
    fn msgsend_1(recv: *const c_void, op: *const c_void, a: *const c_void) -> *const c_void;
    #[link_name = "objc_msgSend"]
    fn msgsend_2(
        recv: *const c_void,
        op: *const c_void,
        a: *const c_void,
        b: *const c_void,
    ) -> *const c_void;
    #[link_name = "objc_msgSend"]
    fn msgsend_3(
        recv: *const c_void,
        op: *const c_void,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
    ) -> *const c_void;
    #[link_name = "objc_msgSend"]
    fn msgsend_void(recv: *const c_void, op: *const c_void) -> ();
    #[link_name = "objc_msgSend"]
    fn msgsend_p_u_u(
        recv: *const c_void,
        op: *const c_void,
        a: *const c_void,
        b: u64,
        c: u64,
    ) -> ();
    #[link_name = "objc_msgSend"]
    fn msgsend_pipeline(recv: *const c_void, op: *const c_void, a: *const c_void) -> ();
    #[link_name = "objc_msgSend"]
    fn msgsend_p_u(recv: *const c_void, op: *const c_void, a: u64) -> ();
    #[link_name = "objc_msgSend"]
    fn msgsend_u_u(recv: *const c_void, op: *const c_void, a: u64, b: u64) -> *const c_void;
    #[link_name = "objc_msgSend"]
    fn msgsend_dispatch(recv: *const c_void, op: *const c_void, grid: MTLSize, tptg: MTLSize)
    -> ();
    #[link_name = "objc_msgSend"]
    fn msgsend_groups(recv: *const c_void, op: *const c_void, groups: MTLSize, tptg: MTLSize)
    -> ();
    fn sel_registerName(name: *const c_char) -> *const c_void;
    fn objc_getClass(name: *const c_char) -> *const c_void;
    fn objc_msgSend(receiver: *const c_void, op: *const c_void, ...) -> *const c_void;
    fn CFStringCreateWithCString(
        alloc: *const c_void,
        c_str: *const c_char,
        encoding: u32,
    ) -> *const c_void;
}

/// MTLSize: three NSUIntegers, passed by value.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MTLSize {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

/// kCFStringEncodingUTF8
const C_ENCODING_UTF8: u32 = 0x0800_0100;

/// MTLResourceStorageModeShared (0) → CPU+GPU unified access (for weights).
pub const STORAGE_MODE_SHARED: usize = 0;
pub const HAZARD_TRACKING_UNTRACKED: usize = 1 << 8;

pub struct MetalDevice {
    pub id: *const c_void,
}

pub struct MetalBuffer {
    pub id: *const c_void,
}

pub struct CommandQueue {
    pub id: *const c_void,
}

pub struct CommandBuffer {
    pub id: *const c_void,
}

/// Compiled shader source (id<MTLLibrary>).
pub struct Library {
    pub id: *const c_void,
}

/// A named entry point inside a library (id<MTLFunction>).
pub struct Function {
    pub id: *const c_void,
}

/// Ready-to-dispatch compute kernel (id<MTLComputePipelineState>).
pub struct ComputePipeline {
    pub id: *const c_void,
}

/// A live compute encoder attached to a command buffer.
pub struct ComputeEncoder {
    pub id: *const c_void,
}

/// A dedicated MTLHeap. Buffers carved from a heap share ONE storage option, so
/// they never collide with Metal's mixed-option device sub-allocator.
pub struct Heap {
    pub id: *const c_void,
}

// SAFETY: these are opaque, immutable handles to Metal objects (MTLDevice,
// MTLBuffer, MTLCommandQueue, ...) which Apple documents as thread-safe. The
// serving layer serializes all mutation through a mutex, and buffers are only
// released by their owning runner's Drop while that runner is exclusively held.
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}
unsafe impl Send for CommandQueue {}
unsafe impl Sync for CommandQueue {}
unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}
unsafe impl Send for Library {}
unsafe impl Sync for Library {}
unsafe impl Send for Function {}
unsafe impl Sync for Function {}
unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}
unsafe impl Send for ComputeEncoder {}
unsafe impl Sync for ComputeEncoder {}
unsafe impl Send for Heap {}
unsafe impl Sync for Heap {}

impl Default for MetalDevice {
    fn default() -> Self {
        let id = unsafe { MTLCreateSystemDefaultDevice() };
        assert!(!id.is_null(), "no Metal device (no GPU?)");
        MetalDevice { id }
    }
}

impl MetalDevice {
    /// Query the human-readable device name via `[MTLDevice name]`.
    pub fn name(&self) -> String {
        let ns_name = unsafe { objc_msgSend(self.id, sel("name")) };
        assert!(!ns_name.is_null(), "nil device name");
        let cstr = unsafe { objc_msgSend(ns_name, sel("UTF8String")) as *const c_char };
        assert!(!cstr.is_null(), "nil UTF8String");
        unsafe { CStr::from_ptr(cstr) }
            .to_string_lossy()
            .into_owned()
    }

    /// recommendedMaxWorkingSetSize (uint64) for the active GPU.
    pub fn max_working_set(&self) -> u64 {
        unsafe { objc_msgSend(self.id, sel("recommendedMaxWorkingSetSize")) as u64 }
    }

    /// Allocate a shared-storage buffer of `len` bytes.
    pub fn new_buffer(&self, len: usize) -> MetalBuffer {
        self.new_buffer_with_options(len, STORAGE_MODE_SHARED)
    }

    /// Allocate an immutable/shared buffer without Metal hazard bookkeeping.
    /// The caller must not write it while GPU work can reference it.
    pub fn new_untracked_buffer(&self, len: usize) -> MetalBuffer {
        self.new_buffer_with_options(len, STORAGE_MODE_SHARED | HAZARD_TRACKING_UNTRACKED)
    }

    fn new_buffer_with_options(&self, len: usize, options: usize) -> MetalBuffer {
        let id = unsafe {
            msgsend_u_u(
                self.id,
                sel("newBufferWithLength:options:"),
                len as u64,
                options as u64,
            )
        };
        assert!(!id.is_null(), "newBufferWithLength failed");
        MetalBuffer { id }
    }

    /// Create a command queue from this device.
    pub fn new_command_queue(&self) -> CommandQueue {
        let id = unsafe { objc_msgSend(self.id, sel("newCommandQueue")) };
        assert!(!id.is_null(), "newCommandQueue failed");
        CommandQueue { id }
    }

    /// Build a shared-storage MTLHeap of `size` bytes.
    pub fn new_heap(&self, size: usize) -> Heap {
        with_autorelease_pool(|| {
            let cls = objc_get_c_class("MTLHeapDescriptor");
            let raw = unsafe { objc_msgSend(cls, sel("alloc")) };
            let desc = unsafe { objc_msgSend(raw, sel("init")) };
            assert!(!desc.is_null(), "MTLHeapDescriptor init failed");
            // size (bytes) + StorageModeShared (0)
            unsafe {
                msgsend_p_u(desc, sel("setSize:"), size as u64);
                msgsend_p_u(desc, sel("setStorageMode:"), 0);
            }
            let id = unsafe { msgsend_1(self.id, sel("newHeapWithDescriptor:"), desc) };
            assert!(!id.is_null(), "newHeapWithDescriptor failed");
            Heap { id }
        })
    }

    /// Compile a Metal shader library from source string.
    pub fn new_library(&self, source: &str) -> Library {
        with_autorelease_pool(|| {
            let src = ns_string(source);
            let options = new_compile_options();
            let mut err: *const c_void = std::ptr::null_mut();
            let errpp = &mut err as *mut *const c_void;
            let id = unsafe {
                msgsend_3(
                    self.id,
                    sel("newLibraryWithSource:options:error:"),
                    src,
                    options,
                    errpp as *const c_void,
                )
            };
            if id.is_null() {
                if !err.is_null() {
                    panic!("Metal shader compile error: {}", ns_to_string(err));
                }
                panic!("newLibraryWithSource failed");
            }
            Library { id }
        })
    }

    /// Build a compute pipeline from a named kernel function.
    pub fn new_compute_pipeline(&self, function: &Function) -> ComputePipeline {
        with_autorelease_pool(|| {
            let id = unsafe {
                msgsend_2(
                    self.id,
                    sel("newComputePipelineStateWithFunction:error:"),
                    function.id,
                    std::ptr::null_mut::<c_void>(),
                )
            };
            assert!(!id.is_null(), "newComputePipelineStateWithFunction failed");
            ComputePipeline { id }
        })
    }
}

impl Library {
    /// Look up a function by name inside the compiled library.
    pub fn function_named(&self, name: &str) -> Function {
        with_autorelease_pool(|| {
            let n = ns_string(name);
            let id = unsafe { msgsend_1(self.id, sel("newFunctionWithName:"), n) };
            assert!(!id.is_null(), "newFunctionWithName failed: {name}");
            Function { id }
        })
    }
}

impl Heap {
    /// Allocate a buffer from the heap (same storage mode as the heap).
    pub fn new_buffer(&self, len: usize) -> MetalBuffer {
        with_autorelease_pool(|| {
            let id = unsafe {
                msgsend_u_u(
                    self.id,
                    sel("newBufferWithLength:options:"),
                    len as u64,
                    STORAGE_MODE_SHARED as u64,
                )
            };
            assert!(!id.is_null(), "heap newBufferWithLength failed");
            MetalBuffer { id }
        })
    }
}

impl CommandQueue {
    /// Grab a command buffer; call commit() + wait before reading results back.
    pub fn new_command_buffer(&self) -> CommandBuffer {
        let id = unsafe { objc_msgSend(self.id, sel("commandBuffer")) };
        assert!(!id.is_null(), "commandBuffer failed");
        CommandBuffer { id }
    }
}

impl CommandBuffer {
    /// Create a compute encoder bound to this command buffer.
    pub fn compute_compute_encoder(&self) -> ComputeEncoder {
        let id = unsafe { objc_msgSend(self.id, sel("computeCommandEncoder")) };
        assert!(!id.is_null(), "computeCommandEncoder failed");
        ComputeEncoder { id }
    }

    pub fn commit(&self) {
        unsafe { msgsend_void(self.id, sel("commit")) };
    }

    pub fn wait_until_completed(&self) {
        unsafe { msgsend_void(self.id, sel("waitUntilCompleted")) };
    }
}

impl ComputeEncoder {
    pub fn set_compute_pipeline_state(&self, pipe: &ComputePipeline) {
        unsafe { msgsend_pipeline(self.id, sel("setComputePipelineState:"), pipe.id) };
    }

    /// Bind a GPU buffer at an argument index.
    pub fn set_buffer(&self, buf: &MetalBuffer, offset: u64, index: u64) {
        unsafe {
            msgsend_p_u_u(
                self.id,
                sel("setBuffer:offset:atIndex:"),
                buf.id,
                offset,
                index,
            )
        };
    }

    /// Bind a small CPU value blob at an argument index.
    pub fn set_bytes(&self, bytes: &[u8], index: u64) {
        unsafe {
            msgsend_p_u_u(
                self.id,
                sel("setBytes:length:atIndex:"),
                bytes.as_ptr() as *const c_void,
                bytes.len() as u64,
                index,
            )
        };
    }

    /// Dispatch the grid of threads with a given threadgroup shape.
    pub fn dispatch_threads(&self, grid: MTLSize, threads_per_group: MTLSize) {
        unsafe {
            msgsend_dispatch(
                self.id,
                sel("dispatchThreads:threadsPerThreadgroup:"),
                grid,
                threads_per_group,
            )
        };
    }

    /// Dispatch by explicit threadgroup count (for tgid-indexed kernels).
    pub fn dispatch_thread_groups(&self, threadgroups: MTLSize, threads_per_group: MTLSize) {
        unsafe {
            msgsend_groups(
                self.id,
                sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                threadgroups,
                threads_per_group,
            )
        };
    }

    pub fn end_encoding(&self) {
        unsafe { msgsend_void(self.id, sel("endEncoding")) };
    }
}

impl MetalBuffer {
    pub fn is_null(&self) -> bool {
        self.id.is_null()
    }

    /// CPU-side pointer for shared-storage buffers (`[MTLBuffer contents]`).
    pub fn contents(&self) -> *mut u8 {
        let p = unsafe { objc_msgSend(self.id, sel("contents")) };
        assert!(!p.is_null(), "buffer contents null");
        p as *mut u8
    }

    /// Copy `len` bytes out of the buffer from `offset`.
    pub fn read_bytes(&self, offset: u64, len: usize) -> Vec<u8> {
        let p = self.contents();
        let start = unsafe { p.add(offset as usize) };
        let mut out = Vec::<u8>::new();
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(start, len) });
        out
    }

    /// Copy `len` bytes into the buffer from `offset`.
    pub fn write_bytes(&self, offset: u64, data: &[u8]) {
        let p = self.contents();
        let dst = unsafe { p.add(offset as usize) };
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
    }
}

// Release keeps the small shared buffer off Metal's sub-allocator heap when
// more objects are created later in the same process.
impl Drop for MetalBuffer {
    fn drop(&mut self) {
        if !self.id.is_null() {
            unsafe { objc_msgSend(self.id, sel("release")) };
        }
    }
}

fn sel(name: &str) -> *const c_void {
    let c = CString::new(name).expect("selector");
    unsafe { sel_registerName(c.as_ptr()) }
}

/// Build an NSString via CoreFoundation (toll-free bridged with NSString).
/// Pure C — no msgSend, no autorelease pool needed.
fn ns_string(text: &str) -> *const c_void {
    let c = CString::new(text).expect("c string");
    let obj = unsafe {
        CFStringCreateWithCString(std::ptr::null::<c_void>(), c.as_ptr(), C_ENCODING_UTF8)
    };
    assert!(!obj.is_null(), "NSString creation failed");
    obj
}

/// Instantiate `MTLCompileOptions` via `[MTLCompileOptions new]` and enable
/// fast math (needed for NAX / tensor-unit shaders).
fn new_compile_options() -> *const c_void {
    let cls = objc_get_c_class("MTLCompileOptions");
    let raw = unsafe { objc_msgSend(cls, sel("alloc")) };
    assert!(!raw.is_null(), "MTLCompileOptions alloc failed");
    let obj = unsafe { objc_msgSend(raw, sel("init")) };
    assert!(!obj.is_null(), "MTLCompileOptions init failed");
    unsafe { msgsend_p_u(obj, sel("setFastMathEnabled:"), 1u64) };
    obj
}

/// Read an NSObject/NSString to a Rust String via UTF8String.
fn ns_to_string(obj: *const c_void) -> String {
    let s = unsafe { objc_msgSend(obj, sel("localizedDescription")) };
    if s.is_null() {
        return String::from("(no description)");
    }
    let cstr = unsafe { objc_msgSend(s, sel("UTF8String")) as *const c_char };
    if cstr.is_null() {
        return String::from("(empty)");
    }
    unsafe { CStr::from_ptr(cstr) }
        .to_string_lossy()
        .into_owned()
}

fn objc_get_c_class(name: &str) -> *const c_void {
    let c = CString::new(name).expect("class name");
    let cls = unsafe { objc_getClass(c.as_ptr()) };
    assert!(!cls.is_null(), "class not found: {name}");
    cls
}

/// Run `f` while an NSAutoreleasePool is current on this thread.
/// Metal shader compile / pipeline creation autorelease temporary objects.
fn with_autorelease_pool<R>(f: impl FnOnce() -> R) -> R {
    let cls = objc_get_c_class("NSAutoreleasePool");
    let raw = unsafe { objc_msgSend(cls, sel("alloc")) };
    let pool = unsafe { objc_msgSend(raw, sel("init")) };
    assert!(!pool.is_null(), "autorelease pool init failed");
    let r = f();
    unsafe { objc_msgSend(pool, sel("drain")) };
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquires_the_gpu() {
        let dev = MetalDevice::default();
        assert!(!dev.id.is_null());
        let n = dev.name();
        println!("Metal device: {n}");
        assert!(!n.is_empty());
        assert_ne!(dev.max_working_set(), 0);
    }

    #[test]
    fn allocates_shared_buffer() {
        let dev = MetalDevice::default();
        // Real weight shards are multi-MB; small shared buffers can collide
        // with Metal's internal sub-allocator heap. Use a production-sized block.
        let buf = dev.new_buffer(64 << 20);
        assert!(!buf.is_null());
    }

    #[test]
    fn builds_ns_string() {
        let obj = ns_string("hello");
        assert!(!obj.is_null());
    }
}
