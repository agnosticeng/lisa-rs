// Runs in its own process (integration test) so no shared Metal buffer exists
// on the same device sub-allocator heap as the command-queue internals.
use lisa_rs::device::metal::MetalDevice;

#[test]
fn creates_command_queue_and_command_buffer() {
    let dev = MetalDevice::default();
    let q = dev.new_command_queue();
    assert!(!q.id.is_null());
    let cb = q.new_command_buffer();
    assert!(!cb.id.is_null());
}
