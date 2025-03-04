CLEANUP_BINARY=photo-cleanup/target/release/photo-cleanup
CUT_BINARY=photo-cut/target/release/photo-cut

binary: $(CLEANUP_BINARY) $(CUT_BINARY)
.PHONY: binary

$(CLEANUP_BINARY): photo-cleanup/src/*.rs photo-cleanup/Cargo.*
	cd photo-cleanup && cargo build --release

$(CUT_BINARY): photo-cut/src/*.rs photo-cut/Cargo.*
	cd photo-cut && cargo build --release
