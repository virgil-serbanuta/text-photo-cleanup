BINARY=photo-cleanup/target/release/photo-cleanup

binary: $(BINARY)
.PHONY: binary

$(BINARY): photo-cleanup/src/*.rs photo-cleanup/Cargo.*
	cd photo-cleanup && cargo build --release
