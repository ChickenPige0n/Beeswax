fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/beeswax.rc");
    println!("cargo:rerun-if-changed=assets/beeswax.manifest.xml");
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-changed=assets/icon.png");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        embed_resource::compile(
            "assets/beeswax.rc",
            embed_resource::ParamsIncludeDirs(["assets"]),
        )
        .manifest_required()
        .unwrap();
    }
}
