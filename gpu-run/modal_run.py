import modal

app = modal.App("nano-gpt-v3")

volume = modal.Volume.from_name("nano-gpt-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("torch")
    .pip_install("numpy")
    .add_local_dir(".", remote_path="/root/nano-gpt")
)
@app.function(image=image, gpu="A10G", timeout=60 * 60, volumes={"/outputs": volume})
def train():
    import subprocess
    subprocess.run(
        ["python", "-u", "scripts/rope_updates/v6.py"],
        cwd="/root/nano-gpt",
        check=True,
    )
    
    volume.commit()