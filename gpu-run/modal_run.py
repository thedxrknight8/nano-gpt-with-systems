import modal

app = modal.App("nano-gpt-v3")

image = (
    modal.Image.debian_slim()
    .pip_install("torch")
    .pip_install("numpy")
    .add_local_dir(".", remote_path="/root/nano-gpt")
)
@app.function(image=image, gpu="A10G", timeout=60 * 60)
def train():
    import subprocess
    subprocess.run(
        ["python", "-u", "scripts/v3.py"],
        cwd="/root/nano-gpt",
        check=True,
    )