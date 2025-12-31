import modal
import huggingface_hub

app = modal.App("example-get-started")
image = modal.Image.debian_slim().uv_pip_install("huggingface-hub")


@app.function(image=image)
def square(x):
    print("This code is running on a remote worker!")
    print("Testing Hugging Face Hub access from within a Modal function")
    print(huggingface_hub.auth_list())
    return x**2


@app.local_entrypoint()
def main():
    print("Testing Hugging Face Hub access locally")
    print(huggingface_hub.auth_list())
    print("the square is", square.remote(42))
