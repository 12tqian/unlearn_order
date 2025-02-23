def main():

    import os
    from dotenv import load_dotenv
    from pathlib import Path
    from research_tools.gpu import get_gpus_available

    load_dotenv()

    hf_access_token = os.getenv("HUGGINGFACE_API_KEY")

    n_gpus = 1

    gpus_available = get_gpus_available()
    n_gpus = min(n_gpus, len(gpus_available))
    gpus = gpus_available[:n_gpus]

    assert n_gpus > 0, "No GPUs available"

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpus])


if __name__ == "__main__":
    main()
