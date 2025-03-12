import torch
from datasets import load_dataset, get_dataset_config_names
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import multiprocessing
from pathlib import Path

def process_subset(subset, gpu_id, done_queue):


    # Set the device for this process
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Processing {subset} on {device}")

    # Initialize the model for this process
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
    model.to(device)

    def embed_batch(batch):
        return model.encode(batch['text'], show_progress_bar=False, prompt_name='passage')

    # Embed the dataset in batches and save in shards

    data_dir = Path(f'./wikivecs/{subset}')
    data_dir.mkdir(parents=True, exist_ok=True)

    shard_files = list(data_dir.glob(f'{subset}_embeddings_shard_*.npy'))
    if not shard_files:
        batch_size = 100000
        shard_size = 1000000  
        shard_count = 0
        current_shard = []

        ds = load_dataset("wikimedia/wikipedia", subset)
        for i in tqdm(range(0, len(ds['train']), batch_size), desc=f"Processing {subset}"):
            batch = ds['train'][i:i+batch_size]
            batch_embeddings = embed_batch(batch)
            current_shard.extend(batch_embeddings)
            
            if len(current_shard) >= shard_size:
                # Save the current shard
                shard_file = data_dir / f'{subset}_embeddings_shard_{shard_count}.npy'
                np.save(shard_file, np.array(current_shard))
                print(f"Saved shard {shard_count} to {shard_file}")
                
                # Reset for next shard
                current_shard = []
                shard_count += 1

        # Save any remaining embeddings in the last shard
        if current_shard:
            shard_file = data_dir / f'{subset}_embeddings_shard_{shard_count}.npy'
            np.save(shard_file, np.array(current_shard))
            print(f"Saved final shard {shard_count} to {shard_file}")

        print(f"Processed and saved embeddings for subset: {subset} in {shard_count + 1} shards")
        done_queue.put(gpu_id)  # Signal that this GPU is now free

def launch_embed():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    subsets = get_dataset_config_names("wikimedia/wikipedia")
    done_queue = multiprocessing.Queue()
    processes = []
    active_gpus = set()

    for subset in subsets:
        # Wait for a free GPU
        while len(active_gpus) >= num_gpus:
            free_gpu = done_queue.get()
            active_gpus.remove(free_gpu)

        # Find the first available GPU
        for gpu_id in range(num_gpus):
            if gpu_id not in active_gpus:
                break
        
        active_gpus.add(gpu_id)
        p = multiprocessing.Process(target=process_subset, args=(subset, gpu_id, done_queue))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for _ in processes:
        done_queue.get()

    # Ensure all processes have finished
    for p in processes:
        p.join()

    print("All subsets processed.")

if __name__ == "__main__":
    launch_embed()