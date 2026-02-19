import os
import torch
import librosa
import numpy as np  # <-- NEW: Import numpy
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from loader import load, setup_logging, load_config, get_save_path
import random 


def process_batch(batch_files, loaded_audios, prompt_text, eval_type, config, processor, model, logger):
    """Handles the text formatting, tokenization, forward pass, and saving for a specific prompt."""
    texts = []
    
    # 1. Prepare Text Inputs
    for file_name in batch_files:
        conversation = [
            {'role': 'system', 'content': ''}, 
            {"role": "user", "content": [
                {"type": "audio", "audio_url": file_name}, 
                {"type": "text", "text": prompt_text}, 
            ]}
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        texts.append(text)

    # 2. Process Batch
    inputs = processor(
        text=texts, 
        audio=loaded_audios, 
        return_tensors="pt", 
        padding=True, 
        sampling_rate=processor.feature_extractor.sampling_rate
    ).to(model.device)
    
    # 3. Forward Pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    # Move to CPU for saving
    penultimate_embeddings = outputs.hidden_states[-2].cpu()
    
    # 4. Save Embeddings
    for batch_idx, file_path in enumerate(batch_files):
        # Pass file_path and audio_folder exactly as get_save_path expects
        save_path = get_save_path(
            base_folder=config['output_base_dir'],
            file_path=file_path, 
            audio_folder=config['audio_folder'], 
            folder_mode=config['folder_mode'],
            eval_type=eval_type,
            run_type=config['run_type']
        )
        
        single_embedding_np = penultimate_embeddings[batch_idx].numpy() 
        np.save(save_path, single_embedding_np)
        
    # Cleanup to prevent OOM
    del outputs, inputs, penultimate_embeddings, single_embedding_np
    torch.cuda.empty_cache()

def main():
    config = load_config()
    logger = setup_logging(config['run_type'])
    
    logger.info("--- Step 1: Loading Processor ---")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

    logger.info("--- Step 2: Loading Model ---")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", 
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    logger.info(f"--- Step 3: Model Loaded on device {model.device} ---")

    # Load file list
    all_audios = load(config['audio_folder'])
    logger.info(f"Found {len(all_audios)} total files in {config['audio_folder']}.")
    
    MAX_SAMPLES = config['max_samples']
    if len(all_audios) > MAX_SAMPLES:
        logger.info(f"Sampling {MAX_SAMPLES} files from the dataset...")
        random.seed(42) # Ensure reproducible sampling across runs
        all_audios = random.sample(all_audios, MAX_SAMPLES)
        logger.info(f"Now processing {len(all_audios)} files.")

    batch_size = config['batch_size']

    # 4. BATCHED EXTRACTION LOOP
    for i in range(0, len(all_audios), batch_size):
        batch_files = all_audios[i : i + batch_size]
        loaded_audios = []
        valid_files = []
        
        # Load audio data once per batch
        for file_path in batch_files:
            try:
                audio_array, _ = librosa.load(file_path, sr=processor.feature_extractor.sampling_rate)
                loaded_audios.append(audio_array)
                valid_files.append(file_path)
            except Exception as e:
                logger.error(f"Failed to load audio {file_path}: {e}")
                
        if not valid_files:
            continue
            
        try:
            # Run Inference 1: Without Task Prompt (01)
            for prompt_item in config['prompts']:
                process_batch(
                    batch_files=valid_files,
                    loaded_audios=loaded_audios,
                    prompt_text=prompt_item['text'],
                    eval_type=prompt_item['eval_type'],
                    config=config,
                    processor=processor,
                    model=model,
                    logger=logger
                )
            
            logger.info(f"[{min(i + batch_size, len(all_audios))}/{len(all_audios)}] Processed batch for all prompts")
            
        except Exception as e:
            logger.error(f"Failed during inference for batch starting with {valid_files[0]}: {e}")
            continue

if __name__ == "__main__":
    main()