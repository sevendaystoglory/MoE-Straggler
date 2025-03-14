import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline, set_seed
# Import plotting functionality
from plot import generate_expert_plots

# モデルのロード
model_name = "DataPilot/sarashina2.2-3Bx4-moe"

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Create a diverse dataset of prompts
prompts_dataset = [
    "Write a short story about a space explorer discovering a new planet.",
    "Explain quantum computing to a 5-year-old child.",
    "Compose a poem about autumn leaves.",
    "Describe the perfect recipe for chocolate chip cookies.",
    "Write a dialogue between two AIs discussing consciousness.",
    "Create a travel guide for a fictional fantasy city.",
    "Explain the importance of sustainable energy in the modern world.",
    "Write a letter from the perspective of a time traveler from the year 3000.",
    "Describe how to train a neural network in simple terms.",
    "Create a short mystery story with an unexpected twist.",
    "Write about the ethical implications of AI in healthcare.",
    "Compose a folk tale explaining why the sky is blue.",
    "Create a tutorial on how to learn a new language efficiently.",
    "Write a futuristic news article from the year 2150.",
    "Describe the evolution of human communication throughout history.",
    "Create a story about an unlikely friendship between a robot and a bird.",
    "Explain how blockchain technology works to a beginner.",
    "Write a motivational speech for graduating students.",
    "Create a detailed description of an alien ecosystem.",
    "Write a scene where two strangers meet during a power outage.",
    "Explain the process of photosynthesis in an engaging way.",
    "Create a fictional interview with a historical figure.",
    "Write about the day in the life of a deep sea creature.",
    "Describe a world where humans can communicate telepathically.",
    "Create a beginner's guide to stargazing.",
    "Write a story from the perspective of a house plant.",
    "Explain the concept of infinity in an intuitive way.",
    "Create a dialogue between the sun and the moon.",
    "Write about how technology might change education in the next 50 years.",
    "Describe a post-apocalyptic world where nature has reclaimed cities."
]

all_expert_assignments = {}

print(f"Running inference on {len(prompts_dataset)} different prompts...")
for run_idx, prompt in enumerate(prompts_dataset):
    set_seed(run_idx)  # Use different seed each time
    
    user_input = [{"role": "user", "content": prompt}]
    
    print(f"\nRun {run_idx+1}/{len(prompts_dataset)}:")
    print(f"Prompt: {prompt}")
    
    responses = chat_pipeline(
        user_input,
        max_length=200,  # Increased to generate longer responses (around 200 tokens)
        do_sample=True,
        temperature=0.5,  # Set temperature to 1
        num_return_sequences=1,
    )
    
    # Print the response
    print(f"Response: {responses[0]['generated_text']}")
    
    # Run inference with the input and collect expert assignments
    generate_expert_plots(model, tokenizer, prompt, run_idx, all_expert_assignments)

# After all runs, generate the cumulative plots
print("\nGenerating cumulative expert assignment plots...")
from plot import generate_cumulative_plots
generate_cumulative_plots(model, all_expert_assignments)