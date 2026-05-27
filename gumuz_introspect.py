import time
import os
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock
from config import GOOGLE_API_KEY, OPENAI_API_KEY
import google.generativeai as genai
from openai import OpenAI

# Configure OpenAI API
openai_client = OpenAI(api_key=OPENAI_API_KEY)
model_name = "GPT-4o"
use_gemini = False
min_interval = 0  # No rate limiting for GPT
use_gemini = None

# Test API connection first
print(f"\nTesting {model_name} API connection...")
def test_api_connection():
    """Test if the API is accessible and has quota available."""
    try:
        if use_gemini:
            # Test with a simple prompt
            test_prompt = "Hello, please respond with 'API test successful'."
            response = gemini_model.generate_content(test_prompt)
            if response.text:
                print("✓ Gemini API test successful")
                return True
        else:
            # Test OpenAI API
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello, please respond with 'API test successful'."}],
                max_tokens=10
            )
            if response.choices[0].message.content:
                print("✓ GPT-4o API test successful")
                return True
    except Exception as e:
        error_msg = str(e)
        print(f"✗ API test failed: {error_msg}")
        
        # Check for specific errors
        if "limit: 0" in error_msg or "quota exceeded for metric" in error_msg.lower():
            print("\n⚠️  PERMANENT QUOTA EXHAUSTION DETECTED")
            print("Your Gemini API key has NO free tier quota (limit: 0).")
            print("You must enable billing or switch to GPT-4o (option 2).")
            return "PERMANENT_QUOTA_EXHAUSTION"
        elif "quota" in error_msg.lower():
            print("\n⚠️  CRITICAL: API QUOTA EXHAUSTED")
            print("Your Gemini API key has run out of quota.")
            print("Please check: https://aistudio.google.com/app/apikey")
        elif "429" in error_msg:
            print("\n⚠️  RATE LIMIT: Too many requests")
        elif "api key" in error_msg.lower():
            print("\n⚠️  API KEY ERROR: Check your GOOGLE_API_KEY in config.py")
        return False
    return False

# Run API test
api_test_result = test_api_connection()
if api_test_result == "PERMANENT_QUOTA_EXHAUSTION":
    print("\n❌ Cannot proceed with Gemini due to permanent quota exhaustion.")
    print("Please switch to GPT-4o (run script again, choose option 2).")
    # Exit gracefully
    import sys
    sys.exit(0)
elif not api_test_result:
    print(f"\n⚠️  WARNING: {model_name} API test failed!")
    print("You may encounter errors during processing.")
    
    if use_gemini:
        print("\nGemini-specific issues:")
        print("1. Check quota at: https://aistudio.google.com/app/apikey")
        print("2. You might need to enable billing")
        print("3. Try GPT-4o instead (option 2)")
    
    # Ask user if they want to continue
    proceed = input("\nDo you want to proceed anyway? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("Exiting...")
        import sys
        sys.exit(0)

# Rate limiting with exponential backoff
if use_gemini:
    last_request_time = 0
    rate_lock = Lock()
    consecutive_errors = 0
    
    def wait_for_rate_limit():
        """Ensure we wait between Gemini requests with exponential backoff on errors."""
        global last_request_time, consecutive_errors
        with rate_lock:
            now = time.time()
            time_since_last = now - last_request_time
            
            # Base delay
            base_delay = min_interval
            
            # Add exponential backoff if we've had recent errors
            extra_delay = 0
            if consecutive_errors > 0:
                extra_delay = (2 ** min(consecutive_errors, 5)) * 5
                print(f"  Adding {extra_delay}s backoff due to {consecutive_errors} consecutive errors")
            
            total_delay = base_delay + extra_delay
            
            if time_since_last < total_delay:
                wait_time = total_delay - time_since_last
                if wait_time > 0:
                    time.sleep(wait_time)
            
            last_request_time = time.time()
else:
    consecutive_errors = 0
    
    def wait_for_rate_limit():
        """No rate limiting needed for GPT."""
        pass

def call_model(prompt, max_retries=3):
    """Call the selected model with a prompt and return the response text."""
    global consecutive_errors
    
    for attempt in range(max_retries):
        try:
            if use_gemini:
                wait_for_rate_limit()
                response = gemini_model.generate_content(prompt)
                # Reset error counter on success
                consecutive_errors = 0
                return response.text.strip()
            else:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            
            # Update error counter for Gemini
            if use_gemini:
                consecutive_errors += 1
            
            # Check for permanent quota exhaustion
            if "limit: 0" in error_str or "quota exceeded for metric" in error_str.lower():
                print(f"  PERMANENT QUOTA EXHAUSTION DETECTED. Stopping all retries.")
                return f"ERROR: PERMANENT QUOTA EXHAUSTION. Free tier limit is 0. Please enable billing or switch to GPT-4o."
            
            # Check error type
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) * 15  # 15s, 30s, 60s...
                    print(f"  Rate limit/quota error, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"ERROR: Rate limit/quota exceeded after {max_retries} retries"
            elif "safety" in error_str.lower() or "blocked" in error_str.lower():
                return f"ERROR: Content blocked by safety filters"
            else:
                # For other errors
                return f"ERROR: {str(e)[:100]}"  # Limit error length
    
    return f"ERROR: Failed after {max_retries} retries"

# Define ONLY sentences and nouns datasets
common_sentences = [
    "ለድ ውድ ሁሳ!", "ዳሽህ ጃ!", "ዳግህት ድጓ!", 
    "ዳችሉቅ አያ!", "ዳክዱቁ አንቅህ  ሽሻ!", "ላ ጀ አንዝግት?", 
    "ጃ መቶ ብር ሊያ.", "እንግፍ አዳኛ ሩቅ.", "ዳዱ ሊቁማ", "ዳፑቅ ዊንዝ  ድዋ"
]

common_nouns = [
    "ድጋንፂጋ ጉንዛ", "ባቢያ", "እዮ", "ኡአማ", "ፓቱዋ", 
    "አያ", "ውዱ ሁሳ", "ድጓ", "ማእያ", "ድምዥጋ"
]

# Define prompt templates for ONLY sentence and noun types
PROMPT_TEMPLATES = {
    "sentence": {
        "elab_am": lambda s: f"\"{s}\" የሚለው ዓረፍተ ነገር የምን ቋንቋ እና አረፍተ ነገሩን ራሱ በመልሱ ውስጥ ሳታካትት በአንድ አጭር ዓረፍተ ነገር ብቻ መቸ ጥቅም ላይ እንደምናዉለዉ አብራራልኝ።",
        "ded_am": lambda e: f" \"{e}\" ይህንን መልስ  እኔ የሆነ ጥያቄ ጠይቄህ የሰጠኽኝ መልስ ነዉ። የተጠየቅሁህን ጥያቄ በአንድ አጭር አረፍተነገር ገምትልኝ",
        "elab_en": lambda s: f"Identify the language of the following sentence \"{s}\" and explain when we used it without including the sentence itself in the answer. Your answer should be one short sentence.",
        "ded_en": lambda e: f"The following is the answer you provided for a certain question \"{e}\". Based on this answer, please deduce and provide only one possible original question I might have asked.",
        "elab_am_from_en": lambda s: f"Identify the language of the following sentence \"{s}\" and explain when we used it without including the sentence itself in the answer. Your answer should be one short sentence using Amharic language only.",
        "ded_am_from_en": lambda e: f"The following is the answer you provided for a certain question \"{e}\". Based on this answer, please deduce and provide only one possible original question I might have asked. your response should be in Amharic"
    },
    "noun": {
        "elab_am": lambda s: f"\"{s}\" የሚለው ምን ማለት ነዉ? ቃሉን ራሱ በመልሱ ውስጥ ሳታካትት በአንድ አጭር ዓረፍተ ነገር ብቻ አብራራ።",
        "ded_am": lambda e: f"\"{e}\" ይህንን መልስ እኔ የሆነ ጥያቄ ጠይቄህ የሰጠኽኝ መልስ ነዉ። የተጠየቅሁህን ጥያቄ በአንድ አጭር አረፍተነገር ገምትልኝ",
        "elab_en": lambda s: f"Identify the language of the following word \"{s}\" and explain the meaning of the word using only one sentence without including the word in the answer.",
        "ded_en": lambda e: f" \"{e}\"This is the answer you provided for a certain question. Based on this answer, please deduce and provide only one possible original question I might have asked.",
        "elab_am_from_en": lambda s: f"Identify the language of the following word \"{s}\" and explain the meaning of the word without including the word itself in the answer. Your answer should be one and short using Amharic language only.",
        "ded_am_from_en": lambda e: f"This is the answer you provided for a certain question \"{e}\". Could you deduct the original question I asked using one question? Your answer should be one and short using Amharic language only."
    }
}

# Number of repetitions for each item
reps = 4

# Create output folder if it doesn't exist
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Setup CSV files for incremental saving
model_safe_name = model_name.replace(' ', '_').replace('-', '_').lower()
elaborations_csv = os.path.join(output_folder, f"elaborations_{model_safe_name}.csv")
predictions_csv = os.path.join(output_folder, f"predictions_{model_safe_name}.csv")

# Load existing elaborations if file exists
elaborations = {}
completed_elaborations = set()
retry_elaborations = set()  # Track errors that need retry
if os.path.exists(elaborations_csv):
    print(f"\nFound existing elaborations file: {elaborations_csv}")
    print("Loading previous results...")
    try:
        with open(elaborations_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows_loaded = 0
            for row in reader:
                item = row['Item']
                item_type = row['Item_Type']
                i = int(row['Rep_Number'])
                prompt_type = row['Prompt_Type']
                elaboration = row['Elaboration']
                
                # Create a unique key for each combination
                key = (item, item_type, i, prompt_type)
                
                # Check if it's an error that should be retried
                if "ERROR:" in elaboration:
                    if "Rate limit" in elaboration or "quota" in elaboration.lower() or "PERMANENT" in elaboration:
                        retry_elaborations.add(key)
                        print(f"  Found error for {item_type} '{item}' rep {i} type {prompt_type}, will retry")
                    else:
                        # Keep other errors as-is
                        if item not in elaborations:
                            elaborations[item] = {}
                        if item_type not in elaborations[item]:
                            elaborations[item][item_type] = {}
                        if i not in elaborations[item][item_type]:
                            elaborations[item][item_type][i] = {}
                        elaborations[item][item_type][i][prompt_type] = elaboration
                        completed_elaborations.add(key)
                else:
                    if item not in elaborations:
                        elaborations[item] = {}
                    if item_type not in elaborations[item]:
                        elaborations[item][item_type] = {}
                    if i not in elaborations[item][item_type]:
                        elaborations[item][item_type][i] = {}
                    elaborations[item][item_type][i][prompt_type] = elaboration
                    completed_elaborations.add(key)
                rows_loaded += 1
            print(f"Loaded {rows_loaded} previous elaborations ({len(retry_elaborations)} to retry)")
    except Exception as e:
        print(f"Could not load existing file: {e}")

# Prepare ONLY sentences and nouns datasets
datasets = [
    ("sentence", common_sentences),
    ("noun", common_nouns)
]

# Step 1: Generate elaborations for all items
print(f"\nStep 1: Generating elaborations with {model_name}...")
total_items = sum(len(items) for _, items in datasets)
print(f"Processing {total_items} items total:")
print(f"  - {len(common_sentences)} sentences")
print(f"  - {len(common_nouns)} nouns") 
print(f"Generating {len(PROMPT_TEMPLATES['sentence'])} prompt types for each item × {reps} reps each")
if use_gemini:
    print(f"Using {min_interval} second delay between requests")

# Use utf-8-sig encoding for proper Amharic display in CSV
elaborations_file = open(elaborations_csv, 'a', newline='', encoding='utf-8-sig')
elaborations_writer = None

# Prepare all elaboration prompts (skip completed ones, but include retries)
elaboration_tasks = []
for item_type, items in datasets:
    for item in items:
        for i in range(reps):
            for prompt_id, template_func in PROMPT_TEMPLATES[item_type].items():
                if prompt_id.startswith("elab_"):  # Only elaboration prompts in this step
                    key = (item, item_type, i, prompt_id)
                    if key not in completed_elaborations or key in retry_elaborations:
                        prompt = template_func(item)
                        elaboration_tasks.append((item, item_type, i, prompt_id, prompt))
                    else:
                        # Ensure structure exists for completed ones
                        if item not in elaborations:
                            elaborations[item] = {}
                        if item_type not in elaborations[item]:
                            elaborations[item][item_type] = {}
                        if i not in elaborations[item][item_type]:
                            elaborations[item][item_type][i] = {}
                        if prompt_id not in elaborations[item][item_type][i]:
                            elaborations[item][item_type][i][prompt_id] = ""

# Initialize CSV writer if we have new tasks
if elaboration_tasks:
    if elaborations_writer is None:
        # Check if file is empty or new
        file_exists = os.path.exists(elaborations_csv) and os.path.getsize(elaborations_csv) > 0
        elaborations_writer = csv.DictWriter(elaborations_file, 
                                            fieldnames=['Item', 'Item_Type', 'Rep_Number', 'Prompt_Type', 'Elaboration', 'Model'])
        if not file_exists:
            elaborations_writer.writeheader()
            elaborations_file.flush()

    # Process elaborations - sequentially for Gemini
    print(f"Processing {len(elaboration_tasks)} elaborations...")
    
    if use_gemini:
        print("Using sequential processing to avoid rate limits")
        
        with tqdm(total=len(elaboration_tasks), desc="Generating elaborations") as pbar:
            for item, item_type, i, prompt_id, prompt in elaboration_tasks:
                try:
                    print(f"\nProcessing {item_type}: '{item}' (rep {i+1}/{reps}, type: {prompt_id})")
                    elaboration = call_model(prompt)
                    
                    # Save immediately to CSV
                    elaborations_writer.writerow({
                        'Item': item,
                        'Item_Type': item_type,
                        'Rep_Number': i,
                        'Prompt_Type': prompt_id,
                        'Elaboration': elaboration,
                        'Model': model_name
                    })
                    elaborations_file.flush()
                    
                    # Store in memory
                    if item not in elaborations:
                        elaborations[item] = {}
                    if item_type not in elaborations[item]:
                        elaborations[item][item_type] = {}
                    if i not in elaborations[item][item_type]:
                        elaborations[item][item_type][i] = {}
                    elaborations[item][item_type][i][prompt_id] = elaboration
                    
                    # Show result
                    if "ERROR" in elaboration:
                        print(f"  Result: {elaboration}")
                    else:
                        print(f"  Result: Success")
                    
                except Exception as e:
                    error_msg = f"ERROR: {str(e)[:100]}"
                    elaborations_writer.writerow({
                        'Item': item,
                        'Item_Type': item_type,
                        'Rep_Number': i,
                        'Prompt_Type': prompt_id,
                        'Elaboration': error_msg,
                        'Model': model_name
                    })
                    elaborations_file.flush()
                    if item not in elaborations:
                        elaborations[item] = {}
                    if item_type not in elaborations[item]:
                        elaborations[item][item_type] = {}
                    if i not in elaborations[item][item_type]:
                        elaborations[item][item_type][i] = {}
                    elaborations[item][item_type][i][prompt_id] = error_msg
                    print(f"  Result: {error_msg}")
                
                pbar.update(1)
                
                # Add delay between requests
                if use_gemini and pbar.n < pbar.total:
                    print(f"  Waiting {min_interval} seconds before next request...")
                    time.sleep(min_interval)
    else:
        # For GPT, use concurrent processing with optimal workers
        optimal_workers = min(10, len(elaboration_tasks))
        print(f"Using {optimal_workers} concurrent workers for GPT-4o")
        
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            future_to_task = {executor.submit(call_model, prompt): (item, item_type, i, prompt_id) 
                              for item, item_type, i, prompt_id, prompt in elaboration_tasks}
            
            with tqdm(total=len(elaboration_tasks), desc="Generating elaborations") as pbar:
                for future in as_completed(future_to_task):
                    item, item_type, i, prompt_id = future_to_task[future]
                    try:
                        elaboration = future.result()
                        elaborations_writer.writerow({
                            'Item': item,
                            'Item_Type': item_type,
                            'Rep_Number': i,
                            'Prompt_Type': prompt_id,
                            'Elaboration': elaboration,
                            'Model': model_name
                        })
                        elaborations_file.flush()
                        
                        if item not in elaborations:
                            elaborations[item] = {}
                        if item_type not in elaborations[item]:
                            elaborations[item][item_type] = {}
                        if i not in elaborations[item][item_type]:
                            elaborations[item][item_type][i] = {}
                        elaborations[item][item_type][i][prompt_id] = elaboration
                    except Exception as e:
                        error_msg = f"ERROR: {str(e)[:100]}"
                        elaborations_writer.writerow({
                            'Item': item,
                            'Item_Type': item_type,
                            'Rep_Number': i,
                            'Prompt_Type': prompt_id,
                            'Elaboration': error_msg,
                            'Model': model_name
                        })
                        elaborations_file.flush()
                        if item not in elaborations:
                            elaborations[item] = {}
                        if item_type not in elaborations[item]:
                            elaborations[item][item_type] = {}
                        if i not in elaborations[item][item_type]:
                            elaborations[item][item_type][i] = {}
                        elaborations[item][item_type][i][prompt_id] = error_msg
                    pbar.update(1)

elaborations_file.close()

# Step 2: Generate predictions from elaborations
print("\nStep 2: Generating predictions from elaborations...")
predictions = {}

# Load existing predictions if file exists
completed_predictions = set()
retry_predictions = set()
if os.path.exists(predictions_csv):
    print(f"Found existing predictions file: {predictions_csv}")
    print("Loading previous predictions...")
    try:
        with open(predictions_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows_loaded = 0
            for row in reader:
                item = row['Item']
                item_type = row['Item_Type']
                i = int(row['Rep_Number'])
                prompt_type = row['Prompt_Type']
                predicted_question = row['Predicted_Question']
                
                key = (item, item_type, i, prompt_type)
                
                if "ERROR:" in predicted_question:
                    if "Rate limit" in predicted_question or "quota" in predicted_question.lower() or "PERMANENT" in predicted_question:
                        retry_predictions.add(key)
                    else:
                        if item not in predictions:
                            predictions[item] = {}
                        if item_type not in predictions[item]:
                            predictions[item][item_type] = {}
                        if i not in predictions[item][item_type]:
                            predictions[item][item_type][i] = {}
                        predictions[item][item_type][i][prompt_type] = predicted_question
                        completed_predictions.add(key)
                else:
                    if item not in predictions:
                        predictions[item] = {}
                    if item_type not in predictions[item]:
                        predictions[item][item_type] = {}
                    if i not in predictions[item][item_type]:
                        predictions[item][item_type][i] = {}
                    predictions[item][item_type][i][prompt_type] = predicted_question
                    completed_predictions.add(key)
                rows_loaded += 1
            print(f"Loaded {rows_loaded} previous predictions")
    except Exception as e:
        print(f"Could not load existing file: {e}")

# Prepare all question deduction prompts
question_tasks = []
for item_type, items in datasets:
    for item in items:
        for i in range(reps):
            for prompt_id, template_func in PROMPT_TEMPLATES[item_type].items():
                if prompt_id.startswith("ded_"):  # Only deduction prompts in this step
                    key = (item, item_type, i, prompt_id)
                    # Find corresponding elaboration
                    elaboration = ""
                    elab_prompt_id = prompt_id.replace("ded_", "elab_")
                    
                    # Check if we have this elaboration in memory
                    if (item in elaborations and 
                        item_type in elaborations[item] and 
                        i in elaborations[item][item_type] and 
                        elab_prompt_id in elaborations[item][item_type][i]):
                        elaboration = elaborations[item][item_type][i][elab_prompt_id]
                    
                    if elaboration and "ERROR" not in elaboration and (key not in completed_predictions or key in retry_predictions):
                        prompt = template_func(elaboration)
                        question_tasks.append((item, item_type, i, prompt_id, prompt, elaboration))

# Process question deductions
if question_tasks:
    print(f"Processing {len(question_tasks)} predictions...")
    
    predictions_file = open(predictions_csv, 'a', newline='', encoding='utf-8-sig')
    file_exists = os.path.exists(predictions_csv) and os.path.getsize(predictions_csv) > 0
    predictions_writer = csv.DictWriter(predictions_file,
                                       fieldnames=['Item', 'Item_Type', 'Rep_Number', 'Prompt_Type', 'Predicted_Question', 'Model'])
    if not file_exists:
        predictions_writer.writeheader()
        predictions_file.flush()
    
    if use_gemini:
        print("Using sequential processing to avoid rate limits")
        
        with tqdm(total=len(question_tasks), desc="Generating predictions") as pbar:
            for item, item_type, i, prompt_id, prompt, elaboration in question_tasks:
                try:
                    print(f"\nProcessing prediction for {item_type}: '{item}' (rep {i+1}, type: {prompt_id})")
                    predicted_question = call_model(prompt)
                    
                    predictions_writer.writerow({
                        'Item': item,
                        'Item_Type': item_type,
                        'Rep_Number': i,
                        'Prompt_Type': prompt_id,
                        'Predicted_Question': predicted_question,
                        'Model': model_name
                    })
                    predictions_file.flush()
                    
                    if item not in predictions:
                        predictions[item] = {}
                    if item_type not in predictions[item]:
                        predictions[item][item_type] = {}
                    if i not in predictions[item][item_type]:
                        predictions[item][item_type][i] = {}
                    predictions[item][item_type][i][prompt_id] = predicted_question
                    
                    if "ERROR" in predicted_question:
                        print(f"  Result: {predicted_question}")
                    else:
                        print(f"  Result: Success")
                    
                except Exception as e:
                    error_msg = f"ERROR: {str(e)[:100]}"
                    predictions_writer.writerow({
                        'Item': item,
                        'Item_Type': item_type,
                        'Rep_Number': i,
                        'Prompt_Type': prompt_id,
                        'Predicted_Question': error_msg,
                        'Model': model_name
                    })
                    predictions_file.flush()
                    if item not in predictions:
                        predictions[item] = {}
                    if item_type not in predictions[item]:
                        predictions[item][item_type] = {}
                    if i not in predictions[item][item_type]:
                        predictions[item][item_type][i] = {}
                    predictions[item][item_type][i][prompt_id] = error_msg
                    print(f"  Result: {error_msg}")
                
                pbar.update(1)
                
                if use_gemini and pbar.n < pbar.total:
                    print(f"  Waiting {min_interval} seconds before next request...")
                    time.sleep(min_interval)
    else:
        # For GPT, use concurrent processing
        optimal_workers = min(10, len(question_tasks))
        print(f"Using {optimal_workers} concurrent workers for GPT-4o")
        
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            future_to_task = {executor.submit(call_model, prompt): (item, item_type, i, prompt_id, elaboration) 
                              for item, item_type, i, prompt_id, prompt, elaboration in question_tasks}
            
            with tqdm(total=len(question_tasks), desc="Generating predictions") as pbar:
                for future in as_completed(future_to_task):
                    item, item_type, i, prompt_id, elaboration = future_to_task[future]
                    try:
                        predicted_question = future.result()
                        predictions_writer.writerow({
                            'Item': item,
                            'Item_Type': item_type,
                            'Rep_Number': i,
                            'Prompt_Type': prompt_id,
                            'Predicted_Question': predicted_question,
                            'Model': model_name
                        })
                        predictions_file.flush()
                        
                        if item not in predictions:
                            predictions[item] = {}
                        if item_type not in predictions[item]:
                            predictions[item][item_type] = {}
                        if i not in predictions[item][item_type]:
                            predictions[item][item_type][i] = {}
                        predictions[item][item_type][i][prompt_id] = predicted_question
                    except Exception as e:
                        error_msg = f"ERROR: {str(e)[:100]}"
                        predictions_writer.writerow({
                            'Item': item,
                            'Item_Type': item_type,
                            'Rep_Number': i,
                            'Prompt_Type': prompt_id,
                            'Predicted_Question': error_msg,
                            'Model': model_name
                        })
                        predictions_file.flush()
                        if item not in predictions:
                            predictions[item] = {}
                        if item_type not in predictions[item]:
                            predictions[item][item_type] = {}
                        if i not in predictions[item][item_type]:
                            predictions[item][item_type][i] = {}
                        predictions[item][item_type][i][prompt_id] = error_msg
                    pbar.update(1)
    
    predictions_file.close()
else:
    print("No valid elaborations to generate predictions from.")

# Create comprehensive Excel files
print("\nStep 3: Creating comprehensive output files...")

# Create separate Excel files for each dataset type
for item_type, items in datasets:
    excel_data = []
    
    for item in items:
        for i in range(reps):
            row = {'Item': item, 'Item_Type': item_type, 'Rep_Number': i}
            
            # Add elaborations
            for prompt_id in PROMPT_TEMPLATES[item_type]:
                if prompt_id.startswith("elab_"):
                    elab = ""
                    if (item in elaborations and 
                        item_type in elaborations[item] and 
                        i in elaborations[item][item_type] and 
                        prompt_id in elaborations[item][item_type][i]):
                        elab = elaborations[item][item_type][i][prompt_id]
                    row[f'Elaboration_{prompt_id}'] = elab
            
            # Add predictions
            for prompt_id in PROMPT_TEMPLATES[item_type]:
                if prompt_id.startswith("ded_"):
                    pred = ""
                    if (item in predictions and 
                        item_type in predictions[item] and 
                        i in predictions[item][item_type] and 
                        prompt_id in predictions[item][item_type][i]):
                        pred = predictions[item][item_type][i][prompt_id]
                    row[f'Prediction_{prompt_id}'] = pred
            
            excel_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(excel_data)
    
    # Reorder columns for better readability
    column_order = ['Item', 'Item_Type', 'Rep_Number']
    for prompt_id in ['elab_am', 'ded_am', 'elab_en', 'ded_en', 'elab_am_from_en', 'ded_am_from_en']:
        if prompt_id in PROMPT_TEMPLATES[item_type]:
            if prompt_id.startswith('elab_'):
                column_order.append(f'Elaboration_{prompt_id}')
            else:
                column_order.append(f'Prediction_{prompt_id}')
    
    df = df[column_order]
    
    # Save to Excel with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{item_type}s_comprehensive_{model_safe_name}_{timestamp}.xlsx"
    output_path = os.path.join(output_folder, output_filename)
    
    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"✓ Created {item_type}s Excel file: {output_filename}")

# Also create a combined master file
print("\nCreating combined master file...")
master_data = []
for item_type, items in datasets:
    for item in items:
        for i in range(reps):
            row = {'Item': item, 'Item_Type': item_type, 'Rep_Number': i}
            
            # Add elaborations
            for prompt_id in PROMPT_TEMPLATES[item_type]:
                if prompt_id.startswith("elab_"):
                    elab = ""
                    if (item in elaborations and 
                        item_type in elaborations[item] and 
                        i in elaborations[item][item_type] and 
                        prompt_id in elaborations[item][item_type][i]):
                        elab = elaborations[item][item_type][i][prompt_id]
                    row[f'Elaboration_{item_type}_{prompt_id}'] = elab
            
            # Add predictions
            for prompt_id in PROMPT_TEMPLATES[item_type]:
                if prompt_id.startswith("ded_"):
                    pred = ""
                    if (item in predictions and 
                        item_type in predictions[item] and 
                        i in predictions[item][item_type] and 
                        prompt_id in predictions[item][item_type][i]):
                        pred = predictions[item][item_type][i][prompt_id]
                    row[f'Prediction_{item_type}_{prompt_id}'] = pred
            
            master_data.append(row)

master_df = pd.DataFrame(master_data)
master_filename = f"all_datasets_master_{model_safe_name}_{timestamp}.xlsx"
master_path = os.path.join(output_folder, master_filename)
master_df.to_excel(master_path, index=False, engine="openpyxl")
print(f"✓ Created master Excel file: {master_filename}")

print(f"\n{'='*60}")
print("PROCESSING COMPLETE")
print('='*60)
print(f"Elaborations CSV: {elaborations_csv}")
print(f"Predictions CSV: {predictions_csv}")
print(f"Model used: {model_name}")

# Summary statistics
total_items = sum(len(items) for _, items in datasets)
total_prompts_possible = total_items * reps * 12  # 6 elaboration + 6 deduction per item

successful_elaborations = 0
for item in elaborations:
    for item_type in elaborations[item]:
        for i in elaborations[item][item_type]:
            for prompt_id in elaborations[item][item_type][i]:
                if elaborations[item][item_type][i][prompt_id] and "ERROR" not in elaborations[item][item_type][i][prompt_id]:
                    successful_elaborations += 1

successful_predictions = 0
for item in predictions:
    for item_type in predictions[item]:
        for i in predictions[item][item_type]:
            for prompt_id in predictions[item][item_type][i]:
                if predictions[item][item_type][i][prompt_id] and "ERROR" not in predictions[item][item_type][i][prompt_id]:
                    successful_predictions += 1

total_elaboration_tasks = len(elaboration_tasks)
total_question_tasks = len(question_tasks)

print(f"\n📊 Summary Statistics:")
print(f"Total datasets: 2 (sentences, nouns)")
print(f"Total items processed: {total_items}")
print(f"Total elaborations attempted: {total_elaboration_tasks}")
success_rate_elab = (successful_elaborations/total_elaboration_tasks*100) if total_elaboration_tasks > 0 else 0
print(f"Successful elaborations: {successful_elaborations}/{total_elaboration_tasks} ({success_rate_elab:.1f}%)")
print(f"Total predictions attempted: {total_question_tasks}")
success_rate_pred = (successful_predictions/total_question_tasks*100) if total_question_tasks > 0 else 0
print(f"Successful predictions: {successful_predictions}/{total_question_tasks} ({success_rate_pred:.1f}%)")
print(f"Total API calls: {total_elaboration_tasks + total_question_tasks}")

# Dataset-specific statistics
print(f"\n📈 Dataset Breakdown:")
for item_type, items in datasets:
    type_success_elab = 0
    type_total_elab = 0
    for item in items:
        for i in range(reps):
            for prompt_id in PROMPT_TEMPLATES[item_type]:
                if prompt_id.startswith("elab_"):
                    type_total_elab += 1
                    if (item in elaborations and 
                        item_type in elaborations[item] and 
                        i in elaborations[item][item_type] and 
                        prompt_id in elaborations[item][item_type][i] and
                        elaborations[item][item_type][i][prompt_id] and 
                        "ERROR" not in elaborations[item][item_type][i][prompt_id]):
                        type_success_elab += 1
    
    type_success_pred = 0
    type_total_pred = 0
    for item in items:
        for i in range(reps):
            for prompt_id in PROMPT_TEMPLATES[item_type]:
                if prompt_id.startswith("ded_"):
                    type_total_pred += 1
                    if (item in predictions and 
                        item_type in predictions[item] and 
                        i in predictions[item][item_type] and 
                        prompt_id in predictions[item][item_type][i] and
                        predictions[item][item_type][i][prompt_id] and 
                        "ERROR" not in predictions[item][item_type][i][prompt_id]):
                        type_success_pred += 1
    
    if type_total_elab > 0:
        success_pct_elab = (type_success_elab/type_total_elab*100) if type_total_elab > 0 else 0
        success_pct_pred = (type_success_pred/type_total_pred*100) if type_total_pred > 0 else 0
        print(f"  {item_type}s: {len(items)} items")
        print(f"    Elaborations: {type_success_elab}/{type_total_elab} ({success_pct_elab:.1f}%)")
        print(f"    Predictions: {type_success_pred}/{type_total_pred} ({success_pct_pred:.1f}%)")

if use_gemini and successful_elaborations == 0 and elaboration_tasks:
    print(f"\n⚠️  WARNING: All Gemini requests failed!")
    print("Your Gemini API key likely has no quota.")
    print("Solutions:")
    print("1. Check quota at: https://aistudio.google.com/app/apikey")
    print("2. Enable billing if needed")
    print("3. Switch to GPT-4o (run again, choose option 2)")

print(f"\n✅ All output files created successfully!")
print(f"   Check the 'output' folder for comprehensive results.")
