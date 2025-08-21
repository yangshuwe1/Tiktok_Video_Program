import logging
from logger import get_logger
logger = get_logger("shared_models")

from ultralytics import YOLOWorld, YOLO
import torch
import os
import sys
import tempfile
import yaml
import dora
import openai
import mimetypes
import base64
from datetime import datetime

# Import transformers with error handling
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig
    logger.info("Transformers imported successfully")
except ImportError as e:
    logger.warning(f"Error importing transformers: {e}")
    BlipProcessor = None
    BlipForConditionalGeneration = None
    BitsAndBytesConfig = None

# Import CLIP
try:
    import clip as openai_clip
    logger.info("CLIP imported successfully")
except ImportError as e:
    logger.warning(f"Error importing CLIP: {e}")
    openai_clip = None

# Import modelscope
try:
    from modelscope import AutoTokenizer, AutoModelForCausalLM
    from modelscope import snapshot_download
    logger.info("ModelScope imported successfully")
except ImportError as e:
    logger.warning(f"Error importing ModelScope: {e}")
    AutoTokenizer = None
    AutoModelForCausalLM = None
    snapshot_download = None

def load_config():
    """Load configuration from file"""
    config_path = "config/config.yml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

def ensure_openai_key():
    """Ensure OpenAI API key is available and set"""
    config = load_config()
    openai_key = config.get('models', {}).get('openai_api_key')
    if not openai_key or openai_key == "YOUR_OPENAI_API_KEY":
        logger.error("OpenAI API key not set. Please edit config/config.yml and set models.openai_api_key.")
        raise ValueError("OpenAI API key not set. Please edit config/config.yml and set models.openai_api_key.")
    openai.api_key = openai_key
    return openai_key

# Initialize OpenAI client
try:
    ensure_openai_key()
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")

def calc_gpt4o_price(prompt_tokens, completion_tokens, config=None):
    """Calculate GPT4O API cost based on token usage"""
    if config is None:
        config = load_config()
    
    gpt4o_config = config.get('models', {}).get('gpt4o', {})
    input_price_per_1k = gpt4o_config.get('input_price_per_1k', 0.0006)
    output_price_per_1k = gpt4o_config.get('output_price_per_1k', 0.0024)
    
    input_price = (prompt_tokens / 1000) * input_price_per_1k
    output_price = (completion_tokens / 1000) * output_price_per_1k
    total_price = input_price + output_price
    
    return total_price

def image_to_data_url(image_path):
    """Convert image file to data URL for GPT4O API"""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode()
    return f"data:{mime_type};base64,{img_b64}"

# Global models for GPU acceleration
# YOLO model - Using YOLO-World for better product detection
def load_yolo_model():
    """Load YOLO model with fallback options"""
    try:
        # Try to load YOLO-World first
        if os.path.exists("models/yolov8s-world.pt"):
            logger.info("Loading YOLO-World model: models/yolov8s-world.pt")
            return YOLOWorld("models/yolov8s-world.pt")
        elif os.path.exists("yolo-world-s.pt"):
            logger.info("Loading YOLO-World model: yolo-world-s.pt")
            return YOLOWorld("yolo-world-s.pt")
        else:
            # Fallback to YOLOv8x if YOLO-World not available
            logger.info("Loading YOLOv8x model as fallback")
            return YOLO("models/yolov8x.pt")
    except Exception as e:
        logger.error(f"Error loading YOLO-World model: {e}")
        # Final fallback to YOLOv8x
        logger.warning("Falling back to YOLOv8x model")
        return YOLO("models/yolov8x.pt")

yolo_model = load_yolo_model()

# BLIP processor and model
if BlipProcessor is not None and BlipForConditionalGeneration is not None:
    try:
        logger.info("Loading BLIP model...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = blip_model.to("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("BLIP model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading BLIP model: {e}")
        logger.warning("Setting BLIP model to None - some features may be limited")
        blip_processor = None
        blip_model = None
else:
    logger.warning("BLIP modules not available, setting to None")
    blip_processor = None
    blip_model = None

# CLIP model and preprocess
if openai_clip is not None:
    try:
        logger.info("Loading CLIP model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = openai_clip.load("ViT-B/32", device=device)
        # Ensure CLIP model has device attribute
        if not hasattr(clip_model, 'device'):
            clip_model.device = device
        logger.info("CLIP model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading CLIP model: {e}")
        logger.warning("Setting CLIP model to None - some features may be limited")
        clip_model = None
        clip_preprocess = None
else:
    logger.warning("CLIP module not available, setting to None")
    clip_model = None
    clip_preprocess = None

# GPT4O model configuration (replaces Qwen)
gpt4o_config = None
try:
    config = load_config()
    gpt4o_config = config.get('models', {}).get('gpt4o', {})
    logger.info("GPT4O configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load GPT4O configuration: {e}")

# TikTok product classes - Enhanced for YOLO-World product detection
TIKTOK_PRODUCT_CLASSES = {
    # Fashion & Beauty
    'clothing', 'shoes', 'bag', 'jewelry', 'watch', 'accessories', 'hat', 'scarf', 'glove', 'sunglasses',
    'ring', 'necklace', 'bracelet', 'earring', 'belt', 'wallet', 'perfume', 'makeup', 'cosmetics',
    'lipstick', 'foundation', 'blush', 'eyeshadow', 'mascara', 'eyeliner', 'nail', 'skincare',
    'shampoo', 'conditioner', 'body wash', 'toothbrush', 'toothpaste', 'soap', 'razor', 'deodorant',
    'lotion', 'cream', 'serum', 'mask', 'supplement', 'vitamin',
    
    # Electronics
    'phone', 'laptop', 'tablet', 'camera', 'tv', 'headphone', 'speaker', 'mouse', 'keyboard', 'monitor',
    'printer', 'router', 'charger', 'cable', 'adapter', 'power bank', 'memory card', 'usb', 'hard drive',
    'ssd', 'camera lens', 'tripod', 'microphone', 'drone', 'projector', 'screen', 'remote', 'controller',
    'joystick', 'console', 'glasses', 'case',
    
    # Home & Kitchen
    'appliance', 'kitchen', 'furniture', 'home', 'cleaning', 'lighting', 'fan', 'air conditioner',
    'refrigerator', 'microwave', 'oven', 'vacuum', 'hair dryer', 'towel', 'blanket', 'pillow',
    'mattress', 'curtain', 'rug', 'decoration', 'plant', 'flower', 'umbrella', 'mop', 'broom',
    'bucket', 'sponge', 'brush', 'detergent', 'disinfectant', 'sanitizer',
    
    # Food & Beverages
    'food', 'snack', 'drink', 'medicine', 'bandage', 'thermometer',
    
    # Toys & Entertainment
    'toys', 'game', 'board game', 'puzzle', 'toy car', 'doll', 'lego', 'block', 'action figure',
    'plush', 'book', 'stationery',
    
    # Sports & Outdoor
    'sports', 'outdoor', 'car', 'bicycle', 'tool',
    
    # Pet Products
    'pet', 'pet food', 'pet toy', 'pet bed', 'pet clothes', 'pet leash', 'pet bowl', 'pet litter',
    'pet shampoo', 'pet brush', 'pet carrier', 'pet cage', 'pet house', 'pet scratching',
    'pet training', 'pet medicine', 'pet supplement', 'pet collar', 'pet harness', 'pet tag',
    'pet feeder', 'pet waterer', 'pet mat', 'pet blanket', 'pet towel', 'pet toothbrush',
    'pet toothpaste', 'pet deodorant', 'pet cleaner', 'pet comb', 'pet nail', 'pet scissors',
    'pet clipper', 'pet dryer', 'pet perfume', 'pet treat', 'pet snack', 'pet chew', 'pet bone',
    'pet stick', 'pet rope', 'pet ball', 'pet frisbee', 'pet tunnel', 'pet tent', 'pet backpack',
    'pet stroller', 'pet car seat', 'pet seat belt', 'pet ramp', 'pet stairs', 'pet fence',
    'pet gate', 'pet playpen', 'pet pool', 'pet fountain', 'pet filter', 'pet pump', 'pet heater',
    'pet cooler', 'pet humidifier', 'pet dehumidifier', 'pet air purifier', 'pet camera',
    'pet monitor', 'pet tracker', 'pet gps', 'pet smart', 'pet automatic', 'pet interactive',
    'pet training', 'pet clicker', 'pet whistle', 'pet bell',
    
    # Baby Products
    'baby',
    
    # Audio & Video
    'audio', 'video', 'recording', 'streaming', 'broadcasting'
}

# TikTok product categorization hierarchy
TIKTOK_CATEGORIES = {
    "Beauty and Personal Care": {
        "Makeup": ["Foundation", "Concealer", "Powder", "Blush", "Bronzer", "Highlighter", "Eyeshadow", "Eyeliner", "Mascara", "Lipstick", "Lip gloss", "Lip liner", "Nail polish", "Makeup brushes", "Makeup sponges", "Makeup remover"],
        "Skincare": ["Cleanser", "Toner", "Serum", "Moisturizer", "Sunscreen", "Face mask", "Eye cream", "Acne treatment", "Anti-aging", "Exfoliator", "Face oil", "Face mist"],
        "Hair Care": ["Shampoo", "Conditioner", "Hair mask", "Hair oil", "Hair spray", "Hair gel", "Hair wax", "Hair dye", "Hair extensions", "Hair accessories"],
        "Fragrance": ["Perfume", "Body spray", "Deodorant", "Body lotion", "Body wash", "Soap"],
        "Tools": ["Hair dryer", "Straightener", "Curling iron", "Makeup mirror", "Tweezers", "Nail clippers", "Razor", "Electric shaver"]
    },
    "Fashion": {
        "Clothing": ["Dresses", "Tops", "Bottoms", "Outerwear", "Activewear", "Lingerie", "Swimwear", "Shoes", "Bags", "Accessories"],
        "Jewelry": ["Necklaces", "Earrings", "Bracelets", "Rings", "Watches", "Anklets"],
        "Accessories": ["Hats", "Scarves", "Belts", "Sunglasses", "Gloves", "Wallets", "Phone cases"]
    },
    "Electronics": {
        "Mobile Devices": ["Smartphones", "Tablets", "Laptops", "Smartwatches", "Earbuds", "Headphones"],
        "Home Electronics": ["TVs", "Speakers", "Cameras", "Gaming consoles", "Smart home devices"],
        "Accessories": ["Chargers", "Cables", "Cases", "Screen protectors", "Power banks"]
    },
    "Home and Garden": {
        "Kitchen": ["Appliances", "Cookware", "Utensils", "Storage", "Decor"],
        "Furniture": ["Living room", "Bedroom", "Office", "Outdoor", "Storage"],
        "Decor": ["Lighting", "Art", "Plants", "Rugs", "Curtains", "Candles"]
    },
    "Health and Wellness": {
        "Fitness": ["Exercise equipment", "Workout clothes", "Supplements", "Fitness trackers"],
        "Nutrition": ["Vitamins", "Protein powder", "Superfoods", "Healthy snacks"],
        "Wellness": ["Essential oils", "Meditation apps", "Yoga mats", "Massage tools"]
    },
    "Food and Beverages": {
        "Snacks": ["Chips", "Nuts", "Candy", "Chocolate", "Dried fruits"],
        "Beverages": ["Coffee", "Tea", "Juice", "Soda", "Energy drinks", "Water"],
        "Cooking": ["Ingredients", "Spices", "Oils", "Sauces", "Condiments"]
    },
    "Toys and Entertainment": {
        "Toys": ["Educational toys", "Building blocks", "Dolls", "Action figures", "Board games"],
        "Books": ["Fiction", "Non-fiction", "Children's books", "Educational"],
        "Hobbies": ["Arts and crafts", "Collectibles", "Musical instruments"]
    },
    "Sports and Outdoor": {
        "Sports Equipment": ["Balls", "Rackets", "Weights", "Yoga mats", "Running gear"],
        "Outdoor": ["Camping gear", "Hiking equipment", "Bicycles", "Skateboards"],
        "Fitness": ["Gym equipment", "Workout clothes", "Sports shoes"]
    },
    "Baby and Kids": {
        "Baby Care": ["Diapers", "Baby food", "Baby clothes", "Baby toys", "Baby gear"],
        "Kids Fashion": ["Children's clothing", "Kids shoes", "Kids accessories"],
        "Education": ["Learning toys", "Books", "School supplies"]
    },
    "Pet Supplies": {
        "Pet Food": ["Dog food", "Cat food", "Bird food", "Fish food"],
        "Pet Care": ["Grooming supplies", "Toys", "Beds", "Carriers"],
        "Pet Health": ["Vitamins", "Medicine", "Dental care", "Flea treatment"]
    }
} 

# === Audio Models (Whisper, Silero-VAD, Demucs, Shazam) ===

os.environ['NUMBA_CACHE_DIR'] = tempfile.gettempdir()

def pip_install(package):
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        logger.error(f"Failed to install {package}: {e}")

whisper_model = None
try:
    import whisper
    # Load config to get whisper model size
    config = load_config()
    whisper_config = config.get('models', {}).get('whisper', {})
    model_size = whisper_config.get('model_size', 'large-v3')
    
    logger.info(f"Loading Whisper model ({model_size})...")
    try:
        whisper_model = whisper.load_model(model_size, device="cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Whisper {model_size} loaded on {'CUDA' if torch.cuda.is_available() else 'CPU'}!")
    except Exception as e:
        logger.warning(f"Failed to load Whisper {model_size}: {e}")
        logger.info("Trying to load smaller model as fallback...")
        try:
            whisper_model = whisper.load_model("small", device="cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Whisper small loaded as fallback!")
        except Exception as e2:
            logger.error(f"Failed to load Whisper small as well: {e2}")
            whisper_model = None
except ImportError as e:
    logger.error(f"Error importing whisper: {e}")
    whisper_model = None

# ========== Silero VAD ==========
silero_vad_model = None
try:
    vad_package = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    silero_vad_model = vad_package[0] if isinstance(vad_package, tuple) else vad_package
    silero_vad_model = silero_vad_model.to("cuda" if torch.cuda.is_available() else "cpu")
    silero_vad_model.eval()
    logger.info("Silero VAD model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Silero VAD model: {e}")
    silero_vad_model = None

# ========== Demucs (Hybrid) ==========
demucs_model = None
try:
    try:
        from demucs.pretrained import get_model
    except ImportError:
        get_model = None
    if get_model is not None:
        logger.info("Loading Demucs model (pip version)...")
        demucs_model = get_model("htdemucs")
        if hasattr(demucs_model, 'to'):
            demucs_model = demucs_model.to("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(demucs_model, 'eval'):
            demucs_model.eval()
        logger.info("Demucs model loaded successfully")
    else:
        logger.warning("demucs.pretrained.get_model not available.")
        demucs_model = None
except Exception as e:
    logger.error(f"Error loading Demucs model: {e}")
    logger.warning("Demucs will be unavailable.")
    demucs_model = None

# ========== Shazamio (for music recognition) ==========
shazamio = None
shazam_client = None
try:
    try:
        import shazamio
        from shazamio import Shazam
    except ImportError:
        shazamio = None
        Shazam = None
    if shazamio is not None and Shazam is not None:
        logger.info("Shazamio imported successfully")
        shazam_client = Shazam()
    else:
        logger.warning("shazamio not found, installing...")
        pip_install('shazamio')
        try:
            import shazamio
            from shazamio import Shazam
            logger.info("Shazamio installed and imported successfully")
            shazam_client = Shazam()
        except Exception as e:
            logger.error(f"Shazamio still not available: {e}")
            shazamio = None
            shazam_client = None
except Exception as e:
    logger.error(f"Error loading Shazamio: {e}")
    shazamio = None
    shazam_client = None

__all__ = [
    'yolo_model', 'blip_processor', 'blip_model', 'clip_model', 'clip_preprocess',
    'gpt4o_config', 'calc_gpt4o_price', 'image_to_data_url', 'ensure_openai_key',
    'whisper_model', 'silero_vad_model', 'demucs_model', 'shazam_client'
] 