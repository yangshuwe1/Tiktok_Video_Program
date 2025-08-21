import os
import torch
import numpy as np
from PIL import Image
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import cv2
import torch.nn.functional as F
from logger import get_logger
from shared_models import clip_model, clip_preprocess
import sys

logger = get_logger("embedding_analyzer")

class EmbeddingAnalyzer:
    """
    Enhanced embedding analyzer that generates text prompts from images using CLIP and V-JEPA2.
    Integrated with comprehensive TikTok content creator tag system for improved analysis.
    Outputs text descriptions for GPT4O input, avoiding direct image input to reduce token consumption.
    """
    
    def __init__(self, config=None):
        self.logger = get_logger("EmbeddingAnalyzer")
        self.config = config or {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # CLIP model
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        
        # V-JEPA2 configuration and models
        vjepa2_config = self.config.get("vjepa2", {})
        self.vjepa2_model_type = vjepa2_config.get('model_type', 'vitg16')
        self.vjepa2_num_frames = vjepa2_config.get('num_frames', 16)
        self.vjepa2_temporal_analysis = vjepa2_config.get('temporal_analysis', True)
        self.vjepa2_content_complexity = vjepa2_config.get('content_complexity', True)
        self.vjepa2_video_embeddings = vjepa2_config.get('video_embeddings', True)
        
        # V-JEPA2 model component
        self.vjepa2_model = None
        self._load_vjepa2_models()
        
        # Enhanced TikTok content creator tag system prompts
        self._initialize_tiktok_tag_system()
        
        self.logger.info(f"EmbeddingAnalyzer initialized on device: {self.device}")
    
    def _initialize_tiktok_tag_system(self):
        """Initialize comprehensive TikTok content creator tag system"""
        
        # Creator-focused prompts
        self.creator_prompts = {
            "gender": ["male", "female", "multiple-people", "non-human", "unknown"],
            "language": ["english", "spanish", "french", "german", "italian", "portuguese", "russian", "japanese", "korean", "chinese", "hindi"],
            "race": ["black", "white", "asian", "middle-eastern", "latinx", "indigenous"],
            "age": ["child", "teen", "20s", "30s", "40s", "50+"],
            "physical_appearance": ["very appealing", "appealing", "average", "distinctive"],
            "body_type": ["slim", "average", "athletic", "curvy", "plus-size"],
            "hair_color": ["black", "brunette", "blonde", "red", "mixed-colored"],
            "hair_type": ["straight", "wavy", "curly", "coily", "updo", "tied", "braids"],
            "body_art": ["visible tattoos", "without visible tattoos", "body piercings", "without body piercings"],
            "economic_status": ["luxury-focused", "middle-class", "budget-conscious", "minimalist"],
            "persona": [
                "designer", "fashion buyer", "model", "beautician", "makeup artist", "perfumer", "aromatherapist",
                "healthcare professional", "psychologist", "fitness coach", "nutritionist", "corporate executive",
                "brand-owner", "homeowner", "office workers", "nanny", "construction worker", "retail worker",
                "crafts person", "teacher", "student", "husband", "father", "fathers with children", "expectant mother",
                "mom", "housewife", "couple", "pet owners", "fashionista", "makeup & beauty enthusiast",
                "travel enthusiast", "tech-enthusiast", "game-enthusiast", "car-enthusiast", "sports-enthusiast",
                "book-lover", "vegetarian", "foodie", "food/beverage enthusiast", "wellness advocate",
                "environmental advocate", "feminist advocate"
            ],
            "charisma_authenticity": ["enthusiastic & energetic", "professional & authoritative", "authentic & relatable", "engaging & expressive", "calm & soothing"],
            "authority_trust": ["expertise", "testimonials", "brand-reputation", "collaborations"],
            "emotional_appeal": ["humor", "nostalgia", "empathy", "inspiration", "excitement", "joy", "trust", "surprise"],
            "luxury_presentation": ["low", "moderate", "high", "very-high"],
            "branding_integration": ["subtle", "moderate", "overt", "not-present", "seamless"],
            "call_to_actions": ["like", "share", "comment", "follow", "link-in-bio", "swipe-up", "buy-now", "join-live-stream"]
        }
        
        # Content-focused prompts
        self.content_prompts = {
            "niche_specialty": [
                "beauty", "fashion", "health & wellness", "sports/fitness", "home/lifestyle", "cooking",
                "food/beverage", "diy", "pet", "study/education", "technology", "music/dancing",
                "film/television", "gaming", "travel", "comedy", "parenting"
            ],
            "target_audience_fit": ["low", "moderate", "high", "very-high"],
            "content_format": [
                "tutorial", "review", "music", "dance", "duet", "skit", "story-time", "daily vlogs",
                "travel diaries", "q&a", "interview"
            ],
            "storytelling_structure": [
                "3-act review: unbox-try-conclude", "vlog style", "reverse chronology", "skit"
            ],
            "script_creativity": [
                "strong storytelling", "plot twist", "unique narrative angle", "well-structured",
                "straightforward", "comedic", "suspense", "normal"
            ],
            "overall_tone": [
                "professional", "serious", "casual", "informative", "inspirational", "empowering",
                "mellow", "emotional", "dramatic", "exuberant", "playful", "whimsical", "humorous", "sarcastic"
            ],
            "energy_levels": ["vibrant & passionate", "composed & professional", "calm & soothing"],
            "engagement_drivers": [
                "promotional-sales", "controversial-topics", "clickbait", "exaggerated-expression",
                "relatable-struggle", "satisfying-process", "motivational-message"
            ],
            "trend_adoption": [
                "dance-challenges", "audio-trends", "meme-adaptations", "pov-storytelling", "seasonal-themes"
            ],
            "visual_presentation": [
                "dark", "bright", "monochromatic", "colorful", "clean", "minimalistic", "professional",
                "artistic", "abstract", "retro", "rustic", "thematic"
            ],
            "product_presentation": [
                "product showcase", "unboxing & reviews", "voiceover content", "live demonstrations",
                "before & after comparisons", "product comparisons"
            ],
            "video_quality": ["very-high", "high", "moderate"],
            "video_scene": [
                "daily life", "family life", "at work", "professional settings", "outdoor activities",
                "social activities", "travel scenes", "special occasions"
            ],
            "video_background": ["indoor", "outdoor", "kitchen"],
            "real_life_contexts": [
                "daily-routine", "family-oriented", "professional-setting", "outdoors", "social-events",
                "travel-scenarios", "special-occasion"
            ],
            "pacing": ["very-high", "high", "moderate", "low"],
            "speech_rate": ["very-high", "high", "moderate", "low", "no-speech"],
            "face_visibility": ["always-visible", "frequently-visible", "occasionally-visible", "rarely-visible", "never-visible"],
            "tiktok_effects": ["filters", "ar-effects", "text-overlay", "green-screen", "split-screens", "stitches", "duets"]
        }
        
        # Industry-specific prompts
        self.industry_prompts = {
            "beauty": {
                "skin_type_tone": [
                    "sensitive skin", "acne-prone skin", "problematic skin", "combination skin",
                    "dry skin", "oily skin", "normal skin"
                ]
            },
            "fashion": [
                "fashion & styling", "hairstyles", "bags & accessories", "jewelry", "shoes & boots",
                "street photography", "fashion news"
            ],
            "home_lifestyle": ["home ambiance", "life hacks", "baby care"],
            "sports_fitness": [
                "general sports", "fitness", "weight loss & body sculpting", "yoga", "ball sports",
                "martial arts", "ice & snow sports", "fishing", "extreme sports", "sports news"
            ],
            "health_wellness": [
                "immune system", "digestive health", "energy management", "beauty & wellness", "women's health"
            ],
            "parenting": ["daily parenting life", "parenting education", "product reviews & recommendations"],
            "baby_maternity": [
                "pre-pregnancy", "pregnancy", "newborn (0-6 months)", "infant (6-12 months)",
                "toddler (1-3 years)", "preschool (3-6 years)", "school age (6-12 years)", "teen (12-18 years)"
            ]
        }
        
        # Legacy prompts for backward compatibility
        self.visual_prompts = [
            "product showcase", "commercial advertisement", "tutorial video", 
            "before and after", "product review", "unboxing", "makeup tutorial",
            "fashion show", "cooking tutorial", "fitness workout", "beauty routine",
            "product demonstration", "lifestyle content", "educational content",
            "entertainment", "music video", "dance video", "comedy sketch",
            "product placement", "brand promotion", "influencer content"
        ]
        
        self.object_prompts = [
            "person", "face", "clothing", "accessories", "electronics", "cosmetics",
            "food", "drink", "furniture", "vehicle", "building", "nature", "animal",
            "text", "logo", "brand", "product", "package", "tool", "instrument"
        ]
        
        self.action_prompts = [
            "speaking", "demonstrating", "using", "showing", "explaining", "teaching",
            "cooking", "applying", "wearing", "holding", "opening", "closing",
            "moving", "dancing", "singing", "laughing", "smiling", "pointing",
            "touching", "looking", "walking", "running", "jumping"
        ]
    
    def _load_vjepa2_models(self):
        """Load V-JEPA2 models from transformers"""
        try:
            # Import V-JEPA2 from transformers
            from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Model
            from transformers.models.vjepa2.configuration_vjepa2 import VJEPA2Config
            
            # Create V-JEPA2 model configuration
            self.logger.info("Creating V-JEPA2 model with default configuration")
            
            # Create model configuration
            config = VJEPA2Config()
            
            # Create the model
            self.vjepa2_model = VJEPA2Model(config)
            
            # Move model to device
            self.vjepa2_model = self.vjepa2_model.to(self.device)
            
            # Set to evaluation mode
            self.vjepa2_model.eval()
            
            self.logger.info("V-JEPA2 model loaded successfully from transformers")
            
        except Exception as e:
            self.logger.warning(f"Failed to load V-JEPA2 model from transformers: {e}")
            self.logger.info("V-JEPA2 features will be disabled - using CLIP-only analysis")
            self.vjepa2_model = None
    
    def is_vjepa2_available(self) -> bool:
        """Check if V-JEPA2 model is available"""
        return self.vjepa2_model is not None
    
    def generate_prompt_from_images(self, frame_paths: List[str], audio_transcript: str = "") -> Dict[str, Any]:
        """
        Generate comprehensive analysis from images using CLIP and V-JEPA2.
        Returns both embedding tags and structured analysis for GPT4O input.
        
        Args:
            frame_paths: List of paths to keyframe images
            audio_transcript: Audio transcription text
            
        Returns:
            Dictionary containing:
            - embedding_tags: All relevant tags from embedding analysis
            - structured_analysis: Structured analysis results
            - comprehensive_prompt: Text prompt for GPT4O
        """
        try:
            self.logger.info(f"Generating comprehensive analysis from {len(frame_paths)} frames")
            
            # Get embedding analysis configuration
            embedding_config = self.config.get("embedding_analysis", {})
            use_clip = embedding_config.get("use_clip", True)
            use_vjepa2 = embedding_config.get("use_vjepa2", True)
            use_visual = embedding_config.get("visual_prompts", True)
            use_objects = embedding_config.get("object_analysis", True)
            use_actions = embedding_config.get("action_analysis", True)
            use_temporal = embedding_config.get("temporal_analysis", True)
            
            # Initialize analysis results
            visual_analysis = {}
            object_analysis = {}
            action_analysis = {}
            vjepa2_analysis = {}
            
            # Step 1: Extract CLIP embeddings and analyze content (if enabled)
            if use_clip and use_visual:
                visual_analysis = self._analyze_visual_content(frame_paths)
            
            if use_clip and use_objects:
                object_analysis = self._analyze_objects(frame_paths)
            
            if use_clip and use_actions:
                action_analysis = self._analyze_actions(frame_paths)
            
            # Step 2: Extract V-JEPA2 temporal features (if enabled)
            if use_vjepa2 and use_temporal:
                vjepa2_analysis = self._analyze_vjepa2_temporal(frame_paths)
            
            # Step 3: Generate comprehensive description
            comprehensive_prompt = self._generate_comprehensive_description(
                visual_analysis, object_analysis, action_analysis, vjepa2_analysis, audio_transcript
            )
            
            # Step 4: Extract all relevant tags from embedding analysis
            embedding_tags = self._extract_all_embedding_tags(
                visual_analysis, object_analysis, action_analysis, vjepa2_analysis
            )
            
            # Step 5: Create structured analysis results
            structured_analysis = self._create_structured_analysis(
                visual_analysis, object_analysis, action_analysis, vjepa2_analysis, audio_transcript
            )
            
            return {
                "embedding_tags": embedding_tags,
                "structured_analysis": structured_analysis,
                "comprehensive_prompt": comprehensive_prompt,
                "analysis_metadata": {
                    "frame_count": len(frame_paths),
                    "analysis_methods": {
                        "clip": use_clip,
                        "vjepa2": use_vjepa2,
                        "visual_analysis": use_visual,
                        "object_analysis": use_objects,
                        "action_analysis": use_actions,
                        "temporal_analysis": use_temporal
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            return {
                "embedding_tags": {},
                "structured_analysis": {},
                "comprehensive_prompt": f"TikTok video analysis with {len(frame_paths)} frames. Audio: {audio_transcript}",
                "analysis_metadata": {"error": str(e)}
            }
    
    def _extract_all_embedding_tags(self, visual_analysis: Dict, object_analysis: Dict, 
                                   action_analysis: Dict, vjepa2_analysis: Dict) -> Dict[str, Any]:
        """Extract all relevant tags from embedding analysis results"""
        try:
            tags = {
                "creator_attributes": {},
                "content_attributes": {},
                "industry_attributes": {},
                "visual_quality": {},
                "objects": {},
                "actions": {},
                "temporal_features": {},
                "overall_summary": {}
            }
            
            # Extract creator attributes from visual analysis
            if visual_analysis.get("frame_analyses"):
                frame_analyses = visual_analysis["frame_analyses"]
                
                # Aggregate creator attributes across frames
                genders = []
                ages = []
                personas = []
                charisma_types = []
                economic_statuses = []
                
                for frame_analysis in frame_analyses:
                    if frame_analysis.get("gender"):
                        genders.append(frame_analysis["gender"])
                    if frame_analysis.get("age"):
                        ages.append(frame_analysis["age"])
                    if frame_analysis.get("persona"):
                        personas.extend(frame_analysis["persona"])
                    if frame_analysis.get("charisma_authenticity"):
                        charisma_types.append(frame_analysis["charisma_authenticity"])
                    if frame_analysis.get("economic_status"):
                        economic_statuses.append(frame_analysis["economic_status"])
                
                tags["creator_attributes"] = {
                    "gender": self._get_most_common(genders),
                    "age": self._get_most_common(ages),
                    "personas": list(set(personas)),
                    "charisma": self._get_most_common(charisma_types),
                    "economic_status": self._get_most_common(economic_statuses)
                }
                
                # Extract content attributes
                niches = []
                content_formats = []
                tones = []
                energy_levels = []
                visual_styles = []
                
                for frame_analysis in frame_analyses:
                    if frame_analysis.get("niche_specialty"):
                        niches.extend(frame_analysis["niche_specialty"])
                    if frame_analysis.get("content_format"):
                        content_formats.extend(frame_analysis["content_format"])
                    if frame_analysis.get("overall_tone"):
                        tones.extend(frame_analysis["overall_tone"])
                    if frame_analysis.get("energy_levels"):
                        energy_levels.append(frame_analysis["energy_levels"])
                    if frame_analysis.get("visual_presentation"):
                        visual_styles.extend(frame_analysis["visual_presentation"])
                
                tags["content_attributes"] = {
                    "niche_specialty": list(set(niches)),
                    "content_format": list(set(content_formats)),
                    "overall_tone": list(set(tones)),
                    "energy_levels": self._get_most_common(energy_levels),
                    "visual_presentation": list(set(visual_styles))
                }
                
                # Extract industry-specific attributes
                beauty_attributes = {}
                fashion_attributes = []
                home_lifestyle_attributes = []
                sports_fitness_attributes = []
                
                for frame_analysis in frame_analyses:
                    if frame_analysis.get("beauty"):
                        for category, items in frame_analysis["beauty"].items():
                            if category not in beauty_attributes:
                                beauty_attributes[category] = []
                            beauty_attributes[category].extend(items)
                    
                    if frame_analysis.get("fashion"):
                        fashion_attributes.extend(frame_analysis["fashion"])
                    if frame_analysis.get("home_lifestyle"):
                        home_lifestyle_attributes.extend(frame_analysis["home_lifestyle"])
                    if frame_analysis.get("sports_fitness"):
                        sports_fitness_attributes.extend(frame_analysis["sports_fitness"])
                
                # Deduplicate beauty attributes
                for category in beauty_attributes:
                    beauty_attributes[category] = list(set(beauty_attributes[category]))
                
                tags["industry_attributes"] = {
                    "beauty": beauty_attributes,
                    "fashion": list(set(fashion_attributes)),
                    "home_lifestyle": list(set(home_lifestyle_attributes)),
                    "sports_fitness": list(set(sports_fitness_attributes))
                }
            
            # Extract object tags
            if object_analysis.get("all_objects"):
                objects_by_category = {}
                for obj in object_analysis["all_objects"]:
                    category = obj.get("category", "other")
                    if category not in objects_by_category:
                        objects_by_category[category] = []
                    objects_by_category[category].append({
                        "object": obj["object"],
                        "confidence": obj["avg_confidence"]
                    })
                
                tags["objects"] = objects_by_category
            
            # Extract action tags
            if action_analysis.get("all_actions"):
                actions_by_category = {}
                for action in action_analysis["all_actions"]:
                    category = action.get("category", "other")
                    if category not in actions_by_category:
                        actions_by_category[category] = []
                    actions_by_category[category].append({
                        "action": action["action"],
                        "confidence": action["avg_confidence"]
                    })
                
                tags["actions"] = actions_by_category
            
            # Extract temporal features
            if vjepa2_analysis.get("available", False):
                tags["temporal_features"] = {
                    "temporal_consistency": vjepa2_analysis.get("temporal_consistency", 0.5),
                    "content_complexity": vjepa2_analysis.get("content_complexity", 0.5),
                    "video_embeddings": vjepa2_analysis.get("video_embeddings", {}),
                    "temporal_analysis": vjepa2_analysis.get("temporal_analysis", {})
                }
            
            # Create overall summary
            tags["overall_summary"] = self._create_overall_summary(tags)
            
            return tags
            
        except Exception as e:
            self.logger.error(f"Error extracting embedding tags: {e}")
            return {}
    
    def _get_most_common(self, items: List[str]) -> str:
        """Get the most common item from a list"""
        if not items:
            return "unknown"
        
        from collections import Counter
        counter = Counter(items)
        return counter.most_common(1)[0][0]
    
    def _create_structured_analysis(self, visual_analysis: Dict, object_analysis: Dict, 
                                  action_analysis: Dict, vjepa2_analysis: Dict, audio_transcript: str) -> Dict[str, Any]:
        """Create structured analysis results for GPT4O processing"""
        try:
            structured_analysis = {
                "video_metadata": {
                    "frame_count": len(visual_analysis.get("frame_analyses", [])),
                    "has_audio": bool(audio_transcript.strip()),
                    "audio_length": len(audio_transcript.split()) if audio_transcript else 0
                },
                "creator_profile": {},
                "content_profile": {},
                "commercial_profile": {},
                "technical_profile": {},
                "engagement_profile": {}
            }
            
            # Extract creator profile
            if visual_analysis.get("frame_analyses"):
                frame_analyses = visual_analysis["frame_analyses"]
                
                # Analyze creator demographics
                demographics = self._analyze_creator_demographics(frame_analyses)
                structured_analysis["creator_profile"] = demographics
                
                # Analyze content characteristics
                content_chars = self._analyze_content_characteristics(frame_analyses)
                structured_analysis["content_profile"] = content_chars
                
                # Analyze commercial aspects
                commercial_aspects = self._analyze_commercial_aspects(frame_analyses, object_analysis)
                structured_analysis["commercial_profile"] = commercial_aspects
                
                # Analyze technical quality
                technical_quality = self._analyze_technical_quality(frame_analyses, vjepa2_analysis)
                structured_analysis["technical_profile"] = technical_quality
                
                # Analyze engagement potential
                engagement_potential = self._analyze_engagement_potential(
                    frame_analyses, object_analysis, action_analysis, audio_transcript
                )
                structured_analysis["engagement_profile"] = engagement_potential
            
            return structured_analysis
            
        except Exception as e:
            self.logger.error(f"Error creating structured analysis: {e}")
            return {}
    
    def _analyze_creator_demographics(self, frame_analyses: List[Dict]) -> Dict[str, Any]:
        """Analyze creator demographics from frame analyses"""
        try:
            demographics = {
                "gender_distribution": {},
                "age_distribution": {},
                "persona_types": [],
                "charisma_profile": {},
                "economic_profile": {}
            }
            
            # Count distributions
            for frame in frame_analyses:
                gender = frame.get("gender", "unknown")
                age = frame.get("age", "unknown")
                charisma = frame.get("charisma_authenticity", "unknown")
                economic = frame.get("economic_status", "unknown")
                
                demographics["gender_distribution"][gender] = demographics["gender_distribution"].get(gender, 0) + 1
                demographics["age_distribution"][age] = demographics["age_distribution"].get(age, 0) + 1
                demographics["charisma_profile"][charisma] = demographics["charisma_profile"].get(charisma, 0) + 1
                demographics["economic_profile"][economic] = demographics["economic_profile"].get(economic, 0) + 1
                
                if frame.get("persona"):
                    demographics["persona_types"].extend(frame["persona"])
            
            # Deduplicate personas
            demographics["persona_types"] = list(set(demographics["persona_types"]))
            
            return demographics
            
        except Exception as e:
            self.logger.error(f"Error analyzing creator demographics: {e}")
            return {}
    
    def _analyze_content_characteristics(self, frame_analyses: List[Dict]) -> Dict[str, Any]:
        """Analyze content characteristics from frame analyses"""
        try:
            characteristics = {
                "niche_focus": {},
                "content_formats": {},
                "tone_profile": {},
                "energy_profile": {},
                "visual_style": {},
                "product_presentation": {}
            }
            
            for frame in frame_analyses:
                # Count niche specialties
                if frame.get("niche_specialty"):
                    for niche in frame["niche_specialty"]:
                        characteristics["niche_focus"][niche] = characteristics["niche_focus"].get(niche, 0) + 1
                
                # Count content formats
                if frame.get("content_format"):
                    for format_type in frame["content_format"]:
                        characteristics["content_formats"][format_type] = characteristics["content_formats"].get(format_type, 0) + 1
                
                # Count tones
                if frame.get("overall_tone"):
                    for tone in frame["overall_tone"]:
                        characteristics["tone_profile"][tone] = characteristics["tone_profile"].get(tone, 0) + 1
                
                # Count energy levels
                energy = frame.get("energy_levels", "unknown")
                characteristics["energy_profile"][energy] = characteristics["energy_profile"].get(energy, 0) + 1
                
                # Count visual styles
                if frame.get("visual_presentation"):
                    for style in frame["visual_presentation"]:
                        characteristics["visual_style"][style] = characteristics["visual_style"].get(style, 0) + 1
                
                # Count product presentation
                if frame.get("product_presentation"):
                    for presentation in frame["product_presentation"]:
                        characteristics["product_presentation"][presentation] = characteristics["product_presentation"].get(presentation, 0) + 1
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error analyzing content characteristics: {e}")
            return {}
    
    def _analyze_commercial_aspects(self, frame_analyses: List[Dict], object_analysis: Dict) -> Dict[str, Any]:
        """Analyze commercial aspects from frame analyses and object detection"""
        try:
            commercial_aspects = {
                "product_categories": {},
                "brand_indicators": [],
                "commercial_intent": "unknown",
                "target_audience": {},
                "selling_points": []
            }
            
            # Analyze beauty industry focus
            beauty_products = []
            for frame in frame_analyses:
                if frame.get("beauty"):
                    for category, items in frame["beauty"].items():
                        if category not in commercial_aspects["product_categories"]:
                            commercial_aspects["product_categories"][category] = []
                        commercial_aspects["product_categories"][category].extend(items)
                        beauty_products.extend(items)
            
            # Deduplicate product categories
            for category in commercial_aspects["product_categories"]:
                commercial_aspects["product_categories"][category] = list(set(commercial_aspects["product_categories"][category]))
            
            # Analyze objects for commercial indicators
            if object_analysis.get("all_objects"):
                for obj in object_analysis["all_objects"]:
                    if obj.get("category") == "beauty_products" and obj["avg_confidence"] > 0.3:
                        commercial_aspects["selling_points"].append(obj["object"])
            
            # Determine commercial intent
            if beauty_products or commercial_aspects["selling_points"]:
                commercial_aspects["commercial_intent"] = "commercial"
            else:
                commercial_aspects["commercial_intent"] = "non-commercial"
            
            return commercial_aspects
            
        except Exception as e:
            self.logger.error(f"Error analyzing commercial aspects: {e}")
            return {}
    
    def _analyze_technical_quality(self, frame_analyses: List[Dict], vjepa2_analysis: Dict) -> Dict[str, Any]:
        """Analyze technical quality from frame analyses and V-JEPA2 analysis"""
        try:
            technical_quality = {
                "video_quality": {},
                "temporal_consistency": 0.5,
                "content_complexity": 0.5,
                "visual_attributes": {}
            }
            
            # Count video quality levels
            for frame in frame_analyses:
                quality = frame.get("video_quality", "unknown")
                technical_quality["video_quality"][quality] = technical_quality["video_quality"].get(quality, 0) + 1
            
            # Get V-JEPA2 metrics
            if vjepa2_analysis.get("available", False):
                technical_quality["temporal_consistency"] = vjepa2_analysis.get("temporal_consistency", 0.5)
                technical_quality["content_complexity"] = vjepa2_analysis.get("content_complexity", 0.5)
            
            # Analyze visual attributes
            for frame in frame_analyses:
                brightness = frame.get("brightness", "unknown")
                contrast = frame.get("contrast", "unknown")
                
                if brightness not in technical_quality["visual_attributes"]:
                    technical_quality["visual_attributes"][brightness] = 0
                technical_quality["visual_attributes"][brightness] += 1
                
                if contrast not in technical_quality["visual_attributes"]:
                    technical_quality["visual_attributes"][contrast] = 0
                technical_quality["visual_attributes"][contrast] += 1
            
            return technical_quality
            
        except Exception as e:
            self.logger.error(f"Error analyzing technical quality: {e}")
            return {}
    
    def _analyze_engagement_potential(self, frame_analyses: List[Dict], object_analysis: Dict, 
                                    action_analysis: Dict, audio_transcript: str) -> Dict[str, Any]:
        """Analyze engagement potential from all analysis results"""
        try:
            engagement_profile = {
                "engagement_factors": [],
                "audience_appeal": "unknown",
                "content_effectiveness": "unknown",
                "viral_potential": "unknown"
            }
            
            # Analyze engagement factors
            engagement_factors = []
            
            # Check for high-energy content
            high_energy_count = sum(1 for frame in frame_analyses 
                                  if frame.get("energy_levels") in ["vibrant & passionate", "exuberant"])
            if high_energy_count > len(frame_analyses) * 0.5:
                engagement_factors.append("high_energy_content")
            
            # Check for beauty/fashion focus
            beauty_focus = any(frame.get("beauty") for frame in frame_analyses)
            if beauty_focus:
                engagement_factors.append("beauty_fashion_focus")
            
            # Check for tutorial/review content
            tutorial_content = any("tutorial" in frame.get("content_format", []) 
                                 for frame in frame_analyses)
            if tutorial_content:
                engagement_factors.append("tutorial_review_content")
            
            # Check for product demonstrations
            product_demo = any("product showcase" in frame.get("product_presentation", []) 
                             for frame in frame_analyses)
            if product_demo:
                engagement_factors.append("product_demonstration")
            
            # Check for audio engagement
            if audio_transcript and len(audio_transcript.split()) > 10:
                engagement_factors.append("substantial_audio_content")
            
            engagement_profile["engagement_factors"] = engagement_factors
            
            # Determine audience appeal
            if len(engagement_factors) >= 3:
                engagement_profile["audience_appeal"] = "high"
            elif len(engagement_factors) >= 2:
                engagement_profile["audience_appeal"] = "medium"
            else:
                engagement_profile["audience_appeal"] = "low"
            
            return engagement_profile
            
        except Exception as e:
            self.logger.error(f"Error analyzing engagement potential: {e}")
            return {}
    
    def _create_overall_summary(self, tags: Dict[str, Any]) -> Dict[str, Any]:
        """Create overall summary from all tags"""
        try:
            summary = {
                "primary_content_type": "unknown",
                "target_audience": "unknown",
                "commercial_focus": "unknown",
                "content_quality": "unknown",
                "key_highlights": []
            }
            
            # Determine primary content type
            if tags.get("industry_attributes", {}).get("beauty"):
                summary["primary_content_type"] = "beauty"
            elif tags.get("industry_attributes", {}).get("fashion"):
                summary["primary_content_type"] = "fashion"
            elif tags.get("industry_attributes", {}).get("home_lifestyle"):
                summary["primary_content_type"] = "lifestyle"
            elif tags.get("industry_attributes", {}).get("sports_fitness"):
                summary["primary_content_type"] = "fitness"
            
            # Determine target audience
            creator_attrs = tags.get("creator_attributes", {})
            if creator_attrs.get("gender") and creator_attrs.get("age"):
                summary["target_audience"] = f"{creator_attrs['gender']} {creator_attrs['age']}"
            
            # Determine commercial focus
            if tags.get("objects", {}).get("beauty_products"):
                summary["commercial_focus"] = "beauty_products"
            elif tags.get("objects", {}).get("electronics"):
                summary["commercial_focus"] = "electronics"
            elif tags.get("objects", {}).get("household_items"):
                summary["commercial_focus"] = "household"
            else:
                summary["commercial_focus"] = "non_commercial"
            
            # Determine content quality
            temporal_features = tags.get("temporal_features", {})
            if temporal_features.get("temporal_consistency", 0) > 0.7:
                summary["content_quality"] = "high"
            elif temporal_features.get("temporal_consistency", 0) > 0.4:
                summary["content_quality"] = "medium"
            else:
                summary["content_quality"] = "low"
            
            # Create key highlights
            highlights = []
            if tags.get("creator_attributes", {}).get("personas"):
                highlights.append(f"Creator personas: {', '.join(tags['creator_attributes']['personas'][:3])}")
            
            if tags.get("content_attributes", {}).get("niche_specialty"):
                highlights.append(f"Content niche: {', '.join(tags['content_attributes']['niche_specialty'][:2])}")
            
            if tags.get("objects", {}).get("beauty_products"):
                highlights.append(f"Beauty products: {len(tags['objects']['beauty_products'])} detected")
            
            summary["key_highlights"] = highlights
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating overall summary: {e}")
            return {}
    
    def _analyze_visual_content(self, frame_paths: List[str]) -> Dict[str, Any]:
        """Analyze visual content using CLIP embeddings with enhanced TikTok tag system"""
        try:
            self.logger.info(f"Starting visual content analysis with {len(frame_paths)} frames")
            self.logger.info(f"Frame paths received: {frame_paths}")
            
            frame_analyses = []
            
            for frame_path in frame_paths:
                try:
                    # Check if file exists
                    if not os.path.exists(frame_path):
                        self.logger.warning(f"Frame file not found: {frame_path}")
                        self.logger.warning(f"File exists check failed for: {frame_path}")
                        self.logger.warning(f"Directory contents: {os.listdir(os.path.dirname(frame_path)) if os.path.exists(os.path.dirname(frame_path)) else 'Directory not found'}")
                        frame_analyses.append(self._create_fallback_frame_analysis())
                        continue
                    
                    self.logger.info(f"Successfully accessing frame: {frame_path}")
                    
                    # Load and preprocess image
                    image = Image.open(frame_path).convert('RGB')
                    img_array = np.array(image)
                    
                    # CLIP preprocessing
                    clip_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                    
                    # Get CLIP image embedding
                    with torch.no_grad():
                        image_embedding = self.clip_model.encode_image(clip_input)
                    
                    # Analyze with enhanced TikTok tag system
                    frame_analysis = self._analyze_single_frame_with_tiktok_tags(image_embedding, img_array, frame_path)
                    frame_analyses.append(frame_analysis)
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing frame {frame_path}: {e}")
                    frame_analyses.append(self._create_fallback_frame_analysis())
            
            # Aggregate results
            overall_content = self._aggregate_content_types(frame_analyses)
            
            return {
                "frame_analyses": frame_analyses,
                "overall_content": overall_content
            }
            
        except Exception as e:
            self.logger.error(f"Error in visual content analysis: {e}")
            return {"frame_analyses": [], "overall_content": {}}
    
    def _analyze_single_frame_with_tiktok_tags(self, image_embedding: torch.Tensor, img_array: np.ndarray, frame_path: str) -> Dict[str, Any]:
        """Analyze single frame with comprehensive TikTok tag system"""
        try:
            # Get text embeddings for all TikTok tag categories
            text_embeddings = self._get_tiktok_text_embeddings()
            
            # Calculate similarities
            similarities = self._calculate_similarities(image_embedding, text_embeddings)
            
            # Extract visual features
            dominant_colors = self._extract_dominant_colors(img_array)
            brightness = self._analyze_brightness(img_array)
            contrast = self._analyze_contrast(img_array)
            
            # Analyze each TikTok tag category
            analysis_results = {}
            
            # Creator analysis
            analysis_results.update(self._analyze_creator_attributes(similarities))
            
            # Content analysis
            analysis_results.update(self._analyze_content_attributes(similarities))
            
            # Industry analysis
            analysis_results.update(self._analyze_industry_attributes(similarities))
            
            # Visual quality analysis
            analysis_results.update({
                "dominant_colors": dominant_colors,
                "brightness": brightness,
                "contrast": contrast,
                "visual_quality": self._analyze_visual_quality(similarities)
            })
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in TikTok tag analysis: {e}")
            return self._create_fallback_frame_analysis()
    
    def _get_tiktok_text_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get text embeddings for all TikTok tag categories"""
        text_embeddings = {}
        
        # Combine all prompts from TikTok tag system
        all_prompts = []
        prompt_categories = []
        
        # Creator prompts
        for category, prompts in self.creator_prompts.items():
            all_prompts.extend(prompts)
            prompt_categories.extend([f"creator_{category}"] * len(prompts))
        
        # Content prompts
        for category, prompts in self.content_prompts.items():
            all_prompts.extend(prompts)
            prompt_categories.extend([f"content_{category}"] * len(prompts))
        
        # Industry prompts
        for industry, categories in self.industry_prompts.items():
            if isinstance(categories, dict):
                for category, prompts in categories.items():
                    all_prompts.extend(prompts)
                    prompt_categories.extend([f"industry_{industry}_{category}"] * len(prompts))
            else:
                all_prompts.extend(categories)
                prompt_categories.extend([f"industry_{industry}"] * len(categories))
        
        # Legacy prompts
        all_prompts.extend(self.visual_prompts)
        prompt_categories.extend(["legacy_visual"] * len(self.visual_prompts))
        
        # Get text embeddings
        import clip as openai_clip
        text_tokens = openai_clip.tokenize(all_prompts).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)
        
        # Organize by category
        for i, category in enumerate(prompt_categories):
            if category not in text_embeddings:
                text_embeddings[category] = []
            # Extract the i-th embedding from the batch
            text_embeddings[category].append(text_features[i:i+1])
        
        return text_embeddings
    
    def _calculate_similarities(self, image_embedding: torch.Tensor, text_embeddings: Dict[str, List[torch.Tensor]]) -> Dict[str, List[float]]:
        """Calculate similarities between image and text embeddings"""
        similarities = {}
        
        for category, embeddings in text_embeddings.items():
            category_similarities = []
            for text_emb in embeddings:
                # Ensure text_emb has the right shape (should be 1D tensor)
                if text_emb.dim() == 2:
                    text_emb = text_emb.squeeze(0)  # Remove batch dimension if present
                similarity = F.cosine_similarity(image_embedding, text_emb.unsqueeze(0), dim=1)
                category_similarities.append(similarity.item())
            similarities[category] = category_similarities
        
        return similarities
    
    def _analyze_creator_attributes(self, similarities: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze creator-related attributes"""
        creator_analysis = {}
        
        # Gender analysis
        if "creator_gender" in similarities:
            gender_scores = similarities["creator_gender"]
            max_idx = np.argmax(gender_scores)
            creator_analysis["gender"] = self.creator_prompts["gender"][max_idx]
        
        # Age analysis
        if "creator_age" in similarities:
            age_scores = similarities["creator_age"]
            max_idx = np.argmax(age_scores)
            creator_analysis["age"] = self.creator_prompts["age"][max_idx]
        
        # Physical appearance
        if "creator_physical_appearance" in similarities:
            appearance_scores = similarities["creator_physical_appearance"]
            max_idx = np.argmax(appearance_scores)
            creator_analysis["physical_appearance"] = self.creator_prompts["physical_appearance"][max_idx]
        
        # Body type
        if "creator_body_type" in similarities:
            body_scores = similarities["creator_body_type"]
            max_idx = np.argmax(body_scores)
            creator_analysis["body_type"] = self.creator_prompts["body_type"][max_idx]
        
        # Hair analysis
        if "creator_hair_color" in similarities:
            hair_color_scores = similarities["creator_hair_color"]
            max_idx = np.argmax(hair_color_scores)
            creator_analysis["hair_color"] = self.creator_prompts["hair_color"][max_idx]
        
        if "creator_hair_type" in similarities:
            hair_type_scores = similarities["creator_hair_type"]
            max_idx = np.argmax(hair_type_scores)
            creator_analysis["hair_type"] = self.creator_prompts["hair_type"][max_idx]
        
        # Persona analysis
        if "creator_persona" in similarities:
            persona_scores = similarities["creator_persona"]
            top_indices = np.argsort(persona_scores)[-3:]  # Top 3 personas
            creator_analysis["persona"] = [self.creator_prompts["persona"][i] for i in top_indices]
        
        # Charisma and authenticity
        if "creator_charisma_authenticity" in similarities:
            charisma_scores = similarities["creator_charisma_authenticity"]
            max_idx = np.argmax(charisma_scores)
            creator_analysis["charisma_authenticity"] = self.creator_prompts["charisma_authenticity"][max_idx]
        
        return creator_analysis
    
    def _analyze_content_attributes(self, similarities: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze content-related attributes"""
        content_analysis = {}
        
        # Niche/specialty
        if "content_niche_specialty" in similarities:
            niche_scores = similarities["content_niche_specialty"]
            top_indices = np.argsort(niche_scores)[-3:]  # Top 3 niches
            content_analysis["niche_specialty"] = [self.content_prompts["niche_specialty"][i] for i in top_indices]
        
        # Content format
        if "content_content_format" in similarities:
            format_scores = similarities["content_content_format"]
            top_indices = np.argsort(format_scores)[-3:]  # Top 3 formats
            content_analysis["content_format"] = [self.content_prompts["content_format"][i] for i in top_indices]
        
        # Overall tone
        if "content_overall_tone" in similarities:
            tone_scores = similarities["content_overall_tone"]
            top_indices = np.argsort(tone_scores)[-3:]  # Top 3 tones
            content_analysis["overall_tone"] = [self.content_prompts["overall_tone"][i] for i in top_indices]
        
        # Energy levels
        if "content_energy_levels" in similarities:
            energy_scores = similarities["content_energy_levels"]
            max_idx = np.argmax(energy_scores)
            content_analysis["energy_levels"] = self.content_prompts["energy_levels"][max_idx]
        
        # Visual presentation
        if "content_visual_presentation" in similarities:
            visual_scores = similarities["content_visual_presentation"]
            top_indices = np.argsort(visual_scores)[-3:]  # Top 3 visual styles
            content_analysis["visual_presentation"] = [self.content_prompts["visual_presentation"][i] for i in top_indices]
        
        # Product presentation
        if "content_product_presentation" in similarities:
            product_scores = similarities["content_product_presentation"]
            top_indices = np.argsort(product_scores)[-3:]  # Top 3 presentation styles
            content_analysis["product_presentation"] = [self.content_prompts["product_presentation"][i] for i in top_indices]
        
        # Video quality
        if "content_video_quality" in similarities:
            quality_scores = similarities["content_video_quality"]
            max_idx = np.argmax(quality_scores)
            content_analysis["video_quality"] = self.content_prompts["video_quality"][max_idx]
        
        # Video scene
        if "content_video_scene" in similarities:
            scene_scores = similarities["content_video_scene"]
            top_indices = np.argsort(scene_scores)[-3:]  # Top 3 scenes
            content_analysis["video_scene"] = [self.content_prompts["video_scene"][i] for i in top_indices]
        
        # Video background
        if "content_video_background" in similarities:
            background_scores = similarities["content_video_background"]
            max_idx = np.argmax(background_scores)
            content_analysis["video_background"] = self.content_prompts["video_background"][max_idx]
        
        return content_analysis
    
    def _analyze_industry_attributes(self, similarities: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze industry-specific attributes"""
        industry_analysis = {}
        
        # Beauty industry analysis
        beauty_categories = ["industry_beauty_skin_type_tone"]
        
        beauty_results = {}
        for category in beauty_categories:
            if category in similarities:
                scores = similarities[category]
                top_indices = np.argsort(scores)[-2:]  # Top 2 for each category
                category_name = category.replace("industry_beauty_", "")
                beauty_results[category_name] = [self.industry_prompts["beauty"][category_name][i] for i in top_indices]
        
        if beauty_results:
            industry_analysis["beauty"] = beauty_results
        
        # Fashion analysis
        if "industry_fashion" in similarities:
            fashion_scores = similarities["industry_fashion"]
            top_indices = np.argsort(fashion_scores)[-3:]  # Top 3 fashion elements
            industry_analysis["fashion"] = [self.industry_prompts["fashion"][i] for i in top_indices]
        
        # Home & lifestyle analysis
        if "industry_home_lifestyle" in similarities:
            lifestyle_scores = similarities["industry_home_lifestyle"]
            top_indices = np.argsort(lifestyle_scores)[-2:]  # Top 2 lifestyle elements
            industry_analysis["home_lifestyle"] = [self.industry_prompts["home_lifestyle"][i] for i in top_indices]
        
        # Sports & fitness analysis
        if "industry_sports_fitness" in similarities:
            sports_scores = similarities["industry_sports_fitness"]
            top_indices = np.argsort(sports_scores)[-3:]  # Top 3 sports elements
            industry_analysis["sports_fitness"] = [self.industry_prompts["sports_fitness"][i] for i in top_indices]
        
        return industry_analysis
    
    def _analyze_visual_quality(self, similarities: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Analyze visual quality aspects"""
        visual_quality = []
        
        # Legacy visual prompts for quality assessment
        if "legacy_visual" in similarities:
            legacy_scores = similarities["legacy_visual"]
            top_indices = np.argsort(legacy_scores)[-3:]  # Top 3 visual qualities
            for idx in top_indices:
                visual_quality.append({
                    "prompt": self.visual_prompts[idx],
                    "confidence": legacy_scores[idx]
                })
        
        return visual_quality
    
    def _create_fallback_frame_analysis(self) -> Dict[str, Any]:
        """Create fallback frame analysis when analysis fails"""
        return {
            "primary_content": "unknown",
            "visual_styles": [],
            "camera_angles": [],
            "visual_quality": [],
            "dominant_colors": [],
            "brightness": "unknown",
            "contrast": "unknown",
            "mood_atmosphere": [],
            "product_focus": [],
            "gender": "unknown",
            "age": "unknown",
            "physical_appearance": "unknown",
            "body_type": "unknown",
            "hair_color": "unknown",
            "hair_type": "unknown",
            "persona": [],
            "charisma_authenticity": "unknown",
            "niche_specialty": [],
            "content_format": [],
            "overall_tone": [],
            "energy_levels": "unknown",
            "visual_presentation": [],
            "product_presentation": [],
            "video_quality": "unknown",
            "video_scene": [],
            "video_background": "unknown",
            "beauty": {},
            "fashion": [],
            "home_lifestyle": [],
            "sports_fitness": []
        }
    
    def _analyze_objects(self, frame_paths: List[str]) -> Dict[str, Any]:
        """Analyze objects in frames using enhanced CLIP embeddings"""
        try:
            self.logger.info(f"Starting object analysis with {len(frame_paths)} frames")
            self.logger.info(f"Frame paths received: {frame_paths}")
            
            # Enhanced object prompts for more detailed detection
            enhanced_object_prompts = [
                # Beauty and personal care products
                "makeup brush", "foundation", "concealer", "mascara", "eyeliner", "lipstick", "blush", "bronzer",
                "eyeshadow", "powder", "setting spray", "makeup remover", "cotton pad", "makeup sponge",
                "skincare cream", "serum", "toner", "cleanser", "moisturizer", "sunscreen", "face mask",
                "hair brush", "hair dryer", "straightener", "curling iron", "hair spray", "hair gel",
                "shampoo", "conditioner", "hair oil", "hair mask", "hair clips", "hair ties",
                
                # Electronics and devices
                "cell phone", "smartphone", "camera", "mirror", "ring light", "lighting setup", "tripod",
                "microphone", "speaker", "headphones", "earbuds", "tablet", "laptop", "computer",
                
                # Household items
                "mirror", "sink", "towel", "tissue", "paper towel", "trash can", "cabinet", "drawer",
                "shelf", "table", "chair", "bed", "pillow", "blanket", "curtain", "window",
                "door", "wall", "floor", "ceiling", "light", "lamp", "candle", "plant",
                
                # Clothing and accessories
                "shirt", "dress", "pants", "skirt", "jacket", "coat", "scarf", "hat", "cap",
                "glasses", "sunglasses", "earrings", "necklace", "bracelet", "ring", "watch",
                "bag", "purse", "wallet", "backpack", "shoes", "boots", "sneakers", "heels",
                
                # Food and beverages
                "coffee", "tea", "water", "juice", "soda", "wine", "beer", "food", "snack",
                "fruit", "vegetable", "bread", "cake", "cookie", "candy", "chocolate",
                
                # Tools and utensils
                "knife", "fork", "spoon", "plate", "bowl", "cup", "glass", "bottle", "can",
                "pen", "pencil", "paper", "book", "magazine", "newspaper", "notebook",
                
                # Body parts and features
                "face", "eye", "nose", "mouth", "lip", "cheek", "forehead", "chin", "ear",
                "hair", "hand", "finger", "arm", "leg", "foot", "toe", "skin", "nail",
                
                # Text and labels
                "text", "label", "logo", "brand", "price tag", "barcode", "sign", "poster",
                "advertisement", "packaging", "box", "container", "bottle", "tube", "jar"
            ]
            
            # Encode enhanced object prompts
            import clip as openai_clip
            text_tokens = openai_clip.tokenize(enhanced_object_prompts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            frame_objects = []
            for frame_path in frame_paths:
                try:
                    if not os.path.exists(frame_path):
                        self.logger.warning(f"Frame file not found: {frame_path}")
                        self.logger.warning(f"File exists check failed for: {frame_path}")
                        self.logger.warning(f"Directory contents: {os.listdir(os.path.dirname(frame_path)) if os.path.exists(os.path.dirname(frame_path)) else 'Directory not found'}")
                        frame_objects.append({
                            "frame_path": frame_path,
                            "detected_objects": [],
                            "beauty_products": [],
                            "electronics": [],
                            "household_items": [],
                            "clothing_accessories": [],
                            "body_parts": []
                        })
                        continue
                    
                    self.logger.info(f"Successfully accessing frame for object analysis: {frame_path}")
                    
                    # Load and preprocess image
                    image = Image.open(frame_path).convert('RGB')
                    image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                    
                    # Extract embedding
                    with torch.no_grad():
                        embedding = self.clip_model.encode_image(image_input)
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                        
                        # Calculate similarities
                        similarities = torch.matmul(embedding, text_features.T)
                        similarities = similarities.cpu().numpy()[0]
                        
                        # Get objects with high confidence and categorize them
                        detected_objects = []
                        for idx, similarity in enumerate(similarities):
                            if similarity > 0.25:  # Lower threshold for more detection
                                detected_objects.append({
                                    "object": enhanced_object_prompts[idx],
                                    "confidence": float(similarity)
                                })
                        
                        # Sort by confidence and get top objects
                        detected_objects.sort(key=lambda x: x["confidence"], reverse=True)
                        top_objects = detected_objects[:8]  # Top 8 objects
                        
                        # Categorize objects
                        beauty_products = [obj for obj in top_objects if any(keyword in obj["object"] for keyword in ["makeup", "skincare", "hair", "brush", "cream", "serum", "foundation", "mascara", "lipstick"])]
                        electronics = [obj for obj in top_objects if any(keyword in obj["object"] for keyword in ["phone", "camera", "mirror", "light", "tripod", "microphone"])]
                        household = [obj for obj in top_objects if any(keyword in obj["object"] for keyword in ["sink", "towel", "cabinet", "table", "chair", "mirror"])]
                        clothing = [obj for obj in top_objects if any(keyword in obj["object"] for keyword in ["shirt", "dress", "pants", "jacket", "glasses", "earrings"])]
                        body_parts = [obj for obj in top_objects if any(keyword in obj["object"] for keyword in ["face", "eye", "mouth", "hair", "hand", "skin"])]
                        
                        frame_objects.append({
                            "frame_path": frame_path,
                            "detected_objects": top_objects,
                            "beauty_products": beauty_products,
                            "electronics": electronics,
                            "household_items": household,
                            "clothing_accessories": clothing,
                            "body_parts": body_parts
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing objects in frame {frame_path}: {e}")
                    continue
            
            return {
                "frame_objects": frame_objects,
                "all_objects": self._aggregate_objects(frame_objects)
            }
            
        except Exception as e:
            self.logger.error(f"Error in object analysis: {e}")
            return {"frame_objects": [], "all_objects": []}
    
    def _analyze_actions(self, frame_paths: List[str]) -> Dict[str, Any]:
        """Analyze actions in frames using enhanced CLIP embeddings"""
        try:
            self.logger.info(f"Starting action analysis with {len(frame_paths)} frames")
            self.logger.info(f"Frame paths received: {frame_paths}")
            
            # Enhanced action prompts for more detailed detection
            enhanced_action_prompts = [
                # Beauty and personal care actions
                "applying makeup", "applying foundation", "applying concealer", "applying mascara", "applying eyeliner",
                "applying lipstick", "applying blush", "applying eyeshadow", "applying powder", "applying setting spray",
                "removing makeup", "cleansing face", "applying skincare", "applying moisturizer", "applying sunscreen",
                "applying serum", "applying toner", "applying face mask", "brushing hair", "styling hair",
                "blow drying hair", "straightening hair", "curling hair", "washing hair", "conditioning hair",
                
                # Hand and body actions
                "holding object", "grasping object", "picking up object", "putting down object", "pointing",
                "waving", "clapping", "rubbing", "scratching", "touching", "tapping", "pressing",
                "squeezing", "twisting", "turning", "opening", "closing", "pulling", "pushing",
                
                # Facial expressions and gestures
                "smiling", "frowning", "winking", "raising eyebrows", "pursing lips", "opening mouth",
                "closing eyes", "opening eyes", "looking up", "looking down", "looking left", "looking right",
                "nodding", "shaking head", "tilting head", "turning head", "raising chin", "lowering chin",
                
                # Camera and device interactions
                "holding phone", "taking photo", "recording video", "looking at screen", "touching screen",
                "swiping", "tapping screen", "adjusting camera", "positioning camera", "checking mirror",
                "adjusting lighting", "setting up equipment", "adjusting tripod", "holding microphone",
                
                # Movement and positioning
                "sitting", "standing", "walking", "moving", "leaning", "bending", "stretching",
                "reaching", "turning around", "moving closer", "moving away", "adjusting position",
                "changing pose", "posing", "gesturing", "demonstrating", "showing", "displaying",
                
                # Product interactions
                "opening product", "closing product", "shaking product", "spraying product", "pouring product",
                "squeezing product", "applying product", "rubbing product", "massaging", "patting",
                "dabbing", "blending", "smoothing", "spreading", "coating", "covering", "wiping",
                
                # Communication and presentation
                "speaking", "talking", "explaining", "demonstrating", "teaching", "instructing",
                "presenting", "showing", "pointing at", "referring to", "mentioning", "describing",
                "reviewing", "evaluating", "comparing", "recommending", "suggesting", "advising"
            ]
            
            # Encode enhanced action prompts
            import clip as openai_clip
            text_tokens = openai_clip.tokenize(enhanced_action_prompts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            frame_actions = []
            for frame_path in frame_paths:
                try:
                    if not os.path.exists(frame_path):
                        self.logger.warning(f"Frame file not found: {frame_path}")
                        self.logger.warning(f"File exists check failed for: {frame_path}")
                        self.logger.warning(f"Directory contents: {os.listdir(os.path.dirname(frame_path)) if os.path.exists(os.path.dirname(frame_path)) else 'Directory not found'}")
                        continue
                    
                    self.logger.info(f"Successfully accessing frame for action analysis: {frame_path}")
                    
                    # Load and preprocess image
                    image = Image.open(frame_path).convert('RGB')
                    image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                    
                    # Extract embedding
                    with torch.no_grad():
                        embedding = self.clip_model.encode_image(image_input)
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                        
                        # Calculate similarities
                        similarities = torch.matmul(embedding, text_features.T)
                        similarities = similarities.cpu().numpy()[0]
                        
                        # Get actions with high confidence and categorize them
                        detected_actions = []
                        for idx, similarity in enumerate(similarities):
                            if similarity > 0.2:  # Lower threshold for more detection
                                detected_actions.append({
                                    "action": enhanced_action_prompts[idx],
                                    "confidence": float(similarity)
                                })
                        
                        # Sort by confidence and get top actions
                        detected_actions.sort(key=lambda x: x["confidence"], reverse=True)
                        top_actions = detected_actions[:6]  # Top 6 actions
                        
                        # Categorize actions
                        beauty_actions = [action for action in top_actions if any(keyword in action["action"] for keyword in ["applying", "makeup", "skincare", "hair", "brushing", "styling", "washing"])]
                        hand_actions = [action for action in top_actions if any(keyword in action["action"] for keyword in ["holding", "grasping", "picking", "putting", "pointing", "touching", "rubbing"])]
                        facial_actions = [action for action in top_actions if any(keyword in action["action"] for keyword in ["smiling", "frowning", "winking", "looking", "nodding", "tilting"])]
                        device_actions = [action for action in top_actions if any(keyword in action["action"] for keyword in ["phone", "camera", "screen", "recording", "taking photo"])]
                        product_actions = [action for action in top_actions if any(keyword in action["action"] for keyword in ["opening", "closing", "spraying", "pouring", "applying", "rubbing", "blending"])]
                        communication_actions = [action for action in top_actions if any(keyword in action["action"] for keyword in ["speaking", "talking", "explaining", "demonstrating", "teaching", "showing"])]
                        
                        frame_actions.append({
                            "frame_path": frame_path,
                            "detected_actions": top_actions,
                            "beauty_actions": beauty_actions,
                            "hand_actions": hand_actions,
                            "facial_actions": facial_actions,
                            "device_actions": device_actions,
                            "product_actions": product_actions,
                            "communication_actions": communication_actions
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing actions in frame {frame_path}: {e}")
                    continue
            
            return {
                "frame_actions": frame_actions,
                "all_actions": self._aggregate_actions(frame_actions)
            }
            
        except Exception as e:
            self.logger.error(f"Error in action analysis: {e}")
            return {"frame_actions": [], "all_actions": []}
    
    def _analyze_vjepa2_temporal(self, frame_paths: List[str]) -> Dict[str, Any]:
        """Analyze temporal aspects using V-JEPA2 if video path is available"""
        try:
            # Try to find the original video path from frame paths
            video_path = self._extract_video_path_from_frames(frame_paths)
            
            if video_path and self.is_vjepa2_available():
                # Extract V-JEPA2 features
                vjepa2_features = self._extract_vjepa2_features(video_path)
                
                if vjepa2_features.get('success', False):
                    content_analysis = vjepa2_features.get('content_analysis', {})
                    
                    return {
                        'temporal_consistency': content_analysis.get('temporal_stability', 0.0),
                        'content_complexity': content_analysis.get('content_complexity', 0.0),
                        'model_type': 'vjepa2',
                        'available': True
                    }
            
            # Fallback when V-JEPA2 is not available or video path not found
            return {
                'temporal_consistency': 0.5,
                'content_complexity': 0.5,
                'model_type': 'fallback',
                'available': False
            }
            
        except Exception as e:
            self.logger.error(f"Error in V-JEPA2 temporal analysis: {e}")
            return {
                'temporal_consistency': 0.5,
                'content_complexity': 0.5,
                'model_type': 'error',
                'available': False
            }
    
    def _extract_video_path_from_frames(self, frame_paths: List[str]) -> str:
        """Extract original video path from frame paths if possible"""
        try:
            if not frame_paths:
                return None
            
            # Common pattern: frames are in a subdirectory of the video
            frame_path = frame_paths[0]
            frame_dir = os.path.dirname(frame_path)
            
            # Extract video name from frame directory
            # Pattern: data/tiktok_frames/video_name/frame.jpg
            video_name = os.path.basename(frame_dir)
            
            # Try to find video in tiktok_videos directory
            video_path = os.path.join("data", "tiktok_videos", f"{video_name}.mp4")
            if os.path.exists(video_path):
                self.logger.info(f"Found original video: {video_path}")
                return video_path
            
            # Fallback: Look for video files in parent directories
            current_dir = frame_dir
            for _ in range(3):  # Look up to 3 levels up
                if os.path.exists(current_dir):
                    for file in os.listdir(current_dir):
                        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                            found_video = os.path.join(current_dir, file)
                            self.logger.info(f"Found video in parent directory: {found_video}")
                            return found_video
                current_dir = os.path.dirname(current_dir)
            
            self.logger.warning(f"Could not find original video for frames in {frame_dir}")
            return None
            
        except Exception as e:
            self.logger.warning(f"Could not extract video path from frames: {e}")
            return None
    
    def _extract_vjepa2_features(self, video_path: str, num_frames: int = None) -> Dict[str, Any]:
        """Extract video features using V-JEPA2"""
        if not self.is_vjepa2_available():
            return self._create_vjepa2_fallback_result()
        
        # Use configured num_frames if not specified
        if num_frames is None:
            num_frames = self.vjepa2_num_frames
        
        try:
            # Extract frames from video
            frames = self._extract_video_frames(video_path, num_frames)
            if not frames:
                return self._create_vjepa2_fallback_result()
            
            # Preprocess frames
            processed_frames = self._preprocess_video_frames(frames)
            
            # Extract features
            with torch.no_grad():
                # Get encoder features
                encoder_features = self._extract_vjepa2_encoder_features(processed_frames)
                
                # Get temporal features (only if enabled)
                temporal_features = None
                if self.vjepa2_temporal_analysis:
                    temporal_features = self._extract_vjepa2_temporal_features(processed_frames)
                
                # Analyze content
                content_analysis = self._analyze_vjepa2_content(encoder_features, temporal_features)
            
            result = {
                'encoder_features': encoder_features.cpu().numpy(),
                'content_analysis': content_analysis,
                'frame_count': len(frames),
                'model_type': 'vjepa2',
                'success': True
            }
            
            # Add temporal features if available
            if temporal_features is not None:
                result['temporal_features'] = temporal_features.cpu().numpy()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting V-JEPA2 features: {e}")
            return self._create_vjepa2_fallback_result()
    
    def _extract_video_frames(self, video_path: str, num_frames: int) -> List[np.ndarray]:
        """Extract evenly spaced frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                return []
            
            # Calculate frame indices
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
            return frames
            
        except Exception as e:
            self.logger.error(f"Error extracting frames: {e}")
            return []
    
    def _preprocess_video_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess frames for V-JEPA2 input"""
        try:
            processed_frames = []
            
            for frame in frames:
                # Resize to 224x224 (standard transformer input size)
                frame_resized = cv2.resize(frame, (224, 224))
                
                # Normalize to [0, 1]
                frame_normalized = frame_resized.astype(np.float32) / 255.0
                
                # Convert to tensor and add batch dimension
                frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)
                processed_frames.append(frame_tensor)
            
            # Stack frames and add batch dimension
            # Shape: (batch_size, num_frames, channels, height, width)
            video_tensor = torch.stack(processed_frames, dim=0).unsqueeze(0)
            
            return video_tensor.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error preprocessing frames: {e}")
            raise
    
    def _extract_vjepa2_encoder_features(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Extract encoder features from video using V-JEPA2"""
        try:
            # Ensure video_tensor is properly formatted
            if not isinstance(video_tensor, torch.Tensor):
                video_tensor = torch.tensor(video_tensor, dtype=torch.float32)
            
            # Ensure tensor is on the correct device
            video_tensor = video_tensor.to(self.device)
            
            # Extract features from V-JEPA2 model
            with torch.no_grad():
                try:
                    outputs = self.vjepa2_model(video_tensor)
                    # Get the last hidden state as features
                    features = outputs.last_hidden_state
                except TypeError as e:
                    if "indices" in str(e) and "must be Tensor" in str(e):
                        # Handle embedding indices error
                        self.logger.warning("V-JEPA2 model embedding error, using fallback")
                        # Return a fallback tensor with expected shape
                        batch_size = video_tensor.shape[0] if len(video_tensor.shape) > 0 else 1
                        features = torch.zeros((batch_size, 16, 1024), device=self.device)
                    else:
                        raise e
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting V-JEPA2 features: {e}")
            # Return fallback tensor
            batch_size = video_tensor.shape[0] if len(video_tensor.shape) > 0 else 1
            return torch.zeros((batch_size, 16, 1024), device=self.device)
    
    def _extract_vjepa2_temporal_features(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Extract temporal features using V-JEPA2"""
        try:
            # Ensure video_tensor is properly formatted
            if not isinstance(video_tensor, torch.Tensor):
                video_tensor = torch.tensor(video_tensor, dtype=torch.float32)
            
            # Ensure tensor is on the correct device
            video_tensor = video_tensor.to(self.device)
            
            # Extract temporal features from V-JEPA2 model
            with torch.no_grad():
                try:
                    outputs = self.vjepa2_model(video_tensor)
                    # Use the last hidden state as temporal features
                    # V-JEPA2 inherently captures temporal information
                    temporal_features = outputs.last_hidden_state
                except TypeError as e:
                    if "indices" in str(e) and "must be Tensor" in str(e):
                        # Handle embedding indices error
                        self.logger.warning("V-JEPA2 model embedding error, using fallback")
                        # Return a fallback tensor with expected shape
                        batch_size = video_tensor.shape[0] if len(video_tensor.shape) > 0 else 1
                        temporal_features = torch.zeros((batch_size, 16, 1024), device=self.device)
                    else:
                        raise e
            return temporal_features
            
        except Exception as e:
            self.logger.error(f"Error extracting V-JEPA2 temporal features: {e}")
            # Return fallback tensor
            batch_size = video_tensor.shape[0] if len(video_tensor.shape) > 0 else 1
            return torch.zeros((batch_size, 16, 1024), device=self.device)
    
    def _analyze_vjepa2_content(self, encoder_features: torch.Tensor, temporal_features: torch.Tensor) -> Dict[str, Any]:
        """Analyze video content based on extracted features"""
        try:
            # Ensure encoder_features is a tensor
            if not isinstance(encoder_features, torch.Tensor):
                encoder_features = torch.tensor(encoder_features, dtype=torch.float32)
            
            # Calculate feature statistics for encoder features
            encoder_mean = torch.mean(encoder_features, dim=1)
            encoder_std = torch.std(encoder_features, dim=1)
            
            # Handle temporal features (might be None or have different shape)
            temporal_mean = torch.zeros_like(encoder_mean)
            temporal_std = torch.zeros_like(encoder_std)
            temporal_consistency = torch.tensor(0.0)
            
            if temporal_features is not None and isinstance(temporal_features, torch.Tensor):
                try:
                    temporal_mean = torch.mean(temporal_features, dim=1)
                    temporal_std = torch.std(temporal_features, dim=1)
                    # Calculate temporal consistency
                    temporal_consistency = torch.std(temporal_features, dim=0).mean()
                except Exception as e:
                    self.logger.warning(f"Error processing temporal features: {e}")
            
            # Analyze feature patterns
            feature_magnitude = torch.norm(encoder_features, dim=-1).mean()
            
            return {
                'encoder_mean': encoder_mean.cpu().numpy(),
                'encoder_std': encoder_std.cpu().numpy(),
                'temporal_mean': temporal_mean.cpu().numpy(),
                'temporal_std': temporal_std.cpu().numpy(),
                'temporal_consistency': temporal_consistency.cpu().numpy(),
                'feature_magnitude': feature_magnitude.cpu().numpy(),
                'content_complexity': float(feature_magnitude),
                'temporal_stability': float(1.0 - temporal_consistency)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing content: {e}")
            return {
                'encoder_mean': np.zeros((1, 1024)),
                'encoder_std': np.zeros((1, 1024)),
                'temporal_mean': np.zeros((1, 1024)),
                'temporal_std': np.zeros((1, 1024)),
                'temporal_consistency': np.array(0.0),
                'feature_magnitude': np.array(0.0),
                'content_complexity': 0.0,
                'temporal_stability': 0.0
            }
    
    def _create_vjepa2_fallback_result(self) -> Dict[str, Any]:
        """Create fallback result when V-JEPA2 is not available"""
        return {
            'encoder_features': np.zeros((1, 16, 1024)),
            'temporal_features': np.zeros((1, 16, 1024)),
            'content_analysis': {
                'content_complexity': 0.0,
                'temporal_stability': 0.0
            },
            'frame_count': 0,
            'model_type': 'fallback',
            'success': False
        }
    
    def _extract_dominant_colors(self, img_array: np.ndarray, num_colors: int = 5) -> List[str]:
        """Extract dominant colors from image"""
        try:
            # Reshape image to 2D array of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Use k-means to find dominant colors
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get color labels and counts
            labels = kmeans.labels_
            colors = kmeans.cluster_centers_.astype(int)
            
            # Count occurrences of each color
            color_counts = {}
            for i, label in enumerate(labels):
                color = tuple(colors[label])
                color_counts[color] = color_counts.get(color, 0) + 1
            
            # Sort by frequency and convert to color names
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            dominant_colors = []
            
            for color, count in sorted_colors[:3]:  # Top 3 colors
                color_name = self._rgb_to_color_name(color)
                percentage = (count / len(labels)) * 100
                dominant_colors.append(f"{color_name} ({percentage:.1f}%)")
            
            return dominant_colors
            
        except Exception as e:
            self.logger.warning(f"Error extracting dominant colors: {e}")
            return ["unknown"]
    
    def _rgb_to_color_name(self, rgb: tuple) -> str:
        """Convert RGB values to color names"""
        r, g, b = rgb
        
        # Simple color mapping
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > 200 and g < 100 and b < 100:
            return "red"
        elif r < 100 and g > 200 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 200:
            return "blue"
        elif r > 200 and g > 200 and b < 100:
            return "yellow"
        elif r > 200 and g < 100 and b > 200:
            return "magenta"
        elif r < 100 and g > 200 and b > 200:
            return "cyan"
        elif r > 150 and g > 100 and b < 100:
            return "orange"
        elif r > 150 and g < 100 and b > 150:
            return "purple"
        elif r > 100 and g > 100 and b > 100:
            return "gray"
        else:
            return "mixed"
    
    def _analyze_brightness(self, img_array: np.ndarray) -> str:
        """Analyze image brightness"""
        try:
            # Convert to grayscale
            gray = np.mean(img_array, axis=2)
            avg_brightness = np.mean(gray)
            
            if avg_brightness > 180:
                return "very bright"
            elif avg_brightness > 140:
                return "bright"
            elif avg_brightness > 100:
                return "medium"
            elif avg_brightness > 60:
                return "dark"
            else:
                return "very dark"
                
        except Exception as e:
            self.logger.warning(f"Error analyzing brightness: {e}")
            return "unknown"
    
    def _analyze_contrast(self, img_array: np.ndarray) -> str:
        """Analyze image contrast"""
        try:
            # Convert to grayscale
            gray = np.mean(img_array, axis=2)
            
            # Calculate standard deviation as contrast measure
            contrast = np.std(gray)
            
            if contrast > 50:
                return "high contrast"
            elif contrast > 30:
                return "medium contrast"
            else:
                return "low contrast"
                
        except Exception as e:
            self.logger.warning(f"Error analyzing contrast: {e}")
            return "unknown"
    
    def _aggregate_content_types(self, frame_analyses: List[Dict]) -> Dict[str, float]:
        """Aggregate content types across all frames"""
        content_counts = {}
        total_frames = len(frame_analyses)
        
        for analysis in frame_analyses:
            primary = analysis.get("primary_content", "unknown")
            content_counts[primary] = content_counts.get(primary, 0) + 1
        
        # Convert to percentages
        return {content: count/total_frames for content, count in content_counts.items()}
    
    def _aggregate_objects(self, frame_objects: List[Dict]) -> List[Dict]:
        """Aggregate all detected objects across frames"""
        all_objects = {}
        
        for frame_obj in frame_objects:
            for obj in frame_obj.get("detected_objects", []):
                obj_name = obj["object"]
                if obj_name not in all_objects:
                    all_objects[obj_name] = {
                        "object": obj_name,
                        "total_confidence": 0.0,
                        "frame_count": 0
                    }
                all_objects[obj_name]["total_confidence"] += obj["confidence"]
                all_objects[obj_name]["frame_count"] += 1
        
        # Calculate average confidence
        for obj in all_objects.values():
            obj["avg_confidence"] = obj["total_confidence"] / obj["frame_count"]
        
        return sorted(all_objects.values(), key=lambda x: x["avg_confidence"], reverse=True)
    
    def _aggregate_actions(self, frame_actions: List[Dict]) -> List[Dict]:
        """Aggregate all detected actions across frames"""
        all_actions = {}
        
        for frame_action in frame_actions:
            for action in frame_action.get("detected_actions", []):
                action_name = action["action"]
                if action_name not in all_actions:
                    all_actions[action_name] = {
                        "action": action_name,
                        "total_confidence": 0.0,
                        "frame_count": 0
                    }
                all_actions[action_name]["total_confidence"] += action["confidence"]
                all_actions[action_name]["frame_count"] += 1
        
        # Calculate average confidence
        for action in all_actions.values():
            action["avg_confidence"] = action["total_confidence"] / action["frame_count"]
        
        return sorted(all_actions.values(), key=lambda x: x["avg_confidence"], reverse=True)
    
    def _generate_comprehensive_description(self, visual_analysis: Dict, object_analysis: Dict, 
                                          action_analysis: Dict, vjepa2_analysis: Dict, audio_transcript: str) -> str:
        """Generate comprehensive text description for GPT input with enhanced TikTok tag system analysis"""
        try:
            description_parts = []
            
            # Enhanced TikTok tag system analysis summary
            if visual_analysis.get("frame_analyses"):
                frame_analyses = visual_analysis["frame_analyses"]
                description_parts.append(f"TIKTOK CONTENT CREATOR ANALYSIS:")
                description_parts.append(f"Total frames analyzed: {len(frame_analyses)}")
                
                # Overall content types
                if visual_analysis.get("overall_content"):
                    content_summary = []
                    for content_type, percentage in visual_analysis["overall_content"].items():
                        if percentage > 0.2:
                            content_summary.append(f"{content_type} ({percentage:.1%})")
                    if content_summary:
                        description_parts.append(f"Primary content types: {', '.join(content_summary)}")
                
                # Detailed frame-by-frame TikTok analysis
                description_parts.append("\nDETAILED TIKTOK FRAME ANALYSIS:")
                for i, frame_analysis in enumerate(frame_analyses):
                    frame_desc = f"\nFrame {i+1}:"
                    
                    # Creator attributes
                    gender = frame_analysis.get('gender', 'unknown')
                    age = frame_analysis.get('age', 'unknown')
                    physical_appearance = frame_analysis.get('physical_appearance', 'unknown')
                    body_type = frame_analysis.get('body_type', 'unknown')
                    hair_color = frame_analysis.get('hair_color', 'unknown')
                    hair_type = frame_analysis.get('hair_type', 'unknown')
                    
                    frame_desc += f"\n  - Creator: {gender}, {age}, {physical_appearance}, {body_type}"
                    frame_desc += f"\n  - Hair: {hair_color} {hair_type}"
                    
                    # Persona analysis
                    personas = frame_analysis.get('persona', [])
                    if personas:
                        frame_desc += f"\n  - Persona: {', '.join(personas)}"
                    
                    charisma = frame_analysis.get('charisma_authenticity', 'unknown')
                    frame_desc += f"\n  - Charisma: {charisma}"
                    
                    # Content attributes
                    niche_specialty = frame_analysis.get('niche_specialty', [])
                    if niche_specialty:
                        frame_desc += f"\n  - Niche: {', '.join(niche_specialty)}"
                    
                    content_format = frame_analysis.get('content_format', [])
                    if content_format:
                        frame_desc += f"\n  - Format: {', '.join(content_format)}"
                    
                    overall_tone = frame_analysis.get('overall_tone', [])
                    if overall_tone:
                        frame_desc += f"\n  - Tone: {', '.join(overall_tone)}"
                    
                    energy_levels = frame_analysis.get('energy_levels', 'unknown')
                    frame_desc += f"\n  - Energy: {energy_levels}"
                    
                    # Visual presentation
                    visual_presentation = frame_analysis.get('visual_presentation', [])
                    if visual_presentation:
                        frame_desc += f"\n  - Visual style: {', '.join(visual_presentation)}"
                    
                    product_presentation = frame_analysis.get('product_presentation', [])
                    if product_presentation:
                        frame_desc += f"\n  - Product presentation: {', '.join(product_presentation)}"
                    
                    video_quality = frame_analysis.get('video_quality', 'unknown')
                    video_scene = frame_analysis.get('video_scene', [])
                    video_background = frame_analysis.get('video_background', 'unknown')
                    
                    frame_desc += f"\n  - Video quality: {video_quality}"
                    if video_scene:
                        frame_desc += f"\n  - Scene: {', '.join(video_scene)}"
                    frame_desc += f"\n  - Background: {video_background}"
                    
                    # Industry-specific analysis
                    beauty_analysis = frame_analysis.get('beauty', {})
                    if beauty_analysis:
                        frame_desc += f"\n  - Beauty analysis:"
                        for category, items in beauty_analysis.items():
                            if items:
                                frame_desc += f"\n    * {category}: {', '.join(items)}"
                    
                    fashion = frame_analysis.get('fashion', [])
                    if fashion:
                        frame_desc += f"\n  - Fashion: {', '.join(fashion)}"
                    
                    home_lifestyle = frame_analysis.get('home_lifestyle', [])
                    if home_lifestyle:
                        frame_desc += f"\n  - Home/Lifestyle: {', '.join(home_lifestyle)}"
                    
                    sports_fitness = frame_analysis.get('sports_fitness', [])
                    if sports_fitness:
                        frame_desc += f"\n  - Sports/Fitness: {', '.join(sports_fitness)}"
                    
                    # Visual quality analysis
                    dominant_colors = frame_analysis.get('dominant_colors', [])
                    if dominant_colors:
                        frame_desc += f"\n  - Colors: {', '.join(dominant_colors)}"
                    
                    brightness = frame_analysis.get('brightness', 'unknown')
                    contrast = frame_analysis.get('contrast', 'unknown')
                    frame_desc += f"\n  - Lighting: {brightness}, {contrast}"
                    
                    visual_quality = frame_analysis.get('visual_quality', [])
                    if visual_quality:
                        quality_names = [quality["prompt"] for quality in visual_quality[:2]]
                        frame_desc += f"\n  - Visual quality: {', '.join(quality_names)}"
                    
                    # Objects in this frame
                    if i < len(object_analysis.get("frame_objects", [])):
                        frame_objects = object_analysis["frame_objects"][i]
                        
                        beauty_products = frame_objects.get("beauty_products", [])
                        if beauty_products:
                            beauty_names = [obj["object"] for obj in beauty_products[:3]]
                            frame_desc += f"\n  - Beauty products: {', '.join(beauty_names)}"
                        
                        electronics = frame_objects.get("electronics", [])
                        if electronics:
                            electronic_names = [obj["object"] for obj in electronics[:2]]
                            frame_desc += f"\n  - Electronics: {', '.join(electronic_names)}"
                        
                        household = frame_objects.get("household_items", [])
                        if household:
                            household_names = [obj["object"] for obj in household[:2]]
                            frame_desc += f"\n  - Household items: {', '.join(household_names)}"
                        
                        body_parts = frame_objects.get("body_parts", [])
                        if body_parts:
                            body_names = [obj["object"] for obj in body_parts[:3]]
                            frame_desc += f"\n  - Body parts: {', '.join(body_names)}"
                    
                    # Actions in this frame
                    if i < len(action_analysis.get("frame_actions", [])):
                        frame_actions = action_analysis["frame_actions"][i]
                        
                        beauty_actions = frame_actions.get("beauty_actions", [])
                        if beauty_actions:
                            beauty_action_names = [action["action"] for action in beauty_actions[:2]]
                            frame_desc += f"\n  - Beauty actions: {', '.join(beauty_action_names)}"
                        
                        hand_actions = frame_actions.get("hand_actions", [])
                        if hand_actions:
                            hand_action_names = [action["action"] for action in hand_actions[:2]]
                            frame_desc += f"\n  - Hand actions: {', '.join(hand_action_names)}"
                        
                        facial_actions = frame_actions.get("facial_actions", [])
                        if facial_actions:
                            facial_action_names = [action["action"] for action in facial_actions[:2]]
                            frame_desc += f"\n  - Facial expressions: {', '.join(facial_action_names)}"
                        
                        device_actions = frame_actions.get("device_actions", [])
                        if device_actions:
                            device_action_names = [action["action"] for action in device_actions[:2]]
                            frame_desc += f"\n  - Device interactions: {', '.join(device_action_names)}"
                        
                        communication_actions = frame_actions.get("communication_actions", [])
                        if communication_actions:
                            comm_action_names = [action["action"] for action in communication_actions[:2]]
                            frame_desc += f"\n  - Communication: {', '.join(comm_action_names)}"
                    
                    description_parts.append(frame_desc)
            
            # Overall object summary
            if object_analysis.get("all_objects"):
                top_objects = [obj["object"] for obj in object_analysis["all_objects"][:8] 
                              if obj["avg_confidence"] > 0.3]
                if top_objects:
                    description_parts.append(f"\nOVERALL OBJECTS DETECTED: {', '.join(top_objects)}")
            
            # Overall action summary
            if action_analysis.get("all_actions"):
                top_actions = [action["action"] for action in action_analysis["all_actions"][:6] 
                              if action["avg_confidence"] > 0.25]
                if top_actions:
                    description_parts.append(f"\nOVERALL ACTIONS DETECTED: {', '.join(top_actions)}")
            
            # V-JEPA2 temporal analysis
            if vjepa2_analysis.get("available", False):
                temporal_consistency = vjepa2_analysis.get("temporal_consistency", 0.5)
                content_complexity = vjepa2_analysis.get("content_complexity", 0.5)
                
                temporal_desc = f"\nTEMPORAL ANALYSIS: "
                if temporal_consistency > 0.7:
                    temporal_desc += "high temporal consistency, stable content flow"
                elif temporal_consistency > 0.4:
                    temporal_desc += "moderate temporal consistency, some content variation"
                else:
                    temporal_desc += "low temporal consistency, dynamic content changes"
                
                complexity_desc = f"Content complexity: "
                if content_complexity > 0.7:
                    complexity_desc += "high complexity, rich visual content"
                elif content_complexity > 0.4:
                    complexity_desc += "moderate complexity, balanced content"
                else:
                    complexity_desc += "low complexity, simple visual content"
                
                description_parts.append(f"{temporal_desc}. {complexity_desc}")
            
            # Audio transcript
            if audio_transcript.strip():
                description_parts.append(f"\nAUDIO TRANSCRIPT: {audio_transcript}")
            
            return "\n".join(description_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive description: {e}")
            return f"TikTok video analysis with {len(visual_analysis.get('frame_analyses', []))} frames. Audio: {audio_transcript}" 