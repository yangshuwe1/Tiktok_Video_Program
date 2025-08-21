import yaml
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """Configuration manager for TikTok video feature extraction"""
    
    def __init__(self, config_path: str = "config/config.yml"):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            config = self._override_with_env(config)
            
            return config
            
        except Exception as e:
            from logger import get_logger
            logger = get_logger("config_manager")
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration values with environment variables"""
        env_mappings = {
            'BATCH_SIZE': ('processing', 'batch_size'),
            'SAMPLE_RATE': ('features', 'audio', 'sample_rate'),
            'WHISPER_MODEL_SIZE': ('models', 'whisper', 'model_size'),
            'LOG_LEVEL': ('logging', 'level'),
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                key = config_path[-1]
                if isinstance(current.get(key), int):
                    current[key] = int(env_value)
                elif isinstance(current.get(key), float):
                    current[key] = float(env_value)
                elif isinstance(current.get(key), list):
                    current[key] = [int(x.strip()) for x in env_value.split(',')]
                else:
                    current[key] = env_value
        
        return config
    
    def _validate_config(self):
        """Validate configuration structure and values"""
        required_sections = ['processing', 'features', 'models', 'output', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        if self.config['processing']['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config['logging']
        log_dir = os.path.dirname(log_config['log_file'])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            'processing': {
                'batch_size': 10,
                'enable_progress_bar': True,
                'save_intermediate_results': False
            },
            'features': {
                'audio': {
                    'enabled': True,
                    'sample_rate': 16000,
                    'chunk_duration': 30.0,
                    'overlap_duration': 2.0,
                    'vad_threshold': 0.2,
                    'noise_reduce_strength': 0.1,
                    'extract_vocals': True,
                    'transcribe_speech': True,
                    'detect_music': True,
                    'detect_events': True
                },
                'visual': {
                    'enabled': True,
                    'extract_keyframes': True,
                    'extract_highlights': True,
                    'frame_rate': 1,
                    'max_frames': 50
                },
                'multimodal': {
                    'enabled': True,
                    'use_yolo': True,
                    'use_blip': True,
                    'use_clip': True,
                    'use_gpt4o': False,  # Disabled GPT4O to avoid additional costs
                    'use_embedding_analyzer': True,  # Enable CLIP+V-JEPA embedding analysis
                    'representative_frames': 5
                }
            },
            'models': {
                'yolo': {
                    'model_path': 'models/yolov8s-world.pt',
                    'confidence_threshold': 0.5,
                    'fallback_model': 'models/yolov8x.pt'
                },
                'whisper': {
                    'model_size': 'large-v3',
                    'language': None,
                    'task': 'transcribe'
                },
                'demucs': {
                    'model_name': 'htdemucs'
                },
                'blip': {
                    'model_name': 'Salesforce/blip-image-captioning-base'
                },
                'clip': {
                    'model_name': 'ViT-B/32'
                },
                'embedding_analyzer': {
                    'enabled': True,
                    'use_clip': True,
                    'use_vjepa2': True,
                    'text_only_mode': True,  # 启用时关闭图片输入，只使用文字prompt
                    'vjepa2': {
                        'model_type': 'vitg16',
                        'num_frames': 16,
                        'temporal_analysis': True,
                        'content_complexity': True,
                        'video_embeddings': True
                    },
                    'embedding_analysis': {
                        'use_clip': True,
                        'use_vjepa2': True,
                        'visual_prompts': True,
                        'object_analysis': True,
                        'action_analysis': True,
                        'temporal_analysis': True
                    }
                },
                'gpt4o': {
                    'model_name': 'gpt-4o-mini',  # Changed from gpt-4o to gpt-4o-mini
                    'max_tokens': 512,
                    'temperature': 0.1,
                    'input_price_per_1k': 0.0006,  # Updated pricing for gpt-4o-mini
                    'output_price_per_1k': 0.0024,  # Updated pricing for gpt-4o-mini
                    'prompts': {
                        'comprehensive_analysis': 'Analyze this TikTok video frame and provide a detailed description of what you see. Focus on products, people, actions, and any text visible in the image. Return a JSON response with the following structure: {"description": "detailed visual description", "primary_category": "main product category", "secondary_category": "sub-category", "tertiary_category": "specific product type", "confidence": 0.95}',
                        'product_detection': 'Identify any products, brands, or commercial items visible in this image. Focus on items that could be sold or promoted. Return a JSON response with: {"products": ["product1", "product2"], "brands": ["brand1", "brand2"], "commercial_intent": true/false, "confidence": 0.95}',
                        'content_analysis': 'Analyze the content of this TikTok frame for commercial intent, product placement, and promotional elements. Return a JSON response with: {"commercial_content": true/false, "product_placement": true/false, "promotional_elements": ["element1", "element2"], "target_audience": "audience description", "confidence": 0.95}',
                        'batch_analysis': """You are a professional TikTok video content analysis expert. 
Please analyze the following video keyframes and provide comprehensive content analysis.

Please carefully observe all frames and analyze based on the following information:
1. Visual content of each frame
2. Detected objects and products
3. Audio transcription content (if available)
4. Coherence and storytelling between frames

Please return a JSON format response with the following structure:
{{
    "video_description": "comprehensive description of the entire video",
    "primary_category": "main product category",
    "secondary_category": "secondary product category", 
    "tertiary_category": "specific product type",
    "commercial_intent": "commercial/non-commercial/educational",
    "product_placement": "explicit/subtle/none",
    "key_products": ["product1", "product2", ...],
    "brands_mentioned": ["brand1", "brand2", ...],
    "content_creative_ideas": "content creative approach (Before/After, Tutorial, Review, etc.)",
    "emotional_value": "emotional appeal (confidence, beauty, empowerment, etc.)",
    "selling_points": ["key selling points like long-wear, oil-control, time-saving, eco-friendly, etc."],
    "user_demographics": "specific target user group (age, gender, lifestyle, etc. )",
    "user_preferences": "specific target user preferences (color, size, etc. )",
    "user_pain_points": "specific target user pain points (skin type, skin concerns, etc. )",
    "user_goals": "specific target user goals (skin care, makeup, etc. )",
    "compliance_issues": ["issue1", "issue2", ...],
    "engagement_potential": "high/medium/low",
    "trend_alignment": ["current_trend1", "current_trend2", ...],
    "optimization_suggestions": {{
        "content": ["suggestion1", "suggestion2", ...],
        "product_placement": ["suggestion1", "suggestion2", ...],
        "content_creative_ideas": ["suggestion1", "suggestion2", ...],
        "emotional_value": ["suggestion1", "suggestion2", ...],
        "compliance_issues": ["suggestion1", "suggestion2", ...],
    }}
}}

Analysis requirements:
- Focus on commercial intent and product showcase
- Identify main product categories and brands
- Analyze target audience demographics and preferences
- Evaluate commercial value of content
- Identify content creative approach (Before/After, Tutorial, etc.)
- Analyze emotional value and appeal
- Extract key selling points and product benefits
- Provide brief analysis for each frame
- Compliance considers 25 items total, such as:
    - Auctions: Flags content promoting bidding or auction-based sales models.
    - Exaggerated promises: Detects unrealistic claims such as "instant results" or "guaranteed outcomes."
    - False or misleading info: Identifies fake reviews, unverified claims, or inconsistent product messaging.
    - Gambling & gamification: Flags lottery, prize wheels, or game-like mechanics that simulate gambling.
    - Illegal or criminal activity: Detects promotion of counterfeits, unlicensed products, or criminal behavior.
- Analyze user journey and conversion funnel
- Identify content effectiveness indicators
- Evaluate brand safety and compliance risks
- Assess competitive positioning
Audio transcription content: {audio_transcript}

Frames:""",
                        'enhanced_analysis': """You are a professional TikTok video content analysis expert. 
Please analyze the following video content based on the detailed embedding analysis results provided below.

DETAILED EMBEDDING ANALYSIS RESULTS:
{embedding_prompt}

VIDEO FRAME INFORMATION:
- Total representative frames analyzed: {frame_count}
- Frame sequence: {frame_sequence}
- Analysis method: CLIP + V-JEPA2 embedding analysis (no direct image input)

AUDIO TRANSCRIPT:
{audio_transcript}

ANALYSIS REQUIREMENTS:
Based on the embedding analysis results above, please provide a comprehensive analysis focusing on:

1. VISUAL CONTENT ANALYSIS:
   - Primary visual content types and their distribution
   - Detected objects and their significance
   - Actions and activities shown in the video
   - Visual style and presentation approach

2. COMMERCIAL ANALYSIS:
   - Commercial intent and product placement strategies
   - Key products and brands identified
   - Target audience demographics and preferences
   - Selling points and value propositions

3. CONTENT STRATEGY:
   - Content creative approach (Before/After, Tutorial, Review, etc.)
   - Emotional appeal and engagement factors
   - User journey and conversion funnel analysis
   - Compliance considerations and risk assessment

4. PERFORMANCE INDICATORS:
   - Engagement potential assessment
   - Trend alignment and market positioning
   - Optimization opportunities and suggestions

Please return a JSON response with the following structure:
{{
    "description": "comprehensive video description based on embedding analysis",
    "primary_category": "main product category",
    "secondary_category": "sub-category", 
    "tertiary_category": "specific product type",
    "commercial_intent": true/false,
    "product_placement": "explicit/subtle/none",
    "target_audience": "detailed audience description",
    "content_style": "content style and approach",
    "key_products": ["product1", "product2", ...],
    "brands_mentioned": ["brand1", "brand2", ...],
    "content_creative_ideas": "creative approach description",
    "emotional_value": "emotional appeal description",
    "selling_points": ["point1", "point2", ...],
    "user_demographics": "demographic details",
    "user_preferences": "preference details",
    "user_pain_points": "pain point details",
    "user_goals": "goal details",
    "compliance_issues": ["issue1", "issue2", ...],
    "engagement_potential": "high/medium/low",
    "trend_alignment": ["trend1", "trend2", ...],
    "optimization_suggestions": {{
        "content": ["suggestion1", "suggestion2", ...],
        "product_placement": ["suggestion1", "suggestion2", ...],
        "content_creative_ideas": ["suggestion1", "suggestion2", ...],
        "emotional_value": ["suggestion1", "suggestion2", ...],
        "compliance_issues": ["suggestion1", "suggestion2", ...]
    }}
}}

IMPORTANT: This analysis is based on CLIP and V-JEPA2 embedding analysis results, not direct image input. 
The embedding analysis provides detailed visual understanding including objects, actions, content types, and temporal patterns.""",
                        'comprehensive_analysis': """You are a professional TikTok video content analysis expert with comprehensive knowledge of TikTok's content creator ecosystem and commercial analysis.

Please analyze the following video content based on the detailed embedding analysis results provided below.

DETAILED EMBEDDING ANALYSIS RESULTS:
{comprehensive_prompt}

EMBEDDING TAGS SUMMARY:
{embedding_tags_summary}

STRUCTURED ANALYSIS:
{structured_analysis_summary}

VIDEO FRAME INFORMATION:
- Total representative frames analyzed: {frame_count}
- Frame sequence: {frame_sequence}
- Analysis method: CLIP + V-JEPA2 embedding analysis + comprehensive tag system

AUDIO TRANSCRIPT:
{audio_transcript}

ANALYSIS REQUIREMENTS:
Based on the comprehensive embedding analysis results above, please provide a detailed analysis covering all aspects of TikTok content creation and commercial potential.

Please return a JSON response with the following structure:
{{
    "video_description": "comprehensive description of the entire video based on embedding analysis",
    "primary_category": "main product category (beauty, fashion, lifestyle, etc.)",
    "secondary_category": "sub-category within primary category", 
    "tertiary_category": "specific product type or niche",
    "commercial_intent": "commercial/non-commercial/educational",
    "product_placement": "explicit/subtle/none",
    "target_audience": "detailed audience description based on creator demographics",
    "content_style": "content style and approach based on visual analysis",
    "key_products": ["product1", "product2", ...],
    "brands_mentioned": ["brand1", "brand2", ...],
    "content_creative_ideas": "creative approach description (Before/After, Tutorial, Review, etc.)",
    "emotional_value": "emotional appeal description based on content analysis",
    "selling_points": ["key selling points like long-wear, oil-control, time-saving, eco-friendly, etc."],
    "user_demographics": "specific target user group (age, gender, lifestyle, etc.)",
    "user_preferences": "specific target user preferences (color, size, etc.)",
    "user_pain_points": "specific target user pain points (skin type, skin concerns, etc.)",
    "user_goals": "specific target user goals (skin care, makeup, etc.)",
    "compliance_issues": ["issue1", "issue2", ...],
    "engagement_potential": "high/medium/low",
    "trend_alignment": ["current_trend1", "current_trend2", ...],
    "optimization_suggestions": {{
        "content": ["suggestion1", "suggestion2", ...],
        "product_placement": ["suggestion1", "suggestion2", ...],
        "content_creative_ideas": ["suggestion1", "suggestion2", ...],
        "emotional_value": ["suggestion1", "suggestion2", ...],
        "compliance_issues": ["suggestion1", "suggestion2", ...]
    }}
}}

IMPORTANT GUIDELINES:
1. Use the embedding tags and structured analysis to inform all aspects of your response
2. Ensure all fields are populated based on the available analysis data
3. Be specific and detailed in your descriptions
4. Consider the creator demographics, content characteristics, and commercial aspects
5. Provide actionable optimization suggestions
6. Consider compliance with TikTok's content policies
7. Base your analysis on the comprehensive embedding analysis, not assumptions

The embedding analysis provides detailed insights into:
- Creator attributes (gender, age, personas, charisma)
- Content characteristics (niche, format, tone, energy)
- Industry-specific analysis (beauty, fashion, lifestyle)
- Object detection and action analysis
- Temporal consistency and content complexity
- Visual quality and presentation style"""
                    }
                }
            },
            'output': {
                'save_frames': False,
                'save_audio': False,
                'output_format': ['csv', 'json'],
                'output_dir': 'data/results',
                'create_video_subdirs': True
            },
            'logging': {
                'level': 'INFO',
                'log_file': 'logs/processing.log',
                'enable_console_output': True
            }
        }
    
    def get(self, *keys, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            *keys: Configuration keys (e.g., 'processing', 'batch_size')
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return self.config['processing']
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration"""
        return self.config['features']['audio']
    
    def get_visual_config(self) -> Dict[str, Any]:
        """Get visual processing configuration"""
        return self.config['features']['visual']
    
    def get_multimodal_config(self) -> Dict[str, Any]:
        """Get multimodal configuration section"""
        return self.config.get('features', {}).get('multimodal', {})
    
    def is_batch_image_upload_enabled(self) -> bool:
        """Check if batch image upload to ChatGPT is enabled"""
        multimodal_config = self.get_multimodal_config()
        return multimodal_config.get('enable_batch_image_upload', True)
    
    def is_embedding_analysis_enabled(self) -> bool:
        """Check if embedding analysis integration in prompts is enabled"""
        multimodal_config = self.get_multimodal_config()
        return multimodal_config.get('enable_embedding_analysis', True)
    
    def get_embedding_analyzer_config(self) -> Dict[str, Any]:
        """Get embedding analyzer configuration"""
        return self.config['models'].get('embedding_analyzer', {})
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get specific model configuration"""
        return self.config['models'].get(model_name, {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.config['output']
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config['logging']
    
    def get_tiktok_config(self) -> Dict[str, Any]:
        """Get TikTok-specific configuration"""
        # Try to get from main config first
        if 'tiktok' in self.config:
            config = self.config['tiktok'].copy()
            
            # If ms_token is not set in config or is placeholder, try environment variable
            if not config.get('ms_token') or config['ms_token'] == "YOUR_MS_TOKEN_HERE":
                env_token = os.environ.get('TIKTOK_MS_TOKEN')
                if env_token:
                    config['ms_token'] = env_token
            
            return config
        
        # Fallback: try to load from separate tiktok_config.yml
        tiktok_config_path = "config/tiktok_config.yml"
        if os.path.exists(tiktok_config_path):
            try:
                with open(tiktok_config_path, 'r', encoding='utf-8') as f:
                    import yaml
                    tiktok_config = yaml.safe_load(f)
                    config = tiktok_config.get('tiktok', {})
                    
                    # If ms_token is not set in config or is placeholder, try environment variable
                    if not config.get('ms_token') or config['ms_token'] == "YOUR_MS_TOKEN_HERE":
                        env_token = os.environ.get('TIKTOK_MS_TOKEN')
                        if env_token:
                            config['ms_token'] = env_token
                    
                    return config
            except Exception as e:
                from logger import get_logger
                logger = get_logger("config_manager")
                logger.warning(f"Error loading TikTok config: {e}")
        
        # Return default TikTok configuration
        return {
            'ms_token': os.environ.get('TIKTOK_MS_TOKEN', ''),
            'browser': os.environ.get('TIKTOK_BROWSER', 'chromium'),
            'download_dir': 'data/tiktok_videos',
            'max_videos_per_creator': 30,
            'enable_video_download': False,
            'rate_limit_delay': 2,
            'max_retries': 3
        }
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return self.config['features'].get(feature_name, {}).get('enabled', False)
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of configuration updates
        """
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
        self._validate_config()
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to file
        
        Args:
            output_path: Output path (defaults to original config path)
        """
        if output_path is None:
            output_path = self.config_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def print_config(self):
        """Print current configuration"""
        from logger import get_logger
        logger = get_logger("config_manager")
        logger.info("Current Configuration:")
        logger.info(yaml.dump(self.config, default_flow_style=False, indent=2))

# Global configuration instance
config_manager = None

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager 