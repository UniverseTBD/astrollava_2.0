"""
    # Single mode: process a single galaxy by its ID
    python caption_galaxies.py --index 0 -s validation --save-png

    # Conversation avec AstroPetey
    python caption_galaxies.py --index 0 -s validation --conversation

    # Specify the split
    python caption_galaxies.py --split train
"""

# envs
import os
import argparse

import json
import io
import asyncio
from datasets import load_dataset
from google import genai
from PIL import Image
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

#Key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "key value")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)
# Model
GEMINI_MODEL = "gemini-2.5-flash-lite" # for tests
# Dataset 
DATASET_ID = "Smith42/galaxies"
DATASET_REVISION = "v2.0"

#  batch parames
DEFAULT_BATCH_SIZE = 256  # Batch size for checkpoints
DEFAULT_MAX_CONCURRENT = 50  # Max concurrent requests (set from API rate limits)
MAX_RETRY_ATTEMPTS = 1  # Nb of retry attempts on error
RETRY_BASE_DELAY = 1.0  # Base delay for exp backoff (s)
RETRY_MAX_DELAY = 60.0  #Max delay between retries



# GALAXY METADATA
METADATA_CATEGORIES = {
    "Basic Information": [
        'dr8_id', 'iauname', 'brickid', 'objid',
        'ra', 'dec', 'photoz_id', 'ra_photoz', 'dec_photoz'
    ],
    
    "Redshift Information": [
        'redshift', 'photo_z', 'photo_zerr', 'spec_z', 'redshift_nsa', 'redshift_ossy'
    ],
    
    "Size and Shape Measurements": [
        'est_petro_th50', 'est_petro_th50_kpc', 'petro_theta', 'petro_th50', 'petro_th90',
        'petro_phi50', 'petro_phi90', 'petro_ba50', 'petro_ba90', 'elpetro_ba', 'elpetro_phi',
        'elpetro_theta_r', 'sersic_n', 'sersic_ba', 'sersic_phi'
    ],
    
    "DESI Magnitudes": [
        'mag_g_desi', 'mag_r_desi', 'mag_z_desi'
    ],
    
    "Legacy Survey Magnitudes": [
        'mag_f', 'mag_n', 'mag_u', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'u_minus_r'
    ],
    
    "Absolute Magnitudes": [
        'mag_abs_g_photoz', 'mag_abs_r_photoz', 'mag_abs_z_photoz',
        'elpetro_absmag_f', 'elpetro_absmag_n', 'elpetro_absmag_u', 'elpetro_absmag_g',
        'elpetro_absmag_r', 'elpetro_absmag_i', 'elpetro_absmag_z'
    ],
    
    "Sersic Profile Measurements": [
        'sersic_nmgy_f', 'sersic_nmgy_n', 'sersic_nmgy_u', 'sersic_nmgy_g',
        'sersic_nmgy_r', 'sersic_nmgy_i', 'sersic_nmgy_z'
    ],
    
    "Mass Estimates": [
        'elpetro_mass', 'elpetro_mass_log', 'mass_inf_photoz', 'mass_med_photoz', 'mass_sup_photoz'
    ],
    
    "Star Formation Properties": [
        'sfr_inf_photoz', 'sfr_sup_photoz', 'ssfr_inf_photoz', 'ssfr_med_photoz', 'ssfr_sup_photoz',
        'fibre_sfr_avg', 'fibre_sfr_entropy', 'fibre_sfr_median', 'fibre_sfr_mode',
        'fibre_sfr_p16', 'fibre_sfr_p2p5', 'fibre_sfr_p84', 'fibre_sfr_p97p5',
        'fibre_ssfr_avg', 'fibre_ssfr_entropy', 'fibre_ssfr_median', 'fibre_ssfr_mode',
        'fibre_ssfr_p16', 'fibre_ssfr_p2p5', 'fibre_ssfr_p84', 'fibre_ssfr_p97p5',
        'total_ssfr_avg', 'total_ssfr_entropy', 'total_ssfr_flag', 'total_ssfr_median',
        'total_ssfr_mode', 'total_ssfr_p16', 'total_ssfr_p2p5', 'total_ssfr_p84', 'total_ssfr_p97p5',
        'total_sfr_avg', 'total_sfr_entropy', 'total_sfr_flag', 'total_sfr_median',
        'total_sfr_mode', 'total_sfr_p16', 'total_sfr_p2p5', 'total_sfr_p84', 'total_sfr_p97p5'
    ],
    
    "Morphology (Galaxy Zoo)": [
        'smooth-or-featured_smooth_fraction', 'smooth-or-featured_featured-or-disk_fraction',
        'smooth-or-featured_artifact_fraction', 'disk-edge-on_yes_fraction', 'disk-edge-on_no_fraction',
        'has-spiral-arms_yes_fraction', 'has-spiral-arms_no_fraction', 'bar_strong_fraction',
        'bar_weak_fraction', 'bar_no_fraction', 'bulge-size_dominant_fraction',
        'bulge-size_large_fraction', 'bulge-size_moderate_fraction', 'bulge-size_small_fraction',
        'bulge-size_none_fraction', 'how-rounded_round_fraction', 'how-rounded_in-between_fraction',
        'how-rounded_cigar-shaped_fraction', 'edge-on-bulge_boxy_fraction', 'edge-on-bulge_none_fraction',
        'edge-on-bulge_rounded_fraction', 'spiral-winding_tight_fraction', 'spiral-winding_medium_fraction',
        'spiral-winding_loose_fraction', 'spiral-arm-count_1_fraction', 'spiral-arm-count_2_fraction',
        'spiral-arm-count_3_fraction', 'spiral-arm-count_4_fraction', 'spiral-arm-count_more-than-4_fraction',
        'spiral-arm-count_cant-tell_fraction', 'merging_none_fraction', 'merging_minor-disturbance_fraction',
        'merging_major-disturbance_fraction', 'merging_merger_fraction'
    ],
    
    "OSSY Spectroscopic Properties": [
        'dr7objid_ossy', 'log_l_oiii', 'fwhm', 'e_fwhm', 'equiv_width', 'log_l_ha',
        'log_m_bh', 'upper_e_log_m_bh', 'lower_e_log_m_bh', 'log_bolometric_l'
    ],
    
    "HI Properties": [
        'W50', 'sigW', 'W20', 'HIflux', 'sigflux', 'SNR', 'RMS', 'Dist', 'sigDist', 'logMH', 'siglogMH'
    ],
    
    "Other Measurements": [
        'sky_separation_arcsec_from_photoz', 'elpetro_flux_r'
    ]
}



def format_metadata_value(key, value):
    """
    Format a metadata value for display in a prompt.
    
    Handles different types of astronomical data with appropriate decimal places:
    - Coordinates (ra, dec): 6 decimals
    - Magnitudes: 2 decimals
    - Redshifts: 6 decimals
    - Fractions/percentages: 2 decimals
    - Masses/SFR in log scale: 4 decimals with indication
    
    Args:
        key: Field name
        value: Value to format
    
    Returns:
        str or None: Formatted value, or None if not displayable
    """
    # Exclude irrelevant values
    if value is None or str(type(value)).find('PIL') >= 0 or key == '__index_level_0__':
        return None
    
    # Format numeric values
    if isinstance(value, (int, float)):
        # Fractions/percentages (Galaxy Zoo)
        if key.endswith('_fraction'):
            return f"{value:.2f}"
        # Magnitudes
        elif key.startswith('mag_') or key.find('_mag_') > 0:
            return f"{value:.2f}"
        # Celestial coordinates
        elif key in ['ra', 'dec', 'ra_photoz', 'dec_photoz', 'ra_ossy', 'dec_ossy', 
                     'ra_alf', 'dec_alf', 'ra_jhu', 'dec_jhu']:
            return f"{value:.6f}"
        # Redshifts
        elif key in ['redshift', 'photo_z', 'redshift_nsa', 'redshift_ossy', 'spec_z']:
            return f"{value:.6f}"
        # Errors
        elif key.find('err') > 0 or key.startswith('e_') or key.startswith('sig'):
            return f"{value:.6f}"
        # Angles
        elif key.find('theta') > 0 or key.find('phi') > 0:
            return f"{value:.4f}"
        # Masses and SFR in log scale
        elif (key.startswith('mass_') or key.startswith('sfr_') or key.startswith('ssfr_')) and (-20 < value < 20):
            return f"{value:.4f} (log10)"
        # Default
        return f"{value:.4f}" if isinstance(value, float) else f"{value}"
    
    # String values
    return value


def save_results(results, filename):
    """Save results to JSON with indentation."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {filename}")


#### PROMPT CONFIGURATION

# CAPTION MODE PROMPT
CAPTION_PROMPT_INTRO = """
Describe this image of a galaxy in detail. Only describe the galaxy, not the foreground objects surrounding it. Act like a citizen scientist. Provide your description in a scientific yet accessible tone.

Here is additional information about this galaxy:
"""

CAPTION_PROMPT_CLOSING = """
Based on this information and what you see in the image, provide a detailed scientific description of this galaxy.

Guidelines:
- Write 5-8 sentences with rich astronomical detail
- Describe morphology (spiral, elliptical, irregular, lenticular), visible structures (arms, bar, bulge, disk), colors, and any notable features
- Use proper astronomical terminology naturally
- Do NOT cite numerical values or reference the data above directly
- Simply let the data guide your visual interpretation
"""


# CONVERSATION MODE PROMPT (AstroPetey)
CONVERSATION_PROMPT_INTRO = """
Generate an entertaining and educational conversation between a curious human and AstroPetey, an astronomical AI assistant passionate about galaxies.

The conversation should use the following galaxy data as context (but never reference values directly):
"""

CONVERSATION_PROMPT_CLOSING = """
The conversation should follow this pattern:
1. Human asks AstroPetey about the galaxy in the image
2. AstroPetey explains the basic features enthusiastically, using a friendly tone with occasional astronomy puns
3. Human asks a follow-up question about something specific (morphology, color, size, etc.)
4. AstroPetey provides a more detailed explanation, connecting the feature to broader astronomical concepts
5. Human expresses curiosity and asks another question
6. AstroPetey shares an interesting fact or comparison about this type of galaxy

Guidelines for AstroPetey's personality:
- Enthusiastic and passionate about astronomy
- Uses accessible language but includes proper scientific terminology
- Occasionally uses space-related puns ("out of this world", "stellar example", etc.)
- Sometimes anthropomorphizes galaxies playfully ("this galaxy seems to be going through a phase")
- Connects observations to broader astronomical concepts
- Expresses wonder at cosmic beauty

Your response must be valid JSON:
{
  "conversation": [
    {"speaker": "human", "text": "..."},
    {"speaker": "astropetey", "text": "..."},
    ...
  ]
}

Make it informative, engaging, and scientifically accurate.
"""


# METADATA FORMATTING
def interpret_galaxy_zoo_textual(example):
    """
    Simple version: formats significant Galaxy Zoo fractions (>0.2)
    and lets the LLM interpret them.
    
    Args:
        example: Dict containing galaxy metadata
    
    Returns:
        str: Formatted significant fractions
    """
    # Galaxy Zoo keys to include, grouped by concept
    morphology_groups = {
        "Overall Type": [
            ('smooth-or-featured_smooth_fraction', 'Smooth/Elliptical'),
            ('smooth-or-featured_featured-or-disk_fraction', 'Featured/Disk'),
            ('smooth-or-featured_artifact_fraction', 'Artifact'),
        ],
        "Disk Orientation": [
            ('disk-edge-on_yes_fraction', 'Edge-on'),
            ('disk-edge-on_no_fraction', 'Face-on/Inclined'),
        ],
        "Spiral Arms": [
            ('has-spiral-arms_yes_fraction', 'Has Spiral Arms'),
            ('has-spiral-arms_no_fraction', 'No Spiral Arms'),
        ],
        "Spiral Winding": [
            ('spiral-winding_tight_fraction', 'Tight'),
            ('spiral-winding_medium_fraction', 'Medium'),
            ('spiral-winding_loose_fraction', 'Loose'),
        ],
        "Bar Structure": [
            ('bar_strong_fraction', 'Strong Bar'),
            ('bar_weak_fraction', 'Weak Bar'),
            ('bar_no_fraction', 'No Bar'),
        ],
        "Bulge Size": [
            ('bulge-size_dominant_fraction', 'Dominant'),
            ('bulge-size_large_fraction', 'Large'),
            ('bulge-size_moderate_fraction', 'Moderate'),
            ('bulge-size_small_fraction', 'Small'),
            ('bulge-size_none_fraction', 'None'),
        ],
        "Shape (Ellipticals)": [
            ('how-rounded_round_fraction', 'Round'),
            ('how-rounded_in-between_fraction', 'Intermediate'),
            ('how-rounded_cigar-shaped_fraction', 'Cigar-shaped'),
        ],
        "Merging/Interaction": [
            ('merging_none_fraction', 'No Interaction'),
            ('merging_minor-disturbance_fraction', 'Minor Disturbance'),
            ('merging_major-disturbance_fraction', 'Major Disturbance'),
            ('merging_merger_fraction', 'Merger'),
        ],
    }
    
    result_lines = []
    
    for group_name, fields in morphology_groups.items():
        group_values = []
        for key, label in fields:
            value = example.get(key, 0) or 0
            if value > 0.2:  # Only significant values
                group_values.append(f"{label}: {value:.0%}")
        
        if group_values:
            result_lines.append(f"• {group_name}: {', '.join(group_values)}")
    
    return "\n".join(result_lines) if result_lines else "• No significant morphology data available"


def interpret_galaxy_zoo_detailed(example): # NOT USED / tests in progress
    """
    Detailed version: interprets Galaxy Zoo fractions with full logic.
    
    Converts raw probabilities into readable morphological analysis
    that helps the model understand the galaxy structure.
    
    Note: This version is heavier but provides pre-digested interpretation.
    Use MORPHOLOGY_INTERPRETATION_MODE = "detailed" to enable.
    
    Args:
        example: Dict containing galaxy metadata
    
    Returns:
        str: Textual description of interpreted morphology
    """
    interpretations = []
    
    # --- General type: Smooth vs Featured ---
    smooth = example.get('smooth-or-featured_smooth_fraction', 0) or 0
    featured = example.get('smooth-or-featured_featured-or-disk_fraction', 0) or 0
    artifact = example.get('smooth-or-featured_artifact_fraction', 0) or 0
    
    if artifact > 0.3:
        interpretations.append(f"⚠ Possible artifact or image quality issue (artifact probability: {artifact:.0%})")
    elif smooth > featured:
        if smooth > 0.7:
            interpretations.append(f"Smooth elliptical-type galaxy (confidence: {smooth:.0%})")
        elif smooth > 0.5:
            interpretations.append(f"Likely smooth/elliptical galaxy ({smooth:.0%} smooth vs {featured:.0%} featured)")
    else:
        if featured > 0.7:
            interpretations.append(f"Featured/disk galaxy with visible structure (confidence: {featured:.0%})")
        elif featured > 0.5:
            interpretations.append(f"Likely disk galaxy with features ({featured:.0%} featured vs {smooth:.0%} smooth)")
    
    # --- Orientation: Edge-on ---
    edge_on_yes = example.get('disk-edge-on_yes_fraction', 0) or 0
    edge_on_no = example.get('disk-edge-on_no_fraction', 0) or 0
    
    if edge_on_yes > 0.6:
        interpretations.append(f"Viewed edge-on (confidence: {edge_on_yes:.0%})")
    elif edge_on_no > 0.6 and featured > 0.5:
        interpretations.append(f"Viewed face-on or at intermediate inclination")
    
    # --- Bras spiraux
    has_spiral = example.get('has-spiral-arms_yes_fraction', 0) or 0
    no_spiral = example.get('has-spiral-arms_no_fraction', 0) or 0
    
    if has_spiral > 0.6:
        # Déterminer l'enroulement
        tight = example.get('spiral-winding_tight_fraction', 0) or 0
        medium = example.get('spiral-winding_medium_fraction', 0) or 0
        loose = example.get('spiral-winding_loose_fraction', 0) or 0
        
        winding = "spiral arms"
        if tight > medium and tight > loose:
            winding = "tightly wound spiral arms (Sa/Sb type)"
        elif loose > medium and loose > tight:
            winding = "loosely wound, open spiral arms (Sc/Sd type)"
        elif medium > 0.3:
            winding = "moderately wound spiral arms (Sb/Sc type)"
        
        # Nb de bras
        arm_counts = {
            '1': example.get('spiral-arm-count_1_fraction', 0) or 0,
            '2': example.get('spiral-arm-count_2_fraction', 0) or 0,
            '3': example.get('spiral-arm-count_3_fraction', 0) or 0,
            '4': example.get('spiral-arm-count_4_fraction', 0) or 0,
            'more than 4': example.get('spiral-arm-count_more-than-4_fraction', 0) or 0,
        }
        best_count = max(arm_counts, key=arm_counts.get)
        if arm_counts[best_count] > 0.4:
            interpretations.append(f"Has {winding} - likely {best_count} arm(s) ({arm_counts[best_count]:.0%})")
        else:
            interpretations.append(f"Has {winding}")
    
    # --- Barre centrale 
    bar_strong = example.get('bar_strong_fraction', 0) or 0
    bar_weak = example.get('bar_weak_fraction', 0) or 0
    bar_no = example.get('bar_no_fraction', 0) or 0
    
    if bar_strong > 0.4:
        interpretations.append(f"Strong central bar visible (SB type, confidence: {bar_strong:.0%})")
    elif bar_weak > 0.4:
        interpretations.append(f"Weak or subtle bar structure (SAB type, confidence: {bar_weak:.0%})")
    elif bar_no > 0.7:
        interpretations.append(f"No bar detected (SA type)")
    
    # --- Bulbe 
    bulge_dominant = example.get('bulge-size_dominant_fraction', 0) or 0
    bulge_large = example.get('bulge-size_large_fraction', 0) or 0
    bulge_moderate = example.get('bulge-size_moderate_fraction', 0) or 0
    bulge_small = example.get('bulge-size_small_fraction', 0) or 0
    bulge_none = example.get('bulge-size_none_fraction', 0) or 0
    
    bulge_sizes = {
        'dominant (bulge-dominated)': bulge_dominant,
        'large': bulge_large,
        'moderate': bulge_moderate,
        'small': bulge_small,
        'no significant': bulge_none
    }
    best_bulge = max(bulge_sizes, key=bulge_sizes.get)
    if bulge_sizes[best_bulge] > 0.4:
        interpretations.append(f"Bulge: {best_bulge} ({bulge_sizes[best_bulge]:.0%})")
    
    # --- Forme (pour elliptiques) ---
    if smooth > 0.5:
        round_frac = example.get('how-rounded_round_fraction', 0) or 0
        between = example.get('how-rounded_in-between_fraction', 0) or 0
        cigar = example.get('how-rounded_cigar-shaped_fraction', 0) or 0
        
        if round_frac > between and round_frac > cigar and round_frac > 0.4:
            interpretations.append(f"Round/spheroidal shape (E0-E2 type, {round_frac:.0%})")
        elif cigar > round_frac and cigar > between and cigar > 0.3:
            interpretations.append(f"Elongated/cigar shape (E5-E7 type, {cigar:.0%})")
        elif between > 0.4:
            interpretations.append(f"Intermediate ellipticity (E3-E4 type)")
    
    # --- Bulbe edge-on ---
    if edge_on_yes > 0.5:
        boxy = example.get('edge-on-bulge_boxy_fraction', 0) or 0
        rounded = example.get('edge-on-bulge_rounded_fraction', 0) or 0
        no_bulge_eo = example.get('edge-on-bulge_none_fraction', 0) or 0
        
        if boxy > 0.4:
            interpretations.append(f"Boxy/peanut-shaped bulge in edge-on view ({boxy:.0%})")
        elif rounded > 0.4:
            interpretations.append(f"Rounded bulge visible in edge-on view ({rounded:.0%})")
    
    # --- Interaction/Fusion ---
    merging_none = example.get('merging_none_fraction', 0) or 0
    merging_minor = example.get('merging_minor-disturbance_fraction', 0) or 0
    merging_major = example.get('merging_major-disturbance_fraction', 0) or 0
    merging_merger = example.get('merging_merger_fraction', 0) or 0
    
    if merging_merger > 0.3:
        interpretations.append(f"🔄 MERGING: Active galaxy merger in progress ({merging_merger:.0%})")
    elif merging_major > 0.3:
        interpretations.append(f"⚡ Major disturbance/interaction signs ({merging_major:.0%})")
    elif merging_minor > 0.3:
        interpretations.append(f"Minor disturbance or tidal features ({merging_minor:.0%})")
    
    # result
    if interpretations:
        return "\n".join(f"• {interp}" for interp in interpretations)
    else:
        return "• Morphological classification uncertain from citizen science data"


def format_all_metadata_for_prompt(example):
    """
    Format ALL metadata for the prompt, organized by category.
    Similar to the original caption_galaxies.py approach.
    
    Args:
        example: Dict containing galaxy metadata
    
    Returns:
        str: Metadata formatted by category
    """
    result = ""
    
    for category, keys in METADATA_CATEGORIES.items():
        category_values = {}
        
        for key in keys:
            if key in example and example[key] is not None:
                formatted_value = format_metadata_value(key, example[key])
                if formatted_value is not None:
                    category_values[key] = formatted_value
        
        if category_values:
            result += f"\n\n{category}:"
            for key, value in category_values.items():
                display_key = key.replace('_', ' ').replace('-', ' ')
                display_key = ' '.join(word.capitalize() for word in display_key.split())
                result += f"\n- {display_key}: {value}"
    
    return result


def create_galaxy_prompt(example, conversation=False):
    """
    Generate a prompt to describe a galaxy (caption or conversation mode).
    
    Based on the original caption_galaxies.py approach - simple and effective.
    The metadata is provided as context, the model describes what it sees.
    
    Args:
        example: Dict containing galaxy metadata
        conversation: If True, generate conversation prompt; else caption prompt
    
    Returns:
        str: Complete formatted prompt for the LLM
    """
    # Format all metadata
    metadata_text = format_all_metadata_for_prompt(example)
    
    if conversation:
        # Conversation mode with AstroPetey
        return CONVERSATION_PROMPT_INTRO + metadata_text + CONVERSATION_PROMPT_CLOSING
    else:
        # Caption mode (simple description)
        return CAPTION_PROMPT_INTRO + metadata_text + CAPTION_PROMPT_CLOSING



# Gemini API calls
async def call_gemini_async(image, prompt, semaphore: asyncio.Semaphore, dr8_id: str = "?"):
    """
    Call Gemini API asynchronously with retry and exponential backoff.
    
    Args:
        image: PIL Image
        prompt: Prompt text
        semaphore: Semaphore to limit concurrency
        dr8_id: Galaxy ID (for logging)
    
    Returns:
        str: Response text or None on failure
    """
    async with semaphore:
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response = await client.aio.models.generate_content(
                    contents=[prompt, image],
                    model=GEMINI_MODEL,
                )
                return response.text
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Rate limit or server overload error -> retry
                if any(x in error_str for x in ['429', 'rate', 'quota', '503', '500', 'overloaded']):
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    if attempt < MAX_RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(delay)
                        continue
                
                # Last attempt or non-recoverable error
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    print(f"  ✗ Final failure for {dr8_id} after {MAX_RETRY_ATTEMPTS} attempts: {e}")
                    return None
                
                # Other errors: retry with delay
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                await asyncio.sleep(delay)
        
        return None


async def caption_image_async(image, metadata, semaphore: asyncio.Semaphore):
    """
    Generate a caption for a galaxy image (async version).
    
    Args:
        image: PIL Image of the galaxy
        metadata: Dict of galaxy metadata
        semaphore: Semaphore to limit concurrency
    
    Returns:
        str: Caption text or None
    """
    prompt = create_galaxy_prompt(metadata)
    dr8_id = str(metadata.get('dr8_id', '?'))
    return await call_gemini_async(image, prompt, semaphore, dr8_id)


# Synchronous version for single mode (kept for simplicity)
def call_gemini(image, prompt):
    """Call Gemini API synchronously (single mode only)."""
    return client.models.generate_content(
        contents=[prompt, image],
        model=GEMINI_MODEL,
    )


def caption_image(image, metadata, conversation=False):
    """
    Generate a caption or conversation synchronously (single mode only).
    
    Args:
        image: PIL Image
        metadata: Dict of galaxy metadata
        conversation: If True, generate conversation; else caption
    
    Returns:
        str or dict: Caption text, or conversation dict if conversation mode
    """
    prompt = create_galaxy_prompt(metadata, conversation=conversation)
    response = call_gemini(image, prompt)
    
    if conversation:
        # Parse JSON response for conversation mode
        import json
        text = response.text
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse conversation JSON: {e}")
            return {"conversation": None, "raw_text": response.text}
    else:
        return response.text



# GALAXY PROCESSING
def extract_image(galaxy_data):
    """
    Extract PIL image from a galaxy data dict.
    
    Handles two possible formats:
    - Streaming: image is directly a PIL.Image
    - Batch/parquet: image is a dict with 'bytes'
    
    Args:
        galaxy_data: Dict containing galaxy image and metadata
    
    Returns:
        PIL.Image: Extracted image
    
    Raises:
        ValueError: If format is not recognized
    """
    image_data = galaxy_data.get('image')
    
    if isinstance(image_data, Image.Image):
        return image_data
    elif isinstance(image_data, dict) and 'bytes' in image_data:
        return Image.open(io.BytesIO(image_data['bytes']))
    else:
        raise ValueError(f"Unrecognized image format: {type(image_data)}")


async def process_galaxy_async(galaxy_data, semaphore: asyncio.Semaphore):
    """
    Process a galaxy asynchronously.
    
    Args:
        galaxy_data: Dict containing galaxy image and metadata
        semaphore: Semaphore to limit concurrency
    
    Returns:
        dict: Result with dr8_id and caption
    """
    try:
        image = extract_image(galaxy_data)
        image_id = galaxy_data['dr8_id']
        
        # Generate caption (async)
        result = await caption_image_async(image, galaxy_data, semaphore)
        
        return {
            'dr8_id': image_id,
            'caption': result
        }
            
    except Exception as e:
        print(f" Error on {galaxy_data.get('dr8_id', '?')}: {e}")
        return {
            'dr8_id': galaxy_data.get('dr8_id'),
            'caption': None
        }


def process_galaxy(galaxy_data, conversation=False):
    """
    Process a galaxy (synchronous version for single mode).
    
    Args:
        galaxy_data: Dict containing galaxy image and metadata
        conversation: If True, generate conversation; else caption
    
    Returns:
        dict: Result with dr8_id and caption/conversation
    """
    try:
        image = extract_image(galaxy_data)
        image_id = galaxy_data['dr8_id']
        
        # Generate caption or conversation
        result = caption_image(image, galaxy_data, conversation=conversation)
        
        if conversation:
            return {
                'dr8_id': image_id,
                'conversation': result.get('conversation') if isinstance(result, dict) else None
            }
        else:
            return {
                'dr8_id': image_id,
                'caption': result
            }
            
    except Exception as e:
        print(f"  ✗ Error on {galaxy_data.get('dr8_id', '?')}: {e}")
        if conversation:
            return {
                'dr8_id': galaxy_data.get('dr8_id'),
                'conversation': None
            }
        else:
            return {
                'dr8_id': galaxy_data.get('dr8_id'),
                'caption': None
            }



### SINGLE MODE: PROCESS A SPECIFIC GALAXY
def process_single_galaxy(dr8_id=None, index=None, split="train", save_png=False, conversation=False):
    """
    Process a single galaxy by its identifier or index.
    
    Args:
        dr8_id: Unique DR8 identifier of the galaxy (optional)
        index: Position in the dataset (optional, takes priority if provided)
        split: Split to use ("train", "validation", "test")
        save_png: Whether to save the galaxy image as PNG
        conversation: If True, generate conversation instead of caption
    
    Returns:
        dict or None: Result with dr8_id and caption/conversation
    """
    # === INDEX MODE: Use streaming to avoid downloading entire dataset ===
    if index is not None:
        print(f"Loading galaxy at index {index} from '{split}' (streaming)...")
        galaxies = load_dataset(
            DATASET_ID, 
            split=split, 
            revision=DATASET_REVISION,
            streaming=True
        )
        
        # Skip to the desired index
        if index > 0:
            galaxies = galaxies.skip(index)
        
        # Get the first (and only needed) galaxy
        try:
            galaxy = next(iter(galaxies))
            print(f"Galaxy dr8_id={galaxy['dr8_id']} at index {index}")
        except StopIteration:
            print(f"✗ Index {index} out of bounds")
            return None
    
    # === DR8_ID MODE: Linear search ===
    elif dr8_id is not None:
        target_id = str(dr8_id)
        print(f"Searching for dr8_id={dr8_id}...")
        print(f"Loading dataset '{split}'...")
        galaxies = load_dataset(DATASET_ID, split=split, revision=DATASET_REVISION)
        
        found_index = None
        for i, gid in enumerate(tqdm(galaxies['dr8_id'], desc="Search")):
            if str(gid) == target_id:
                found_index = i
                break
        
        if found_index is None:
            print(f"Galaxy {dr8_id} not found")
            return None
        print(f"Found at index {found_index}")
        galaxy = galaxies[found_index]
    
    else:
        print("Error: neither dr8_id nor index specified")
        return None
    
    # Processing
    mode_str = "conversation" if conversation else "caption"
    print(f"Generating {mode_str}...")
    result = process_galaxy(galaxy, conversation=conversation)
    
    # Save PNG if requested
    galaxy_id = galaxy.get('dr8_id', 'unknown')
    if save_png:
        image = galaxy.get('image')
        if image:
            png_filename = f"galaxy_{galaxy_id}.png"
            image.save(png_filename)
            print(f"Image saved: {png_filename}")
        else:
            print("No image found")
    
    # Display result
    print("\n" + "=" * 70)
    print(f"Result for dr8_id: {galaxy_id}")
    print("=" * 70)
    
    if conversation:
        conv = result.get('conversation')
        if conv:
            for turn in conv:
                speaker = turn.get('speaker', '?').upper()
                text = turn.get('text', '')
                print(f"\n[{speaker}]: {text}")
        else:
            print(" No conversation generated")
    else:
        if result.get('caption'):
            print(result['caption'])
        else:
            print("No result generated")
    
    print("=" * 70)
    
    return result



# BATCH MODE: PROCESS ENTIRE DATASET
async def process_batch_async(batch_list: list, semaphore: asyncio.Semaphore, desc: str = ""):
    """
    Process a batch of galaxies asynchronously.
    
    Args:
        batch_list: List of galaxy data dicts to process
        semaphore: Semaphore to limit concurrency
        desc: Description for progress bar
    
    Returns:
        list: List of results
    """
    tasks = [process_galaxy_async(galaxy_data, semaphore) for galaxy_data in batch_list]
    
    # Use tqdm_asyncio to display progress
    results = await tqdm_asyncio.gather(*tasks, desc=desc)
    return list(results)


async def process_batch_dataset_async(split="validation", max_concurrent: int = DEFAULT_MAX_CONCURRENT):
    """
    Process the entire dataset in batches with checkpoints (async version).
    
    Uses asyncio to parallelize API calls efficiently.
    Saves regularly to allow resumption on interruption.
    
    Args:
        split: Split to process
        max_concurrent: Max number of concurrent requests
    """
    # Checkpoint file for resumption
    checkpoint_file = f"galaxy_caption_{split}_partial.json"
    
    # Load existing results if resuming
    results = []
    completed_count = 0
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                results = json.load(f)
                completed_count = len(results)
                print(f"Resuming: {completed_count} existing results loaded")
        except Exception as e:
            print(f"Unable to load checkpoint: {e}")
            print(" Starting from scratch...")
    
    # Load dataset in streaming mode
    print(f"Loading dataset '{split}'...")
    galaxies = load_dataset(
        DATASET_ID, 
        split=split, 
        revision=DATASET_REVISION, 
        streaming=True
    )
    
    # Skip already processed examples
    if completed_count > 0:
        print(f"Skipping first {completed_count} examples...")
        galaxies = galaxies.skip(completed_count)
    
    # Info on total count
    try:
        max_examples = galaxies.info.splits[split].num_examples
        remaining_count = max_examples - completed_count
        
        if completed_count >= max_examples:
            print("All examples have already been processed!")
            final_file = f"galaxy_caption_{split}.json"
            save_results(results, final_file)
            return
            
        print(f"Remaining examples: {remaining_count}/{max_examples}")
        
    except Exception:
        remaining_count = None
        print("Unable to determine total number of examples")
    
    # Configuration
    batch_size = DEFAULT_BATCH_SIZE
    semaphore = asyncio.Semaphore(max_concurrent)
    
    print(f"Async mode: {max_concurrent} max concurrent requests")
    
    # Process by batches
    dataset_batched = galaxies.batch(batch_size=batch_size)
    
    for batch_idx, batch in enumerate(dataset_batched):
        # Convert dict of lists to list of dicts
        batch_list = [
            {k: batch[k][i] for k in batch.keys()} 
            for i in range(len(batch[list(batch.keys())[0]]))
        ]
        
        # Calculate progress description
        if remaining_count:
            total_batches = (remaining_count + batch_size - 1) // batch_size
            desc = f"Batch {batch_idx + 1}/{total_batches}"
        else:
            desc = f"Batch {batch_idx + 1}"
        
        # Async parallel processing
        batch_results = await process_batch_async(batch_list, semaphore, desc)
        
        # Save checkpoint
        results.extend(batch_results)
        save_results(results, checkpoint_file)
        
        # Batch stats
        success_count = sum(1 for r in batch_results if r.get('caption'))
        print(f"💾 Checkpoint: {len(results)} galaxies processed ({success_count}/{len(batch_results)} success in this batch)")
    
    # Final save
    final_file = f"galaxy_caption_{split}.json"
    save_results(results, final_file)
    
    # Final stats
    total_success = sum(1 for r in results if r.get('caption'))
    print(f"\nDone! {len(results)} galaxies processed ({total_success} success, {len(results) - total_success} failures)")


def process_batch_dataset(split="validation", max_concurrent: int = DEFAULT_MAX_CONCURRENT):
    """
    Synchronous entry point for batch processing (launches async version).
    
    Args:
        split: Split to process
        max_concurrent: Max number of concurrent requests
    """
    asyncio.run(process_batch_dataset_async(split=split, max_concurrent=max_concurrent))



# MAIN 

def main():
    """Main entry point with CLI argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Generate galaxy captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --id 1237648702986158215           # Search by ID
  %(prog)s --index 0                          # Direct access by position 
  %(prog)s -s train -n 100                    # Batch on train with 100 concurrent requests
        """
    )
    
    parser.add_argument(
        '--id', '--dr8_id',
        type=str,
        default=None,
        metavar='DR8_ID',
        help="Specific galaxy ID to process (single mode)"
    )
    
    parser.add_argument(
        '--index', '-i',
        type=int,
        default=None,
        metavar='N',
        help="Index (position) of a galaxy in the dataset - direct O(1) access"
    )
    
    parser.add_argument(
        '--split', '-s',
        type=str,
        default='validation',
        choices=['train', 'validation', 'test'],
        help="Dataset split (default: validation)"
    )
    
    parser.add_argument(
        '--concurrent', '-n',
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        metavar='N',
        help=f"Max concurrent API requests (default: {DEFAULT_MAX_CONCURRENT}). Adjust based on rate limits."
    )
    
    parser.add_argument(
        '--save-png',
        action='store_true',
        help="Save the galaxy image as PNG alongside the caption (single mode only)"
    )
    
    parser.add_argument(
        '--conversation', '-c',
        action='store_true',
        help="Generate an educational conversation with AstroPetey instead of a caption"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Galaxy Caption ")
    print("=" * 70)
    
    mode = "conversation" if args.conversation else "caption"
    print(f"   Mode: {mode}")
    print(f"   Split: {args.split}")
    if args.index is not None:
        print(f"   Target: index {args.index}")
    elif args.id:
        print(f"   Target: dr8_id {args.id}")
    print("=" * 70 + "\n")
    
    # Single or batch mode
    if args.id is not None or args.index is not None:
        # Single mode
        result = process_single_galaxy(
            dr8_id=args.id,
            index=args.index,
            split=args.split,
            save_png=args.save_png,
            conversation=args.conversation
        )
        
        if result:
            galaxy_id = result.get('dr8_id', args.id or f'index_{args.index}')
            suffix = "conversation" if args.conversation else "caption"
            output_file = f"galaxy_{galaxy_id}_{suffix}.json"
            save_results(result, output_file)
    else:
        # Batch mode
        process_batch_dataset(split=args.split, max_concurrent=args.concurrent)



if __name__ == "__main__":
    main()
