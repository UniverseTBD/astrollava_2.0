"""
X-ray Plots description (light curves et spectres) with Gemini API

Usage:
    # unique file
    python caption_xray.py --file plots/js_lcplot_1050360105_goddard_GTI17.jpeg

    # all plots in folder
    python caption_xray.py --folder plots/

    # save results 
    python caption_xray.py --folder plots/ --output results.json
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from PIL import Image
from google import genai
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio


# CONFIG

# API Key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "key")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Model
GEMINI_MODEL = "gemini-3-pro-preview"  

# Batch params
DEFAULT_MAX_CONCURRENT = 10  # Max concurrent requests
MAX_RETRY_ATTEMPTS = 3  # retry on error
RETRY_BASE_DELAY = 1.0  # delay for exp backoff (s)

# img extensions
SUPPORTED_EXTENSIONS = {'.jpeg', '.jpg', '.png', '.gif', '.webp'}


# ---PROMPT ---

# Light Curve (lcplot) - Count Rate vs Time
LCPLOT_PROMPT = """
You are the Principal Investigator (PI) of a high-energy X-ray astrophysics mission. You are analyzing a preliminary light curve from a target of interest.

YOUR GOAL: Perform a rigorous physical diagnosis of the source state based on the variability properties shown in the plot.

## STEP 1: INTERNAL PHYSICS MONOLOGUE (Reasoning Phase)
(This step is for your internal reasoning ONLY. Absolutely DO NOT include it in the final output.)
1. **Metadata Check**: Is this a known source? If so, what is its typical behavior?
2. **Variability Diagnosis**:
   - Is it Poisson noise dominated (low counts) or intrinsic variability?
   - If variable, is it structured (flaring, dipping) or stochastic (red noise)?
   - Are there QPOs? -> implies inner disk geometry.
   - Are there bursts? -> Type I (thermonuclear) vs Type II (instability).
3. **State Classification**:
   - High/Soft state (stable, disk dominated)?
   - Low/Hard state (variable, jet/corona dominated)?
   - Intermediate state (transitions, possible ejections)?

## STEP 2: GENERATE EXPERT SYNTHESIS
Output a **single, highly dense scientific paragraph** (5-8 sentences) suitable for the abstract of a paper.

**Instructions:**
- **Synthesize** the observation details, variability morphology, and physical interpretation into one cohesive narrative.
- **Be Quantitative:** Integrate numbers naturally (e.g., "The source exhibits rapid flickering (~30% RMS) superimposed on a mean rate of 7500 cts/s...").
- **Be Physical:** Connect the morphology directly to the physics (e.g., "...indicating a turbulent Comptonizing corona typical of the Hard Intermediate State").
- **Style:** Academic, dense, no bullet points.
**IMPORTANT**: Your final response must ONLY contain the text from STEP 2. Do not include headers, the monologue, or any other text.
"""

LCPLOT_PROMPT_SHORT = """
Act as an X-ray PI. Briefly diagnose this light curve.
1. QUANTIFY: Mean rate, variability amplitude (RMS estimate), and timescales using header info.
2. INTERPRET: Classify the variability (red noise, QPO, bursts, dipping) and infer the accretion state (e.g., "High/Soft" vs "Low/Hard").
3. PHYSICS: Mention the likely driver (disk instability, wind clumping, thermonuclear burning).

Keep it dense, jargon-rich, and under 5 sentences.
"""


# Spectral Plot - Counts/PI vs Energy
SPECTRAL_PROMPT = """
You are the Principal Investigator (PI) of a high-energy X-ray astrophysics mission. You are analyzing a quick-look spectrum.

YOUR GOAL: Identify the physical components (Disk, Corona, Reflection) and determine the accretion state.

## STEP 1: INTERNAL PHYSICS MONOLOGUE (Reasoning Phase)
(This step is for your internal reasoning ONLY. Absolutely DO NOT include it in the final output.)
1. **Model Identification**:
   - Soft bump? -> Disk Blackbody. Hard tail? -> Comptonization.
   - Lines? -> Fe K (6.4-7 keV). Absorption? -> Low-E turnover.
2. **State Diagnostics**:
   - Steep spectrum (Gamma > 2) + Disk -> Soft State.
   - Flat spectrum (Gamma < 1.8) + Weak Disk -> Hard State.
3. **Physics Check**: Does the temperature/index make sense for the identified source type?

## STEP 2: GENERATE EXPERT SYNTHESIS
Output a **single, highly dense scientific paragraph** (5-8 sentences) suitable for the abstract of a paper.

**Instructions:**
- **Synthesize** continuum shape, spectral features, and state classification into one cohesive narrative.
- **Be Quantitative:** Integrate parameters naturally (e.g., "...dominated by a soft thermal component peaking at 1 keV...", "...hard power-law tail (Gamma ~ 1.7)...").
- **Be Physical:** Interpret the components (e.g., "suggesting a refined, optically thick disk extending to the ISCO with a quenched corona").
- **Style:** Academic, dense, no bullet points. Use XSPEC terminology (`diskbb`, `powerlaw`, `tbabs`) naturally in sentences.
**IMPORTANT**: Your final response must ONLY contain the text from STEP 2. Do not include headers, the monologue, or any other text.
"""

SPECTRAL_PROMPT_SHORT = """
Act as an X-ray PI. Briefly diagnose this spectrum.
1. IDENTIFY: Dominant continuum (Disk blackbody vs. Comptonized Powerlaw) and estimated hardness.
2. FEATURES: Note specific lines (Fe K, discrete absorption) or residuals.
3. STATE: Conclude the spectral state (e.g., "Soft canonical", "Hard Compton-dominated") and accretion geometry.

Use XSPEC terminology (`diskbb`, `powerlaw`, `gauss`). Keep it dense and under 5 sentences.
"""


# Generic prompt for unknown plot types
GENERIC_XRAY_PROMPT = """
You are the Principal Investigator (PI) of a high-energy X-ray astrophysics mission. You are analyzing an unknown X-ray plot.

YOUR GOAL: Identify the data type, extract physics, and produce an expert summary.

## STEP 1: INTERNAL PHYSICS MONOLOGUE (Reasoning Phase)
(This step is for your internal reasoning ONLY. Absolutely DO NOT include it in the final output.)
1. **Identification**:
   - What are the axes? (Time vs Rate = Lightcurve? Energy vs Counts = Spectrum? Hardness vs Intensity = HID?)
   - What is the target? (Read labels).
2. **Analysis**:
   - If Lightcurve: Look for variability, bursts, QPOs.
   - If Spectrum: Look for continuum shape (thermal/non-thermal), lines (Fe K), edges.
   - If HID/CCD: Look for hysteresis, "q-diagram" tracks, Z-source vs Atoll tracks.
3. **Interpretation**:
   - Connect the visual features to accretion physics (State, Geometry, Object Class).

## STEP 2: GENERATE EXPERT SYNTHESIS
Output a **single, highly dense scientific paragraph** (5-8 sentences).

**Instructions:**
- **Synthesize** the plot identification, key quantitative features, and physical interpretation.
- **Be Quantitative:** Use numbers read from axes.
- **Be Physical:** Use jargon appropriate to the identified plot type (e.g., "hysteretic tracking in the HID", "red noise in the PDS").
- **Style:** Academic, dense, no bullet points.
**IMPORTANT**: Your final response must ONLY contain the text from STEP 2. Do not include headers, the monologue, or any other text.
"""



# --- UTILS ---


def detect_plot_type(filename: str) -> str:
    """
    Detect plot type from filename.
    
    Args:
        filename: Name of the image file
    
    Returns:
        str: 'lcplot', 'spectral', or 'unknown'
    """
    filename_lower = filename.lower()
    
    if 'lcplot' in filename_lower or 'lightcurve' in filename_lower or 'lc_' in filename_lower:
        return 'lcplot'
    elif 'spectral' in filename_lower or 'spectrum' in filename_lower or 'spec_' in filename_lower:
        return 'spectral'
    else:
        return 'unknown'


def get_prompt_for_type(plot_type: str, short: bool = False) -> str:
    """
    Get the appropriate prompt for a plot type.
    
    Args:
        plot_type: Type of plot ('lcplot', 'spectral', 'unknown')
        short: If True, return the short version of the prompt
    
    Returns:
        str: Appropriate prompt
    """
    if plot_type == 'lcplot':
        return LCPLOT_PROMPT_SHORT if short else LCPLOT_PROMPT
    elif plot_type == 'spectral':
        return SPECTRAL_PROMPT_SHORT if short else SPECTRAL_PROMPT
    else:
        return GENERIC_XRAY_PROMPT


def load_image(filepath: str) -> Image.Image:
    """
    Load an image from file.
    
    Args:
        filepath: Path to the image file
    
    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(filepath)


def save_results(results: list, filename: str):
    """Save results to JSON with indentation."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {filename}")


def get_image_files(folder: str) -> list:
    """
    Get all supported image files from a folder.
    
    Args:
        folder: Path to the folder
    
    Returns:
        list: List of image file paths
    """
    folder_path = Path(folder)
    image_files = []
    
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


# --- GEMINI API ---


def call_gemini(image: Image.Image, prompt: str) -> str:
    """
    Call Gemini API synchronously.
    
    Args:
        image: PIL Image
        prompt: Prompt text
    
    Returns:
        str: Response text
    """
    response = client.models.generate_content(
        contents=[prompt, image],
        model=GEMINI_MODEL,
    )
    return response.text


async def call_gemini_async(
    image: Image.Image, 
    prompt: str, 
    semaphore: asyncio.Semaphore,
    filename: str = "?"
) -> str:
    """
    Call Gemini API asynchronously with retry and exp. backoff.
    
    Args:
        image: PIL Image
        prompt: Prompt text
        semaphore: Semaphore to limit concurrency
        filename: Filename for logging
    
    Returns:
        str: Response text or None on failure
    """
    async with semaphore:
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # Run sync call in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        contents=[prompt, image],
                        model=GEMINI_MODEL,
                    )
                )
                return response.text
                
            except Exception as e:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), 60.0)
                print(f"  ⚠ Error on {filename} (attempt {attempt + 1}): {e}")
                
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(delay)
        
        return None


# --- PROCESSING---


def process_single_file(
    filepath: str, 
    plot_type: str = None, 
    short: bool = False,
    verbose: bool = True
) -> dict:
    """
    Process a single X-ray plot file.
    
    Args:
        filepath: Path to the image file
        plot_type: Type of plot (auto-detected if None)
        short: Use short prompt version
        verbose: Print progress
    
    Returns:
        dict: Result with filename, type, and description
    """
    filepath = Path(filepath)
    filename = filepath.name
    
    # Auto-detect plot type if not specified
    if plot_type is None:
        plot_type = detect_plot_type(filename)
    
    if verbose:
        print(f"Processing: {filename}")
        print(f"  Plot type: {plot_type}")
    
    # Load image
    try:
        image = load_image(str(filepath))
    except Exception as e:
        print(f"  ✗ Error loading image: {e}")
        return {
            'filename': filename,
            'filepath': str(filepath),
            'plot_type': plot_type,
            'description': None,
            'error': str(e)
        }
    
    # Get appropriate prompt
    prompt = get_prompt_for_type(plot_type, short=short)
    
    # Call Gemini
    try:
        description = call_gemini(image, prompt)
        if verbose:
            print(f"  ✓ Description generated")
    except Exception as e:
        print(f"  ✗ API error: {e}")
        return {
            'filename': filename,
            'filepath': str(filepath),
            'plot_type': plot_type,
            'description': None,
            'error': str(e)
        }
    
    return {
        'filename': filename,
        'filepath': str(filepath),
        'plot_type': plot_type,
        'description': description,
        'error': None
    }


async def process_file_async(
    filepath: Path,
    semaphore: asyncio.Semaphore,
    short: bool = False
) -> dict:
    """
    Process a single file asynchronously.
    
    Args:
        filepath: Path to the image file
        semaphore: Semaphore for concurrency control
        short: Use short prompt version
    
    Returns:
        dict: Result with filename, type, and description
    """
    filename = filepath.name
    plot_type = detect_plot_type(filename)
    
    # Load image
    try:
        image = load_image(str(filepath))
    except Exception as e:
        return {
            'filename': filename,
            'filepath': str(filepath),
            'plot_type': plot_type,
            'description': None,
            'error': str(e)
        }
    
    # Get prompt and call API
    prompt = get_prompt_for_type(plot_type, short=short)
    description = await call_gemini_async(image, prompt, semaphore, filename)
    
    return {
        'filename': filename,
        'filepath': str(filepath),
        'plot_type': plot_type,
        'description': description,
        'error': None if description else "API call failed"
    }


async def process_folder_async(
    folder: str,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    short: bool = False
) -> list:
    """
    Process all image files in a folder asynchronously.
    
    Args:
        folder: Path to the folder
        max_concurrent: Maximum concurrent API calls
        short: Use short prompt version
    
    Returns:
        list: List of results
    """
    image_files = get_image_files(folder)
    
    if not image_files:
        print(f"No supported image files found in {folder}")
        return []
    
    print(f"Found {len(image_files)} image files")
    print(f"Max concurrent requests: {max_concurrent}")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = [
        process_file_async(filepath, semaphore, short=short)
        for filepath in image_files
    ]
    
    results = await tqdm_asyncio.gather(*tasks, desc="Processing")
    
    return list(results)


def process_folder(
    folder: str,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    short: bool = False
) -> list:
    """
    Process all image files in a folder (sync wrapper).
    
    Args:
        folder: Path to the folder
        max_concurrent: Maximum concurrent API calls
        short: Use short prompt version
    
    Returns:
        list: List of results
    """
    return asyncio.run(process_folder_async(folder, max_concurrent, short))


# DISPLAY

def display_result(result: dict):
    """Display a single result nicely."""
    print("\n" + "=" * 70)
    print(f"File: {result['filename']}")
    print(f"Type: {result['plot_type']}")
    print("=" * 70)
    
    if result.get('description'):
        print("\nDescription:")
        print("-" * 40)
        print(result['description'])
    else:
        print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
    
    print("=" * 70)


def display_summary(results: list):
    """Display summary statistics."""
    total = len(results)
    successful = sum(1 for r in results if r.get('description'))
    failed = total - successful
    
    by_type = {}
    for r in results:
        ptype = r.get('plot_type', 'unknown')
        by_type[ptype] = by_type.get(ptype, 0) + 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nBy plot type:")
    for ptype, count in sorted(by_type.items()):
        print(f"  - {ptype}: {count}")
    print("=" * 70)


# MAIN

def main():
    """Main CLI argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Generate descriptions for X-ray astronomical plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --file plots/js_lcplot_1050360105_goddard_GTI17.jpeg
  %(prog)s --folder plots/ --output xray_captions.json
  %(prog)s --file myplot.png --type spectral --short
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help="Path to a single image file to analyze"
    )
    input_group.add_argument(
        '--folder', '-d',
        type=str,
        help="Path to a folder containing image files to analyze"
    )
    
    # Processing options
    parser.add_argument(
        '--type', '-t',
        type=str,
        choices=['lcplot', 'spectral', 'unknown'],
        default=None,
        help="Plot type (auto-detected from filename if not specified)"
    )
    
    parser.add_argument(
        '--short', '-s',
        action='store_true',
        help="Use short prompt version for faster, more concise descriptions"
    )
    
    parser.add_argument(
        '--concurrent', '-n',
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f"Max concurrent API requests for folder mode (default: {DEFAULT_MAX_CONCURRENT})"
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Output JSON file to save results"
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Header
    print("=" * 70)
    print("X-Ray Plot Caption Generator")
    print("=" * 70)
    print(f"  Model: {GEMINI_MODEL}")
    print(f"  Mode: {'short' if args.short else 'detailed'}")
    print("=" * 70 + "\n")
    
    # Process
    if args.file:
        # Single file mode
        result = process_single_file(
            args.file,
            plot_type=args.type,
            short=args.short,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            display_result(result)
        
        results = [result]
        
    else:
        # Folder mode
        results = process_folder(
            args.folder,
            max_concurrent=args.concurrent,
            short=args.short
        )
        
        if not args.quiet:
            for result in results:
                display_result(result)
            display_summary(results)
    
    # Save results if requested
    if args.output:
        save_results(results, args.output)
    
    failed = sum(1 for r in results if not r.get('description'))
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
