"""
Data Loader for GutMind Explorer
Handles loading and preprocessing of microbiome datasets including:
- American Gut Project (AGP)
- User-uploaded files (Biomesight, Viome, generic CSV)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Key bacterial genera associated with mental health in research
MENTAL_HEALTH_BACTERIA = [
    'Lactobacillus', 'Bifidobacterium', 'Bacteroides', 'Prevotella',
    'Faecalibacterium', 'Roseburia', 'Akkermansia', 'Eubacterium',
    'Clostridium', 'Ruminococcus', 'Streptococcus', 'Enterococcus',
    'Escherichia', 'Blautia', 'Coprococcus', 'Dorea', 'Lachnospira',
    'Oscillospira', 'Dialister', 'Veillonella', 'Sutterella', 
    'Bilophila', 'Desulfovibrio', 'Alistipes', 'Parabacteroides'
]

# Bacteria with research-backed mental health associations
PSYCHOBIOTIC_EFFECTS = {
    'Lactobacillus': {'effect': 'protective', 'confidence': 'high', 'mechanism': 'GABA production, vagus nerve'},
    'Bifidobacterium': {'effect': 'protective', 'confidence': 'high', 'mechanism': 'SCFA production, immune modulation'},
    'Faecalibacterium': {'effect': 'protective', 'confidence': 'high', 'mechanism': 'Butyrate production, anti-inflammatory'},
    'Coprococcus': {'effect': 'protective', 'confidence': 'medium', 'mechanism': 'Dopamine pathway, butyrate'},
    'Dialister': {'effect': 'protective', 'confidence': 'medium', 'mechanism': 'Depleted in depression studies'},
    'Akkermansia': {'effect': 'protective', 'confidence': 'medium', 'mechanism': 'Gut barrier integrity'},
    'Roseburia': {'effect': 'protective', 'confidence': 'medium', 'mechanism': 'Butyrate production'},
    'Blautia': {'effect': 'protective', 'confidence': 'low', 'mechanism': 'SCFA production'},
    'Clostridium': {'effect': 'risk', 'confidence': 'medium', 'mechanism': 'Some species pro-inflammatory'},
    'Bilophila': {'effect': 'risk', 'confidence': 'medium', 'mechanism': 'H2S production, inflammation'},
    'Desulfovibrio': {'effect': 'risk', 'confidence': 'medium', 'mechanism': 'LPS, endotoxin production'},
    'Alistipes': {'effect': 'risk', 'confidence': 'low', 'mechanism': 'Elevated in depression studies'},
}


def generate_synthetic_research_dataset(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset based on American Gut Project distributions.
    This mimics real microbiome + mental health survey data for demonstration.
    """
    np.random.seed(42)
    
    data = {'sample_id': [f'AGP_{i:05d}' for i in range(n_samples)]}
    
    # Generate realistic bacterial abundances (using log-normal distributions)
    abundance_params = {
        'Bacteroides': (2.5, 0.8),      # High abundance
        'Prevotella': (1.0, 1.5),        # Bimodal in reality, simplified
        'Faecalibacterium': (1.8, 0.6),
        'Blautia': (1.5, 0.5),
        'Roseburia': (1.2, 0.6),
        'Ruminococcus': (1.0, 0.5),
        'Coprococcus': (0.8, 0.5),
        'Lachnospira': (0.7, 0.4),
        'Eubacterium': (0.6, 0.5),
        'Oscillospira': (0.5, 0.4),
        'Lactobacillus': (0.3, 0.8),     # Lower but variable
        'Bifidobacterium': (0.4, 0.7),
        'Akkermansia': (0.2, 1.0),       # Highly variable
        'Streptococcus': (0.4, 0.6),
        'Enterococcus': (0.1, 0.5),
        'Escherichia': (0.2, 0.6),
        'Clostridium': (0.5, 0.5),
        'Dorea': (0.6, 0.4),
        'Dialister': (0.3, 0.6),
        'Veillonella': (0.2, 0.5),
        'Sutterella': (0.3, 0.5),
        'Bilophila': (0.1, 0.6),
        'Desulfovibrio': (0.05, 0.5),
        'Alistipes': (0.8, 0.5),
        'Parabacteroides': (0.7, 0.5),
    }
    
    for bacteria, (mean, std) in abundance_params.items():
        # Log-normal distribution, then normalize
        raw = np.random.lognormal(mean, std, n_samples)
        data[bacteria] = raw
    
    # Normalize to relative abundance (sum to 100%)
    df = pd.DataFrame(data)
    bacteria_cols = [c for c in df.columns if c != 'sample_id']
    totals = df[bacteria_cols].sum(axis=1)
    for col in bacteria_cols:
        df[col] = (df[col] / totals) * 100
    
    # Generate mental health scores with realistic correlations
    # Based on Valles-Colomer et al. 2019 findings
    protective = (
        df['Lactobacillus'] * 0.5 +
        df['Bifidobacterium'] * 0.6 +
        df['Faecalibacterium'] * 0.4 +
        df['Coprococcus'] * 0.5 +
        df['Dialister'] * 0.3 +
        df['Akkermansia'] * 0.2
    )
    
    risk = (
        df['Clostridium'] * 0.3 +
        df['Bilophila'] * 0.8 +
        df['Desulfovibrio'] * 0.6 +
        df['Alistipes'] * 0.2
    )
    
    # GAD-7 style anxiety score (0-21 scale)
    noise = np.random.normal(0, 3, n_samples)
    df['anxiety_score'] = np.clip(10 + risk * 0.5 - protective * 0.3 + noise, 0, 21).round(0)
    
    # PHQ-9 style depression score (0-27 scale)
    noise = np.random.normal(0, 3, n_samples)
    df['depression_score'] = np.clip(8 + risk * 0.4 - protective * 0.25 + noise, 0, 27).round(0)
    
    # Binary classifications
    df['anxiety_level'] = (df['anxiety_score'] >= 10).map({True: 'high', False: 'low'})
    df['depression_level'] = (df['depression_score'] >= 10).map({True: 'high', False: 'low'})
    
    # Demographics (for realistic dataset feel)
    df['age'] = np.random.normal(42, 15, n_samples).clip(18, 80).round(0)
    df['sex'] = np.random.choice(['M', 'F'], n_samples, p=[0.45, 0.55])
    
    return df


def parse_biomesight_csv(file_content: str) -> pd.DataFrame:
    """Parse Biomesight export format."""
    df = pd.read_csv(pd.io.common.StringIO(file_content))
    
    # Biomesight typically has columns like: Taxonomy, Percentage, etc.
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Try to identify taxonomy and abundance columns
    taxonomy_col = None
    abundance_col = None
    
    for col in df.columns:
        if 'tax' in col or 'name' in col or 'organism' in col:
            taxonomy_col = col
        if 'percent' in col or 'abundance' in col or 'relative' in col:
            abundance_col = col
    
    if taxonomy_col is None or abundance_col is None:
        # Fallback: assume first col is name, second is abundance
        taxonomy_col = df.columns[0]
        abundance_col = df.columns[1]
    
    # Extract genus level
    result = {}
    for _, row in df.iterrows():
        taxonomy = str(row[taxonomy_col])
        abundance = float(row[abundance_col])
        
        # Try to extract genus from taxonomy string
        # Could be "g__Lactobacillus" or "Bacteria;...;Lactobacillus" etc.
        genus = extract_genus(taxonomy)
        if genus:
            result[genus] = result.get(genus, 0) + abundance
    
    return pd.DataFrame([result])


def extract_genus(taxonomy_string: str) -> Optional[str]:
    """Extract genus name from various taxonomy formats."""
    # Handle "g__Lactobacillus" format
    if 'g__' in taxonomy_string:
        parts = taxonomy_string.split('g__')
        if len(parts) > 1:
            genus = parts[1].split(';')[0].split('|')[0].strip()
            return genus if genus else None
    
    # Handle semicolon-separated taxonomy
    parts = taxonomy_string.split(';')
    if len(parts) >= 6:  # Kingdom;Phylum;Class;Order;Family;Genus
        genus = parts[-1].strip() if parts[-1].strip() else parts[-2].strip()
        return genus
    
    # Handle simple name
    parts = taxonomy_string.split()
    if parts:
        return parts[0].strip()
    
    return None


def parse_generic_csv(file_content: str) -> pd.DataFrame:
    """Parse a generic CSV with bacteria abundances."""
    df = pd.read_csv(pd.io.common.StringIO(file_content))
    df.columns = df.columns.str.strip()
    
    # Check if it's wide format (bacteria as columns) or long format
    if len(df.columns) > 10:
        # Likely wide format - bacteria are columns
        return df
    else:
        # Likely long format - pivot it
        # Try to identify columns
        df.columns = df.columns.str.lower()
        name_col = df.columns[0]
        value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        result = {row[name_col]: row[value_col] for _, row in df.iterrows()}
        return pd.DataFrame([result])


def load_user_data(file_content: str, file_type: str = 'auto') -> Tuple[pd.DataFrame, str]:
    """
    Load user-uploaded microbiome data.
    Returns (dataframe, detected_format)
    """
    content_lower = file_content.lower()
    
    # Try to detect format
    if file_type == 'auto':
        if 'biomesight' in content_lower or 'g__' in content_lower:
            file_type = 'biomesight'
        elif 'viome' in content_lower:
            file_type = 'viome'
        else:
            file_type = 'generic'
    
    if file_type == 'biomesight':
        df = parse_biomesight_csv(file_content)
    else:
        df = parse_generic_csv(file_content)
    
    return df, file_type


def get_bacteria_columns(df: pd.DataFrame) -> List[str]:
    """Identify which columns are bacterial abundances."""
    exclude = {'sample_id', 'anxiety_score', 'depression_score', 
               'anxiety_level', 'depression_level', 'age', 'sex', 
               'id', 'date', 'name'}
    
    bacteria_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col_lower not in exclude and df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            bacteria_cols.append(col)
    
    return bacteria_cols


def calculate_diversity_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate alpha diversity metrics."""
    bacteria_cols = get_bacteria_columns(df)
    abundances = df[bacteria_cols].values.flatten()
    abundances = abundances[abundances > 0]  # Remove zeros
    
    # Normalize to proportions
    proportions = abundances / abundances.sum()
    
    # Shannon diversity
    shannon = -np.sum(proportions * np.log(proportions + 1e-10))
    
    # Simpson's diversity
    simpson = 1 - np.sum(proportions ** 2)
    
    # Richness (number of detected taxa)
    richness = len(abundances)
    
    # Evenness
    evenness = shannon / np.log(richness) if richness > 1 else 0
    
    return {
        'shannon': round(shannon, 3),
        'simpson': round(simpson, 3),
        'richness': richness,
        'evenness': round(evenness, 3)
    }


# Initialize dataset on module load
_cached_dataset = None

def get_research_dataset() -> pd.DataFrame:
    """Get the research dataset (cached)."""
    global _cached_dataset
    if _cached_dataset is None:
        logger.info("Generating synthetic research dataset...")
        _cached_dataset = generate_synthetic_research_dataset(n_samples=500)
        logger.info(f"Dataset ready: {len(_cached_dataset)} samples")
    return _cached_dataset
