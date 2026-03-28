"""
GutMind Explorer - Research-Based Data Loader

This module generates synthetic microbiome data based on published research findings.
The correlation patterns and bacterial associations are derived from peer-reviewed studies.

KEY REFERENCES:
1. Valles-Colomer M, et al. (2019) "The neuroactive potential of the human gut microbiota 
   in quality of life and depression." Nature Microbiology, 4(4):623-632.
   - Coprococcus and Dialister depleted in depression
   - Faecalibacterium associated with quality of life
   
2. Radjabzadeh D, et al. (2022) "Gut microbiome-wide association study of depressive 
   symptoms." Nature Communications, 13:7128.
   - Eggerthella enriched in depression
   - Coprococcus, Subdoligranulum depleted in depression
   - Identified 13 taxa associated with depressive symptoms

3. Jiang H, et al. (2015) "Altered fecal microbiota composition in patients with major 
   depressive disorder." Brain, Behavior, and Immunity, 48:186-194.
   - Reduced Faecalibacterium in MDD patients
   - Increased Enterobacteriaceae in MDD

4. Simpson CA, et al. (2021) "The gut microbiota in anxiety and depression - A systematic 
   review." Clinical Psychology Review, 83:101943.
   - Meta-analysis confirming Lactobacillus and Bifidobacterium associations

NOTE: This is SYNTHETIC data generated to reflect published research patterns.
It is NOT real patient data and should not be used for clinical decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# RESEARCH-BASED BACTERIA DEFINITIONS
# =============================================================================

# Bacteria with PUBLISHED associations to mental health
# Direction and strength based on cited research
RESEARCH_BACKED_BACTERIA = {
    # PROTECTIVE (associated with better mental health)
    # Source: Valles-Colomer 2019, Radjabzadeh 2022
    "Faecalibacterium": {
        "baseline_mean": 8.5,
        "baseline_std": 4.2,
        "depression_effect": -0.35,  # Lower in depression
        "anxiety_effect": -0.25,
        "evidence": "Strong - Valles-Colomer 2019, multiple replications",
        "mechanism": "Butyrate production, anti-inflammatory"
    },
    "Coprococcus": {
        "baseline_mean": 3.2,
        "baseline_std": 2.1,
        "depression_effect": -0.40,  # Significantly lower in depression
        "anxiety_effect": -0.30,
        "evidence": "Strong - Valles-Colomer 2019, Radjabzadeh 2022",
        "mechanism": "DOPAC synthesis pathway"
    },
    "Dialister": {
        "baseline_mean": 2.1,
        "baseline_std": 1.5,
        "depression_effect": -0.30,
        "anxiety_effect": -0.20,
        "evidence": "Moderate - Valles-Colomer 2019",
        "mechanism": "Unknown, correlational"
    },
    "Lactobacillus": {
        "baseline_mean": 1.8,
        "baseline_std": 1.2,
        "depression_effect": -0.25,
        "anxiety_effect": -0.35,  # Stronger anxiety link
        "evidence": "Strong - Multiple RCTs with probiotics",
        "mechanism": "GABA production, vagal signaling"
    },
    "Bifidobacterium": {
        "baseline_mean": 4.5,
        "baseline_std": 2.8,
        "depression_effect": -0.20,
        "anxiety_effect": -0.30,
        "evidence": "Strong - Probiotic trials",
        "mechanism": "Immune modulation, SCFA production"
    },
    "Subdoligranulum": {
        "baseline_mean": 2.8,
        "baseline_std": 1.8,
        "depression_effect": -0.25,
        "anxiety_effect": -0.15,
        "evidence": "Moderate - Radjabzadeh 2022",
        "mechanism": "Butyrate production"
    },
    "Roseburia": {
        "baseline_mean": 3.5,
        "baseline_std": 2.2,
        "depression_effect": -0.20,
        "anxiety_effect": -0.15,
        "evidence": "Moderate - Associated with fiber metabolism",
        "mechanism": "Butyrate production"
    },
    "Blautia": {
        "baseline_mean": 4.2,
        "baseline_std": 2.5,
        "depression_effect": -0.15,
        "anxiety_effect": -0.10,
        "evidence": "Moderate - General gut health marker",
        "mechanism": "SCFA production"
    },
    
    # RISK-ASSOCIATED (elevated in depression/anxiety)
    # Source: Radjabzadeh 2022, Jiang 2015
    "Eggerthella": {
        "baseline_mean": 0.8,
        "baseline_std": 0.6,
        "depression_effect": 0.35,  # Higher in depression
        "anxiety_effect": 0.25,
        "evidence": "Strong - Radjabzadeh 2022",
        "mechanism": "Cortisol metabolism, inflammatory"
    },
    "Sellimonas": {
        "baseline_mean": 0.5,
        "baseline_std": 0.4,
        "depression_effect": 0.30,
        "anxiety_effect": 0.20,
        "evidence": "Moderate - Radjabzadeh 2022",
        "mechanism": "Unknown"
    },
    "Hungatella": {
        "baseline_mean": 0.6,
        "baseline_std": 0.5,
        "depression_effect": 0.25,
        "anxiety_effect": 0.15,
        "evidence": "Moderate - Radjabzadeh 2022",
        "mechanism": "Unknown"
    },
    "Enterobacteriaceae": {
        "baseline_mean": 1.2,
        "baseline_std": 1.0,
        "depression_effect": 0.30,
        "anxiety_effect": 0.35,
        "evidence": "Moderate - Jiang 2015",
        "mechanism": "LPS production, inflammation"
    },
    "Desulfovibrio": {
        "baseline_mean": 0.4,
        "baseline_std": 0.3,
        "depression_effect": 0.20,
        "anxiety_effect": 0.25,
        "evidence": "Moderate - Sulfate-reducing, inflammatory",
        "mechanism": "H2S production"
    },
    "Alistipes": {
        "baseline_mean": 1.5,
        "baseline_std": 1.1,
        "depression_effect": 0.20,
        "anxiety_effect": 0.15,
        "evidence": "Mixed - Some studies show association",
        "mechanism": "Tryptophan metabolism"
    },
    
    # NEUTRAL/CONTEXT-DEPENDENT (included for realistic microbiome)
    "Bacteroides": {
        "baseline_mean": 18.5,
        "baseline_std": 8.2,
        "depression_effect": 0.05,
        "anxiety_effect": 0.0,
        "evidence": "Inconsistent across studies",
        "mechanism": "Dominant commensal, variable effects"
    },
    "Prevotella": {
        "baseline_mean": 8.2,
        "baseline_std": 7.5,  # High variance - diet dependent
        "depression_effect": -0.05,
        "anxiety_effect": -0.05,
        "evidence": "Weak - Diet-associated",
        "mechanism": "Fiber fermentation"
    },
    "Akkermansia": {
        "baseline_mean": 2.5,
        "baseline_std": 2.0,
        "depression_effect": -0.15,
        "anxiety_effect": -0.10,
        "evidence": "Emerging - Metabolic health marker",
        "mechanism": "Mucin degradation, gut barrier"
    },
    "Ruminococcus": {
        "baseline_mean": 3.8,
        "baseline_std": 2.2,
        "depression_effect": -0.10,
        "anxiety_effect": -0.05,
        "evidence": "Mixed",
        "mechanism": "Fiber degradation"
    },
    "Clostridium": {
        "baseline_mean": 2.2,
        "baseline_std": 1.5,
        "depression_effect": 0.10,
        "anxiety_effect": 0.15,
        "evidence": "Mixed - Diverse genus",
        "mechanism": "Variable by species"
    },
    "Streptococcus": {
        "baseline_mean": 1.5,
        "baseline_std": 1.2,
        "depression_effect": 0.10,
        "anxiety_effect": 0.10,
        "evidence": "Weak",
        "mechanism": "Oral-gut axis"
    },
    "Lachnospira": {
        "baseline_mean": 2.8,
        "baseline_std": 1.8,
        "depression_effect": -0.10,
        "anxiety_effect": -0.10,
        "evidence": "Moderate - Radjabzadeh 2022",
        "mechanism": "SCFA production"
    },
    "Veillonella": {
        "baseline_mean": 1.2,
        "baseline_std": 0.9,
        "depression_effect": 0.05,
        "anxiety_effect": 0.05,
        "evidence": "Weak",
        "mechanism": "Lactate metabolism"
    },
    "Parabacteroides": {
        "baseline_mean": 2.0,
        "baseline_std": 1.4,
        "depression_effect": 0.0,
        "anxiety_effect": 0.0,
        "evidence": "Neutral",
        "mechanism": "Commensal"
    },
}

# Psychobiotic effects for the frontend (educational content)
PSYCHOBIOTIC_EFFECTS = {
    "Faecalibacterium": {
        "effect": "protective",
        "description": "Major butyrate producer; anti-inflammatory effects linked to better mental health",
        "citation": "Valles-Colomer et al. 2019"
    },
    "Coprococcus": {
        "effect": "protective", 
        "description": "Associated with dopamine metabolite (DOPAC) synthesis; depleted in depression",
        "citation": "Valles-Colomer et al. 2019"
    },
    "Dialister": {
        "effect": "protective",
        "description": "Consistently depleted in individuals with depression",
        "citation": "Valles-Colomer et al. 2019"
    },
    "Lactobacillus": {
        "effect": "protective",
        "description": "Produces GABA; used in probiotic interventions for anxiety",
        "citation": "Multiple RCTs, Simpson et al. 2021 review"
    },
    "Bifidobacterium": {
        "effect": "protective",
        "description": "Immune modulation; reduced in depression; probiotic strain B. longum shows promise",
        "citation": "Aizawa et al. 2016, probiotic trials"
    },
    "Subdoligranulum": {
        "effect": "protective",
        "description": "Butyrate producer; depleted in depression",
        "citation": "Radjabzadeh et al. 2022"
    },
    "Roseburia": {
        "effect": "protective",
        "description": "Butyrate producer; supports gut-brain signaling",
        "citation": "General microbiome research"
    },
    "Blautia": {
        "effect": "protective",
        "description": "SCFA producer; associated with healthy gut",
        "citation": "General microbiome research"
    },
    "Eggerthella": {
        "effect": "risk",
        "description": "Elevated in depression; involved in cortisol and dopamine metabolism",
        "citation": "Radjabzadeh et al. 2022"
    },
    "Sellimonas": {
        "effect": "risk",
        "description": "Associated with depressive symptoms in population studies",
        "citation": "Radjabzadeh et al. 2022"
    },
    "Hungatella": {
        "effect": "risk",
        "description": "Associated with depressive symptoms",
        "citation": "Radjabzadeh et al. 2022"
    },
    "Enterobacteriaceae": {
        "effect": "risk",
        "description": "Gram-negative family; LPS production may trigger inflammation and neuroinflammation",
        "citation": "Jiang et al. 2015"
    },
    "Desulfovibrio": {
        "effect": "risk",
        "description": "Sulfate-reducing bacteria; H2S production may affect gut-brain axis",
        "citation": "Mechanistic studies"
    },
    "Alistipes": {
        "effect": "risk",
        "description": "May affect tryptophan metabolism; mixed evidence",
        "citation": "Various studies"
    },
    "Akkermansia": {
        "effect": "protective",
        "description": "Gut barrier integrity; associated with metabolic and mental health",
        "citation": "Emerging research"
    },
    "Bacteroides": {
        "effect": "neutral",
        "description": "Dominant gut commensal; effects vary by species and context",
        "citation": "Variable across studies"
    },
    "Prevotella": {
        "effect": "neutral",
        "description": "Diet-associated; high in fiber-rich diets",
        "citation": "Diet studies"
    },
}

# Mental health bacteria lists for frontend
MENTAL_HEALTH_BACTERIA = {
    "protective": [
        "Faecalibacterium", "Coprococcus", "Dialister", "Lactobacillus",
        "Bifidobacterium", "Roseburia", "Blautia", "Akkermansia", "Subdoligranulum"
    ],
    "risk": [
        "Eggerthella", "Sellimonas", "Hungatella", "Enterobacteriaceae",
        "Desulfovibrio", "Alistipes"
    ]
}

# Dataset citation info
DATASET_INFO = {
    "type": "Synthetic (Research-Based)",
    "description": "Synthetic microbiome data generated to reflect published research findings",
    "primary_references": [
        {
            "authors": "Valles-Colomer M, et al.",
            "year": 2019,
            "title": "The neuroactive potential of the human gut microbiota in quality of life and depression",
            "journal": "Nature Microbiology",
            "volume": "4(4):623-632",
            "doi": "10.1038/s41564-018-0337-x",
            "key_findings": "Coprococcus and Dialister depleted in depression; Faecalibacterium associated with QoL"
        },
        {
            "authors": "Radjabzadeh D, et al.",
            "year": 2022,
            "title": "Gut microbiome-wide association study of depressive symptoms",
            "journal": "Nature Communications", 
            "volume": "13:7128",
            "doi": "10.1038/s41467-022-34502-3",
            "key_findings": "13 taxa associated with depression including Eggerthella (elevated) and Coprococcus (depleted)"
        }
    ],
    "disclaimer": "This synthetic dataset reflects correlation patterns from published research. It is NOT real patient data and should not be used for clinical decisions."
}

# =============================================================================
# DATA GENERATION
# =============================================================================

_research_dataset = None

def generate_research_based_dataset(n_samples: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic microbiome + mental health data based on published research.
    
    The correlation structure is derived from peer-reviewed findings, making this
    suitable for demonstration and educational purposes.
    """
    np.random.seed(random_seed)
    logger.info(f"Generating research-based synthetic dataset (n={n_samples})...")
    
    data = {"sample_id": [f"SYNTH_{i:04d}" for i in range(n_samples)]}
    
    # Step 1: Generate latent mental health factors
    # These drive both the mental health scores AND affect bacteria levels
    depression_latent = np.random.normal(0, 1, n_samples)
    anxiety_latent = np.random.normal(0, 1, n_samples)
    
    # Correlation between depression and anxiety (common comorbidity ~0.6-0.7)
    anxiety_latent = 0.6 * depression_latent + 0.8 * anxiety_latent
    
    # Step 2: Generate bacteria abundances influenced by mental health
    for bacteria, params in RESEARCH_BACKED_BACTERIA.items():
        base = np.random.normal(
            params["baseline_mean"], 
            params["baseline_std"], 
            n_samples
        )
        
        # Apply research-based effects (scaled by effect size)
        depression_shift = params["depression_effect"] * depression_latent * params["baseline_std"]
        anxiety_shift = params["anxiety_effect"] * anxiety_latent * params["baseline_std"]
        
        # Add individual variation
        noise = np.random.normal(0, params["baseline_std"] * 0.3, n_samples)
        
        abundance = base + depression_shift + anxiety_shift + noise
        
        # Ensure non-negative and realistic bounds
        abundance = np.clip(abundance, 0.01, params["baseline_mean"] * 4)
        
        data[bacteria] = np.round(abundance, 2)
    
    # Step 3: Generate mental health scores
    # PHQ-9 style scoring (0-27 scale)
    depression_score = 5 + depression_latent * 5 + np.random.normal(0, 2, n_samples)
    depression_score = np.clip(np.round(depression_score), 0, 27).astype(int)
    
    # GAD-7 style scoring (0-21 scale)
    anxiety_score = 4 + anxiety_latent * 4 + np.random.normal(0, 1.5, n_samples)
    anxiety_score = np.clip(np.round(anxiety_score), 0, 21).astype(int)
    
    data["depression_score"] = depression_score
    data["anxiety_score"] = anxiety_score
    
    # Categorical levels based on clinical cutoffs
    # PHQ-9: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe
    data["depression_level"] = pd.cut(
        depression_score,
        bins=[-1, 4, 9, 14, 19, 28],
        labels=["minimal", "mild", "moderate", "moderately_severe", "severe"]
    ).astype(str)
    
    # GAD-7: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-21 severe
    data["anxiety_level"] = pd.cut(
        anxiety_score,
        bins=[-1, 4, 9, 14, 22],
        labels=["minimal", "mild", "moderate", "severe"]
    ).astype(str)
    
    df = pd.DataFrame(data)
    
    logger.info(f"Dataset generated: {len(df)} samples, {len(RESEARCH_BACKED_BACTERIA)} bacteria")
    logger.info(f"Depression distribution: {df['depression_level'].value_counts().to_dict()}")
    
    return df


def get_research_dataset() -> pd.DataFrame:
    """Get or generate the research-based dataset (cached)."""
    global _research_dataset
    if _research_dataset is None:
        _research_dataset = generate_research_based_dataset()
    return _research_dataset


def get_bacteria_columns(df: pd.DataFrame) -> List[str]:
    """Get list of bacteria column names."""
    exclude = ['sample_id', 'depression_score', 'anxiety_score', 
               'depression_level', 'anxiety_level']
    return [col for col in df.columns if col not in exclude]


def get_dataset_info() -> Dict:
    """Return information about the dataset and its research basis."""
    return DATASET_INFO


# =============================================================================
# USER DATA HANDLING  
# =============================================================================

def load_user_data(file_content: str, file_type: str = "csv") -> pd.DataFrame:
    """
    Parse user-uploaded microbiome data.
    Supports various formats from consumer testing services.
    """
    from io import StringIO
    
    try:
        if file_type.lower() == "csv":
            df = pd.read_csv(StringIO(file_content))
        elif file_type.lower() == "tsv":
            df = pd.read_csv(StringIO(file_content), sep='\t')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error parsing user data: {e}")
        raise


def calculate_diversity_metrics(df: pd.DataFrame) -> Dict:
    """Calculate alpha diversity metrics for microbiome data."""
    bacteria_cols = get_bacteria_columns(df)
    abundances = df[bacteria_cols].values
    
    # Handle single sample vs multiple
    if len(abundances.shape) == 1:
        abundances = abundances.reshape(1, -1)
    
    results = []
    for sample in abundances:
        # Normalize to proportions
        total = sample.sum()
        if total == 0:
            results.append({"shannon": 0, "simpson": 0, "richness": 0})
            continue
            
        props = sample / total
        props = props[props > 0]  # Remove zeros for log
        
        # Shannon diversity: H = -sum(p * log(p))
        shannon = -np.sum(props * np.log(props))
        
        # Simpson diversity: 1 - sum(p^2)
        simpson = 1 - np.sum(props ** 2)
        
        # Richness: count of non-zero taxa
        richness = np.sum(sample > 0.01)
        
        results.append({
            "shannon": round(float(shannon), 3),
            "simpson": round(float(simpson), 3),
            "richness": int(richness)
        })
    
    if len(results) == 1:
        return results[0]
    return results
