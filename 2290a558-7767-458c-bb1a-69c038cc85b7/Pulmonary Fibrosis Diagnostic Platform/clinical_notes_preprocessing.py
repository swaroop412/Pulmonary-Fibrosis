import pandas as pd
import numpy as np
import re
from collections import Counter

# Clinical Notes Text Preprocessing Pipeline with Medical Entity Extraction
print("=" * 70)
print("CLINICAL NOTES TEXT PREPROCESSING PIPELINE")
print("=" * 70)

# Medical entity patterns for pulmonary fibrosis clinical notes
medical_entities = {
    'symptoms': [
        'dyspnea', 'shortness of breath', 'sob', 'cough', 'fatigue', 'chest pain',
        'wheezing', 'cyanosis', 'weight loss', 'fever', 'hemoptysis'
    ],
    'findings': [
        'fibrosis', 'honeycombing', 'ground glass', 'reticular pattern',
        'traction bronchiectasis', 'subpleural', 'bibasilar', 'crackles',
        'decreased breath sounds', 'rhonchi', 'decreased lung volumes'
    ],
    'medications': [
        'nintedanib', 'pirfenidone', 'prednisone', 'azathioprine',
        'n-acetylcysteine', 'oxygen', 'bronchodilator', 'immunosuppressant'
    ],
    'procedures': [
        'pulmonary function test', 'pft', 'ct scan', 'chest x-ray',
        'bronchoscopy', 'lung biopsy', '6-minute walk test', 'spirometry'
    ],
    'measurements': [
        'fvc', 'forced vital capacity', 'dlco', 'fev1', 'tco', 'spo2',
        'oxygen saturation', 'vital capacity', 'lung volume'
    ]
}

# Text preprocessing functions
def clean_clinical_text(text):
    """Clean and normalize clinical text"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep medical notation
    text = re.sub(r'[^\w\s\-\.\,\:\;\/]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_medical_entities(text, entity_dict):
    """Extract medical entities from clinical text"""
    text_lower = text.lower()
    found_entities = {}
    
    for entity_type, entity_list in entity_dict.items():
        found = []
        for entity in entity_list:
            if entity in text_lower:
                found.append(entity)
        found_entities[entity_type] = found
    
    return found_entities

def extract_numerical_values(text):
    """Extract numerical measurements from text"""
    # Pattern for measurements like "FVC: 2850 mL" or "SpO2 95%"
    patterns = {
        'fvc_values': r'fvc[:\s]+(\d+\.?\d*)\s*(ml|l)?',
        'oxygen_sat': r'(?:spo2|oxygen saturation)[:\s]+(\d+\.?\d*)%?',
        'dlco_values': r'dlco[:\s]+(\d+\.?\d*)',
        'percentages': r'(\d+\.?\d*)%'
    }
    
    extracted = {}
    for measure_type, pattern in patterns.items():
        matches = re.findall(pattern, text.lower())
        if matches:
            extracted[measure_type] = matches
    
    return extracted

def tokenize_medical_text(text):
    """Tokenize medical text preserving medical terms"""
    # Simple tokenization
    tokens = text.lower().split()
    
    # Remove very short tokens (likely noise)
    tokens = [t for t in tokens if len(t) > 2]
    
    return tokens

def extract_negations(text):
    """Identify negated medical terms"""
    negation_patterns = [
        r'no\s+(\w+)',
        r'denies\s+(\w+)',
        r'without\s+(\w+)',
        r'negative\s+for\s+(\w+)'
    ]
    
    negated_terms = []
    for pattern in negation_patterns:
        matches = re.findall(pattern, text.lower())
        negated_terms.extend(matches)
    
    return negated_terms

# Create sample clinical notes for demonstration
sample_clinical_notes = [
    """Patient presents with progressive dyspnea and dry cough over 6 months. 
    CT scan shows bilateral subpleural honeycombing and ground glass opacities. 
    PFTs reveal FVC 2850 mL (65% predicted), DLCO 45% predicted. 
    Started on pirfenidone 801mg TID.""",
    
    """Follow-up visit. Patient reports worsening shortness of breath. 
    6-minute walk test: 320 meters with SpO2 nadir 88%. 
    Recent CT shows progression of fibrosis. FVC declined to 2450 mL. 
    Increased oxygen to 3L/min.""",
    
    """Stable disease. No significant change in symptoms. 
    Patient tolerating nintedanib well without side effects. 
    PFTs stable: FVC 3100 mL, DLCO 52% predicted. 
    Continue current management.""",
    
    """New patient evaluation. History of smoking (30 pack-years). 
    Bibasilar crackles on exam. Chest X-ray shows reticular pattern. 
    Recommended high-resolution CT and complete PFT battery."""
]

# Process clinical notes
print("\n📝 PROCESSING SAMPLE CLINICAL NOTES")
print("-" * 70)

processed_notes = []
all_entities = []
all_measurements = []

for note_idx, clinical_note in enumerate(sample_clinical_notes):
    print(f"\n--- Clinical Note {note_idx + 1} ---")
    
    # Clean text
    cleaned = clean_clinical_text(clinical_note)
    
    # Extract entities
    entities = extract_medical_entities(cleaned, medical_entities)
    
    # Extract measurements
    measurements = extract_numerical_values(cleaned)
    
    # Tokenize
    tokens = tokenize_medical_text(cleaned)
    
    # Extract negations
    negations = extract_negations(cleaned)
    
    processed = {
        'note_id': f'note_{note_idx+1}',
        'original_length': len(clinical_note),
        'cleaned_text': cleaned,
        'token_count': len(tokens),
        'entities': entities,
        'measurements': measurements,
        'negated_terms': negations
    }
    
    processed_notes.append(processed)
    all_entities.append(entities)
    all_measurements.append(measurements)
    
    print(f"Original length: {len(clinical_note)} chars")
    print(f"Token count: {len(tokens)}")
    print(f"Entities found: {sum(len(v) for v in entities.values())}")
    print(f"Measurements: {list(measurements.keys())}")

# Create preprocessing configuration
clinical_text_config = {
    'preprocessing_steps': [
        '1. Text cleaning and normalization',
        '2. Medical entity recognition',
        '3. Numerical value extraction',
        '4. Negation detection',
        '5. Tokenization',
        '6. Stop word removal (medical-aware)',
        '7. Feature engineering (TF-IDF, embeddings)',
        '8. Sequence padding for model input'
    ],
    'entity_categories': list(medical_entities.keys()),
    'total_entities': sum(len(v) for v in medical_entities.values()),
    'embedding_options': [
        'BioBERT (medical domain)',
        'ClinicalBERT',
        'PubMedBERT',
        'SciBERT',
        'Word2Vec (trained on clinical corpus)'
    ],
    'max_sequence_length': 512,
    'vocabulary_size': 10000
}

print("\n\n📊 TEXT PREPROCESSING CONFIGURATION")
print("-" * 70)
print(f"Entity Categories: {len(clinical_text_config['entity_categories'])}")
print(f"Total Entity Terms: {clinical_text_config['total_entities']}")
print(f"Max Sequence Length: {clinical_text_config['max_sequence_length']}")

print("\n🔧 PREPROCESSING PIPELINE STEPS")
print("-" * 70)
for text_step in clinical_text_config['preprocessing_steps']:
    print(f"  {text_step}")

print("\n🤖 RECOMMENDED EMBEDDINGS")
print("-" * 70)
for text_embedding in clinical_text_config['embedding_options']:
    print(f"  • {text_embedding}")

print("\n📈 ENTITY EXTRACTION SUMMARY")
print("-" * 70)
for text_entity_type, text_entity_list in medical_entities.items():
    print(f"  {text_entity_type.capitalize()}: {len(text_entity_list)} terms")

print("\n✅ CLINICAL TEXT PREPROCESSING PIPELINE READY")
print("=" * 70)

# Store outputs
clinical_pipeline_ready = True
total_entity_categories = len(medical_entities)
sample_notes_processed = len(processed_notes)
