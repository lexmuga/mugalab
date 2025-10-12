"""
Text_Preprocessing_Helpers_v3.py

Text preprocessing utilities for legal document analysis with improved error handling and logging.

Features:
- Comprehensive text cleaning and normalization
- Legal phrase preservation
- HTML and artifact removal
- Enhanced spaCy preprocessing with multiprocessing support
- Improved logging and error handling

References:
- spaCy documentation: https://spacy.io/usage
- NLTK documentation: https://www.nltk.org/
"""

import re
import unicodedata
import warnings
import logging
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from spacy.language import Language

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

def load_spacy_model(model: str = "en_core_web_sm") -> Language:
    """
    Load a spaCy English pipeline configured for fast text preprocessing.
    The dependency parser and NER are disabled to reduce latency and memory
    since vectorization relies on tokenization and lemmatization (and optionally POS).
    
    Args:
        model: Name of the spaCy model to load.
        
    Returns:
        Loaded spaCy language model.
    """
    try:
        nlp = spacy.load(model, disable=["parser", "ner"])
        logger.info(f"Successfully loaded spaCy model: {model}")
        return nlp
    except Exception as e:
        logger.error(f"Failed to load spaCy model {model}: {e}")
        raise

def fix_unicode_normalization(text: str) -> str:
    """
    Normalize Unicode and repair common mojibake to stabilize tokenization.
    
    Args:
        text: Raw text string.
        
    Returns:
        Cleaned, normalized text string suitable for tokenization.
    """
    if not isinstance(text, str):
        return "" if pd.isna(text) else str(text)
    
    # 1) Unicode normalization to NFC (compose base + diacritics)
    text = unicodedata.normalize('NFC', text)
    
    # 2) Repair frequent mojibake patterns in legal texts (extend as needed)
    mojibake_fixes = {
        'äúthe': 'the',
        'äì': 'a',
        'äòthe': 'the',
        'äôthe': 'the',
        'äù': 'u',
        'äî': 'i',
        'äó': 'o',
        'äë': 'e',
        'Ã¡': 'á',
        'Ã©': 'é',
        'Ã\xad': 'í',
        'Ã³': 'ó',
        'Ãº': 'ú',
        'â€™': "'",
        'â€œ': '"',
        'â€\x9d': '"',   # note: can also be part of " or " depending on source
        'â€”': '—',  # em dash; verify against your corpus
        # If you intended en dash, consider a separate mapping for 'â€"' -> '–'
    }
    for bad, good in mojibake_fixes.items():
        text = text.replace(bad, good)
    
    # 3) Guard against truncation artifacts at boundaries (spurious fragments)
    text = re.sub(r'\b[a-zA-Z]{1,2}$', '', text)  # drop 1-2 letter trailing fragment
    text = re.sub(r'^[a-zA-Z]{1,2}\b', '', text)  # drop 1-2 letter leading fragment
    
    # 4) Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preserve_legal_phrases(text: str) -> str:
    """
    Join domain-specific legal multiword expressions (MWEs) with underscores so
    vectorizers and tokenizers treat them as single, semantically meaningful tokens.
    
    Args:
        text: Preprocessed text.
        
    Returns:
        Text with curated legal phrases underscore-joined.
    """
    # Common legal MWEs (extend from the companion phrase-mining notebook)
    legal_phrases = [
        # Procedural
        'summary judgment', 'summary judgement', 'motion to dismiss',
        'preliminary injunction', 'temporary restraining order',
        'class action', 'due process', 'equal protection',
        'probable cause', 'reasonable doubt', 'burden of proof',
        'preponderance of evidence', 'clear and convincing',
        'beyond reasonable doubt', 'strict scrutiny', 'intermediate scrutiny',
        'rational basis', 'compelling interest', 'least restrictive means',
        # Constitutional
        'first amendment', 'second amendment', 'fourth amendment',
        'fifth amendment', 'sixth amendment', 'eighth amendment',
        'fourteenth amendment', 'commerce clause', 'supremacy clause',
        'establishment clause', 'free exercise clause', 'free speech',
        'freedom of religion', 'right to counsel', 'miranda rights',
        # Civil rights
        'civil rights', 'voting rights', 'equal opportunity',
        'affirmative action', 'disparate impact', 'disparate treatment',
        'hostile environment', 'quid pro quo', 'reasonable accommodation',
        # Criminal law
        'criminal procedure', 'search and seizure', 'exclusionary rule',
        'fruit of poisonous tree', 'good faith exception',
        'inevitable discovery', 'plain view', 'hot pursuit',
        'exigent circumstances', 'stop and frisk', 'terry stop',
        # Contract/tort
        'breach of contract', 'specific performance', 'liquidated damages',
        'punitive damages', 'compensatory damages', 'negligence per se',
        'res ipsa loquitur', 'contributory negligence', 'comparative negligence',
        'assumption of risk', 'statute of limitations',
        # Corporate/business
        'piercing corporate veil', 'business judgment rule',
        'derivative suit', 'shareholder derivative', 'insider trading',
        'securities fraud', 'antitrust violation', 'price fixing',
        'market manipulation', 'unfair competition',
    ]
    
    # Prioritize longest matches to prevent partial replacements
    legal_phrases.sort(key=len, reverse=True)
    
    # Regex word-boundary replacement (case-insensitive)
    for phrase in legal_phrases:
        phrase_pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
        replacement = phrase.replace(' ', '_')
        text = phrase_pattern.sub(replacement, text)
    
    return text

def clean_html_and_artifacts(text: str) -> str:
    """
    Remove HTML tags and common extraction artifacts that do not contribute
    to semantic content, while keeping the transformation conservative.
    """
    if not isinstance(text, str):
        return ""

    # 1) Drop simple HTML tags (fast heuristic). Consider a full parser if needed.
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2) Remove common boilerplate/artifacts
    #    - Bracketed tokens often denote editorial notes like [citation needed], [Page 12].
    text = re.sub(r'\[[^\[\]\n]{0,80}\]', ' ', text)

    #    - Parenthetical removal is optional; many legal citations live in parentheses.
    #      Keep this if you've validated that your use case benefits from stripping.
    text = re.sub(r'\([^\(\)\n]{0,120}\)', ' ', text)

    #    - Remove leading list numbers or outline markers (e.g., "1. ", "2) ", "III.").
    text = re.sub(r'^\s*(?:\(?\\d{1,3}\)?[.)]|\b[IVXLC]{1,6}[.)])\s+', ' ', text, flags=re.MULTILINE)

    # 3) Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def enhanced_spacy_preprocessing(
    texts: pd.Series,
    nlp: Language,
    keep_negations: bool = True,
    allowed_pos: Optional[Set[str]] = None,
    preserve_legal_terms: bool = True,
    batch_size: int = 100,
    n_process: int = 4
) -> pd.Series:
    """
    Enhanced spaCy preprocessing pipeline with legal domain awareness.
    
    Args:
        texts: Series of raw text documents
        nlp: Loaded spaCy model
        keep_negations: Whether to preserve negation words
        allowed_pos: Set of allowed POS tags (None = all)
        preserve_legal_terms: Whether to join legal multiword terms
        batch_size: Batch size for spaCy processing
        n_process: Number of processes for parallel processing
        
    Returns:
        Series of preprocessed texts
    """
    logger.info("Starting enhanced text preprocessing pipeline...")
    
    # Step 1: Basic cleaning and normalization
    logger.info("  Step 1/4: Unicode normalization and mojibake removal")
    cleaned_texts = texts.fillna("").apply(clean_html_and_artifacts)
    normalized_texts = cleaned_texts.apply(fix_unicode_normalization)
    
    # Step 2: Legal phrase preservation
    if preserve_legal_terms:
        logger.info("  Step 2/4: Preserving legal multiword terms")
        normalized_texts = normalized_texts.apply(preserve_legal_phrases)
    else:
        logger.info("  Step 2/4: Skipping legal phrase preservation")
    
    # Step 3: spaCy processing
    logger.info(f"  Step 3/4: spaCy processing (batch_size={batch_size}, n_process={n_process})")
    
    # Negation words to preserve
    NEGATIONS = {
        "no", "not", "n't", "never", "none", "cannot", "cant", "won't",
        "don't", "didn't", "doesn't", "haven't", "hasn't", "hadn't",
        "wouldn't", "couldn't", "shouldn't", "mustn't", "needn't"
    }
    
    def process_doc(doc):
        """Process individual spaCy document"""
        tokens = []
        for token in doc:
            # Skip spaces and punctuation
            if token.is_space or token.is_punct:
                continue
                
            # Skip numbers and currency (optional for legal texts)
            if token.like_num or token.is_currency:
                continue
                
            # POS filtering
            if allowed_pos is not None and token.pos_ not in allowed_pos:
                continue
                
            # Handle stop words with negation preservation
            if token.is_stop:
                if keep_negations and (
                    token.text.lower() in NEGATIONS or 
                    token.lemma_.lower() in NEGATIONS
                ):
                    pass  # Keep negations
                else:
                    continue  # Skip other stop words
            
            # Get lemma and clean
            lemma = token.lemma_.lower().strip()
            if not lemma or not lemma.isalpha():
                continue
                
            tokens.append(lemma)
        
        return " ".join(tokens)
    
    # Process with spaCy pipeline
    docs = nlp.pipe(normalized_texts, batch_size=batch_size, n_process=n_process)
    processed_texts = [process_doc(doc) for doc in docs]
    
    logger.info("  Step 4/4: Final cleanup and validation")
    # Convert back to pandas Series with original index
    result = pd.Series(processed_texts, index=texts.index)
    
    # Final validation
    empty_count = (result.str.len() == 0).sum()
    if empty_count > 0:
        logger.warning(f"  Warning: {empty_count} documents became empty after preprocessing")
    
    logger.info(f"Preprocessing complete. Average tokens per document: {result.str.split().str.len().mean():.1f}")
    
    return result

def create_train_val_test_split(
    df: pd.DataFrame,
    text_col: str,
    target_col: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 103
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Create a 70/15/15 train/validation/test split with stratification when possible.
    
    Args:
        df: Input dataframe
        text_col: Name of text column
        target_col: Name of target column
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with train/val/test splits for texts and labels
    """
    # Validate split ratios
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_size + val_size + test_size}")
    
    # Data preparation
    X_text = df[text_col].astype(str)
    y = df[target_col].dropna()
    
    # Stratify classification when labels are not degenerate
    stratify = y if (y.nunique() > 1 and y.nunique() < len(y) * 0.5) else None
    
    # Two-step split to get exact 70/15/15 proportions
    # Step 1: Split off test set (15%)
    X_temp, X_test_text, y_temp, y_test = train_test_split(
        X_text, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    
    # Step 2: Split remaining into train (70%) and val (15%)
    # val_size_adjusted = val_size / (train_size + val_size) to get exact proportions
    val_size_adjusted = val_size / (train_size + val_size)
    
    # Re-stratify on the remaining data
    stratify_temp = y_temp if (y_temp.nunique() > 1 and y_temp.nunique() < len(y_temp) * 0.5) else None
    
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_temp
    )
    
    logger.info(f"Data split: {len(X_train_text)} train ({len(X_train_text)/len(X_text):.1%}), "
                f"{len(X_val_text)} val ({len(X_val_text)/len(X_text):.1%}), "
                f"{len(X_test_text)} test ({len(X_test_text)/len(X_text):.1%})")
    
    return {
        'X_train_text': X_train_text,
        'X_val_text': X_val_text,
        'X_test_text': X_test_text,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

def main():
    """
    Example usage of the text preprocessing helpers.
    """
    # Load spaCy model
    nlp = load_spacy_model("en_core_web_sm")
    
    # Example dataframe (replace with your actual data)
    df = pd.DataFrame({
        "text": [
            "This is a sample legal document with some <html> tags and [citation needed].",
            "Another example with mojibake: Ã¡ccent and legal terms like summary judgment.",
            "A third document with numbers 123 and currency $500."
        ],
        "label": [0, 1, 0]
    })
    
    # Apply preprocessing
    df["clean_text"] = enhanced_spacy_preprocessing(
        texts=df["facts"].fillna(""),
        # texts=df["text"],
        nlp=nlp,
        keep_negations=True,
        preserve_legal_terms=True,
        batch_size=100,
        n_process=4
    )
    
    # Create train/val/test splits
    splits = create_train_val_test_split(
        df=df,
        text_col="clean_text",
        target_col="label",
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=103
    )
    
    print("Preprocessing complete!")
    print(f"Train set size: {len(splits['X_train_text'])}")
    print(f"Validation set size: {len(splits['X_val_text'])}")
    print(f"Test set size: {len(splits['X_test_text'])}")

if __name__ == "__main__":
    main()
