#!/usr/bin/env python3
"""
US to UK English Conversion Dictionary

Comprehensive mapping of US spellings to UK English spellings.
Organized by category for easy maintenance and extension.
"""

# Core spelling differences (-or/-our)
OR_TO_OUR = {
    'behavior': 'behaviour',
    'color': 'colour',
    'favor': 'favour',
    'flavor': 'flavour',
    'harbor': 'harbour',
    'honor': 'honour',
    'labor': 'labour',
    'neighbor': 'neighbour',
    'rumor': 'rumour',
    'vapor': 'vapour',
}

# -er/-re endings
ER_TO_RE = {
    'center': 'centre',
    'fiber': 'fibre',
    'liter': 'litre',
    'meter': 'metre',
    'theater': 'theatre',
    'caliber': 'calibre',
    'saber': 'sabre',
    'somber': 'sombre',
}

# -ize/-ise endings
IZE_TO_ISE = {
    'analyze': 'analyse',
    'authorize': 'authorise',
    'capitalize': 'capitalise',
    'categorize': 'categorise',
    'centralize': 'centralise',
    'customize': 'customise',
    'digitize': 'digitise',
    'emphasize': 'emphasise',
    'finalize': 'finalise',
    'initialize': 'initialise',
    'maximize': 'maximise',
    'minimize': 'minimise',
    'modernize': 'modernise',
    'normalize': 'normalise',
    'optimize': 'optimise',
    'organize': 'organise',
    'realize': 'realise',
    'recognize': 'recognise',
    'standardize': 'standardise',
    'synchronize': 'synchronise',
    'utilize': 'utilise',
    'visualize': 'visualise',
}

# -yze/-yse endings
YZE_TO_YSE = {
    'analyze': 'analyse',
    'paralyze': 'paralyse',
    'catalyze': 'catalyse',
}

# -og/-ogue endings
OG_TO_OGUE = {
    'analog': 'analogue',
    'catalog': 'catalogue',
    'dialog': 'dialogue',
}

# -ense/-ence endings
ENSE_TO_ENCE = {
    'defense': 'defence',
    'license': 'licence',  # noun
    'offense': 'offence',
    'pretense': 'pretence',
}

# Double consonants
SINGLE_TO_DOUBLE = {
    'canceled': 'cancelled',
    'canceling': 'cancelling',
    'counselor': 'counsellor',
    'fueled': 'fuelled',
    'fueling': 'fuelling',
    'labeled': 'labelled',
    'labeling': 'labelling',
    'leveled': 'levelled',
    'leveling': 'levelling',
    'modeled': 'modelled',
    'modeling': 'modelling',
    'traveled': 'travelled',
    'traveling': 'travelling',
    'traveler': 'traveller',
}

# Miscellaneous common differences
MISCELLANEOUS = {
    'aging': 'ageing',
    'aluminum': 'aluminium',
    'artifact': 'artefact',
    'ax': 'axe',
    'check': 'cheque',  # financial context
    'cozy': 'cosy',
    'draft': 'draught',  # specific contexts
    'gray': 'grey',
    'jewelry': 'jewellery',
    'maneuver': 'manoeuvre',
    'pajamas': 'pyjamas',
    'plow': 'plough',
    'program': 'programme',  # TV/radio context
    'skeptic': 'sceptic',
    'tire': 'tyre',  # wheel context
}

# Technology terms (keep US spelling)
TECH_EXCEPTIONS = {
    'program',  # computer program (keep US)
    'disk',  # computer disk (keep US)
    'byte',  # keep US
}

def build_full_dictionary():
    """
    Build complete US→UK conversion dictionary.

    Returns:
        dict: Complete mapping with all forms (including capitalized)
    """
    full_dict = {}

    # Combine all dictionaries
    all_mappings = {}
    all_mappings.update(OR_TO_OUR)
    all_mappings.update(ER_TO_RE)
    all_mappings.update(IZE_TO_ISE)
    all_mappings.update(YZE_TO_YSE)
    all_mappings.update(OG_TO_OGUE)
    all_mappings.update(ENSE_TO_ENCE)
    all_mappings.update(SINGLE_TO_DOUBLE)
    all_mappings.update(MISCELLANEOUS)

    # Add base forms and variations
    for us_word, uk_word in all_mappings.items():
        # Base form
        full_dict[us_word] = uk_word

        # Capitalized
        full_dict[us_word.capitalize()] = uk_word.capitalize()

        # All caps
        full_dict[us_word.upper()] = uk_word.upper()

        # Add common variations
        if us_word.endswith('e'):
            base = us_word[:-1]
            uk_base = uk_word[:-1]
            # -ed, -ing, -er, -est forms
            full_dict[base + 'ed'] = uk_base + 'ed'
            full_dict[base + 'ing'] = uk_base + 'ing'
            full_dict[base + 'er'] = uk_base + 'er'
            full_dict[base + 'est'] = uk_base + 'est'

        # Add plural forms
        if not us_word.endswith('s'):
            full_dict[us_word + 's'] = uk_word + 's'
            full_dict[us_word.capitalize() + 's'] = uk_word.capitalize() + 's'

    return full_dict


def get_conversion_dictionary():
    """Get the complete US→UK conversion dictionary."""
    return build_full_dictionary()


def is_tech_term(word: str) -> bool:
    """
    Check if word should keep US spelling (technical term).

    Args:
        word: Word to check

    Returns:
        bool: True if word should keep US spelling
    """
    return word.lower() in TECH_EXCEPTIONS


if __name__ == '__main__':
    # Test the dictionary
    full_dict = build_full_dictionary()
    print(f"Total US→UK mappings: {len(full_dict)}")
    print("\nSample conversions:")
    for us, uk in list(full_dict.items())[:20]:
        print(f"  {us} → {uk}")
