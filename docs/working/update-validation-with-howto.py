import json

# Load existing validation
with open('hive-diataxis-validation.json', 'r') as f:
    data = json.load(f)

# Normalize "howto" to "guide" in Diataxis framework
if 'howto' in data['category_distribution']:
    howto_count = data['category_distribution'].pop('howto')
    data['category_distribution']['guide'] = data['category_distribution'].get('guide', 0) + howto_count

# Recalculate percentages
total_categorized = sum(data['category_distribution'].values())
category_percentages = {}
for cat, count in data['category_distribution'].items():
    percentage = (count / total_categorized * 100) if total_categorized > 0 else 0
    category_percentages[cat] = {
        "count": count,
        "percentage": round(percentage, 2)
    }

data['category_percentages'] = category_percentages

# Recalculate compliance score
tutorial_pct = category_percentages.get("tutorial", {}).get("percentage", 0)
guide_pct = category_percentages.get("guide", {}).get("percentage", 0)
reference_pct = category_percentages.get("reference", {}).get("percentage", 0)
explanation_pct = category_percentages.get("explanation", {}).get("percentage", 0)

compliance_score = 0
if 10 <= tutorial_pct <= 15: compliance_score += 25
elif tutorial_pct > 5: compliance_score += 15

if 25 <= guide_pct <= 35: compliance_score += 25
elif guide_pct > 15: compliance_score += 15

if 20 <= reference_pct <= 30: compliance_score += 25
elif reference_pct > 10: compliance_score += 15

if 30 <= explanation_pct <= 40: compliance_score += 25
elif explanation_pct > 20: compliance_score += 15

data['diataxis_compliance_percentage'] = round(compliance_score, 2)

# Update mismatches to normalize "howto" -> "guide"
for mismatch in data['category_mismatches']:
    if mismatch['declared_category'] == 'howto':
        mismatch['declared_category'] = 'guide'
    if mismatch['suggested_category'] == 'howto':
        mismatch['suggested_category'] = 'guide'

# Save updated JSON
with open('hive-diataxis-validation.json', 'w') as f:
    json.dump(data, f, indent=2)

print(json.dumps({
    "updated": True,
    "new_compliance": f"{data['diataxis_compliance_percentage']}%",
    "categories": {k: v['percentage'] for k, v in category_percentages.items()}
}, indent=2))
