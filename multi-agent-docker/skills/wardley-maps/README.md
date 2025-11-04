# Wardley Mapper Skill - Complete Package

## 🎯 Overview

This comprehensive Claude skill transforms ANY input into strategic Wardley maps. Whether you have structured data, unstructured text, business descriptions, technical architectures, or competitive landscapes - this skill will create insightful visual maps for strategic decision-making.

## 📦 What's Included

### Core Files
- **`wardley-mapper-skill/`** - The complete skill package
  - `SKILL.md` - Main skill instructions
  - `references/` - Specialized mappers for different inputs
  - `scripts/` - Python tools for generating maps
  - `assets/` - Templates and examples

### Demonstration Files
- **`visionflow_wardley_map.html`** - Example interactive Wardley map
- **`visionflow_analysis.md`** - Strategic analysis example
- **`wardley_demo.py`** - Demonstration script

## 🚀 Quick Start

### Method 1: Interactive Mode
```python
cd wardley-mapper-skill
python3 scripts/quick_map.py
# Follow the prompts to create your map
```

### Method 2: From Text Description
```python
from scripts.generate_wardley_map import WardleyMapGenerator

# Your business description
text = "We provide cloud-based analytics..."

# Generate map
generator = WardleyMapGenerator()
components = parse_text_to_components(text)
html = generator.create_map(components)
```

### Method 3: From Structured Data
```python
# Define components with visibility and evolution
components = [
    {"name": "User Interface", "visibility": 0.9, "evolution": 0.7},
    {"name": "Backend API", "visibility": 0.6, "evolution": 0.5},
    {"name": "Database", "visibility": 0.3, "evolution": 0.8}
]

# Define relationships
dependencies = [
    ("User Interface", "Backend API"),
    ("Backend API", "Database")
]

# Generate map
generator = WardleyMapGenerator()
html = generator.create_map(components, dependencies)
```

## 🎨 Key Features

### Universal Input Processing
- **Business Descriptions** → Strategic maps
- **Technical Architectures** → System maps
- **Competitive Intelligence** → Market maps
- **Financial Data** → Value chain maps
- **Organizational Structures** → Capability maps

### Intelligent Component Positioning
- **Y-Axis (Value Chain)**: Automatic visibility assessment
- **X-Axis (Evolution)**: Smart evolution stage detection
  - Genesis (0.0-0.2): Novel, experimental
  - Custom (0.2-0.5): Differentiated, proprietary
  - Product (0.5-0.8): Standardizing, competing
  - Commodity (0.8-1.0): Utility, outsourced

### Visual Output Options
- **Interactive HTML**: Zoom, pan, hover details
- **Static SVG**: For presentations
- **PNG Export**: For documents
- **JSON Format**: For programmatic use

## 🧠 How It Works

### 1. Input Analysis
The skill uses pattern recognition to identify:
- Components (nouns, entities, capabilities)
- Relationships (dependencies, flows)
- Evolution indicators (maturity keywords)
- Value indicators (user proximity)

### 2. Intelligent Mapping
- **NLP Processing**: Extracts meaning from text
- **Pattern Matching**: Identifies strategic patterns
- **Context Analysis**: Understands domain specifics
- **Relationship Inference**: Detects dependencies

### 3. Strategic Analysis
Beyond visualization, the skill provides:
- Evolution predictions
- Competitive positioning
- Strategic options
- Risk identification
- Opportunity detection

## 📊 Example Use Cases

### Startup Strategy
```
"We're building an AI chatbot platform using GPT-4, 
with custom training on industry data, deployed on AWS"
```
→ Map shows GPT-4 as commodity, custom training as differentiator

### Digital Transformation
```
"Modernizing our legacy mainframe systems with cloud-native 
microservices and API-first architecture"
```
→ Map reveals evolution gaps and transformation pathway

### Competitive Analysis
```
"Competitors use standard CRM, we've built proprietary 
customer intelligence with predictive analytics"
```
→ Map highlights competitive advantage in custom analytics

## 🛠️ Customization

### Modify Evolution Assessment
Edit `references/business-mapper.md` evolution keywords

### Add Industry Templates
Add to `assets/templates.json`

### Enhance NLP Processing
Modify `scripts/quick_map.py` parsing functions

### Style Customization
Edit HTML/CSS in `generate_wardley_map.py`

## 📈 Strategic Patterns Included

The skill includes advanced strategic patterns:
- **Commoditization plays**
- **Innovation strategies**
- **Ecosystem building**
- **Disruption patterns**
- **Platform strategies**
- **Red Queen dynamics**

## 🔍 Validation

Each generated map includes:
- ✅ Clear user need
- ✅ Justified evolution positions
- ✅ Mapped dependencies
- ✅ No orphaned components
- ✅ Actionable insights

## 💡 Pro Tips

1. **Start Simple**: Begin with high-level components, refine later
2. **Challenge Positions**: Question evolution assumptions
3. **Look for Gaps**: Empty spaces often reveal opportunities
4. **Track Movement**: Components evolve over time
5. **Consider Inertia**: Not everything evolves at same pace

## 📚 References

Based on Simon Wardley's pioneering work in strategic mapping:
- Book: "Wardley Maps" (included as source)
- Evolution characteristics
- Climatic patterns
- Doctrine principles
- Strategic gameplay

## 🚦 Getting Started

1. **Open the example**: `visionflow_wardley_map.html`
2. **Read the analysis**: `visionflow_analysis.md`
3. **Try the interactive tool**: Run `scripts/quick_map.py`
4. **Create your own map**: Use any input method above

## 🎯 This Skill Enables You To

- **See** your competitive landscape clearly
- **Understand** evolution and change
- **Identify** strategic opportunities
- **Predict** market movements
- **Communicate** strategy visually
- **Make** better decisions

## 🔮 Future Enhancements

Potential additions:
- Real-time collaboration features
- AI-powered strategy suggestions
- Industry benchmark overlays
- Evolution simulation over time
- Competitive war gaming
- API integration for live data

---

**Created with the Wardley Mapper Skill v1.0**
Transform anything into strategic insight! 🗺️
