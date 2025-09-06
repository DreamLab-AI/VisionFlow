#!/bin/sed -f
# US to UK English Conversion Script
# This script converts common American spellings to British English equivalents

# -ize to -ise endings (but preserve some technical terms)
s/\b([Cc])entralize\b/\1entralise/g
s/\b([Cc])haracterize\b/\1haracterise/g
s/\b([Cc])ivilize\b/\1ivilise/g
s/\b([Cc])ustomize\b/\1ustomise/g
s/\b([Dd])ramatize\b/\1ramatise/g
s/\b([Ee])mphasize\b/\1mphasise/g
s/\b([Ff])ertilize\b/\1ertilise/g
s/\b([Gg])eneralize\b/\1eneralise/g
s/\b([Ii])dealize\b/\1dealise/g
s/\b([Ll])egalize\b/\1egalise/g
s/\b([Mm])aximize\b/\1aximise/g
s/\b([Mm])inimize\b/\1inimise/g
s/\b([Mm])odernize\b/\1odernise/g
s/\b([Nn])ormalize\b/\1ormalise/g
s/\b([Oo])rganize\b/\1rganise/g
s/\b([Oo])ptimize\b/\1ptimise/g
s/\b([Pp])rioritize\b/\1rioritise/g
s/\b([Rr])ecognize\b/\1ecognise/g
s/\b([Rr])ealize\b/\1ealise/g
s/\b([Ss])pecialize\b/\1pecialise/g
s/\b([Ss])tandardize\b/\1tandardise/g
s/\b([Ss])ynchronize\b/\1ynchronise/g
s/\b([Ss])ynthesiz\b/\1ynthesis/g
s/\b([Uu])tilize\b/\1tilise/g
s/\b([Vv])isualize\b/\1isualise/g

# -ization to -isation
s/\b([Cc])entralization\b/\1entralisation/g
s/\b([Cc])haracterization\b/\1haracterisation/g
s/\b([Cc])ivilization\b/\1ivilisation/g
s/\b([Cc])ustomization\b/\1ustomisation/g
s/\b([Gg])eneralization\b/\1eneralisation/g
s/\b([Oo])rganization\b/\1rganisation/g
s/\b([Oo])ptimization\b/\1ptimisation/g
s/\b([Pp])rioritization\b/\1rioritisation/g
s/\b([Rr])ecognition\b/\1ecognition/g
s/\b([Rr])ealization\b/\1ealisation/g
s/\b([Ss])pecialization\b/\1pecialisation/g
s/\b([Ss])tandardization\b/\1tandardisation/g
s/\b([Ss])ynchronization\b/\1ynchronisation/g
s/\b([Uu])tilization\b/\1tilisation/g
s/\b([Vv])isualization\b/\1isualisation/g

# -or to -our endings
s/\b([Bb])ehavior\b/\1ehaviour/g
s/\b([Cc])olor\b/\1olour/g
s/\b([Ff])avor\b/\1avour/g
s/\b([Hh])arbor\b/\1arbour/g
s/\b([Hh])onor\b/\1onour/g
s/\b([Hh])umor\b/\1umour/g
s/\b([Ll])abor\b/\1abour/g
s/\b([Nn])eighbor\b/\1eighbour/g
s/\b([Rr])umor\b/\1umour/g
s/\b([Ss])avor\b/\1avour/g
s/\b([Vv])apor\b/\1apour/g

# -er to -re endings
s/\b([Cc])enter\b/\1entre/g
s/\b([Ff])iber\b/\1ibre/g
s/\b([Ll])iter\b/\1itre/g
s/\b([Ll])ustre\b/\1ustre/g
s/\b([Mm])eter\b/\1etre/g
s/\b([Tt])heater\b/\1heatre/g

# -ense to -ence
s/\b([Dd])efense\b/\1efence/g
s/\b([Ll])icense\b/\1icence/g
s/\b([Oo])ffense\b/\1ffence/g

# -og to -ogue
s/\b([Aa])nalog\b/\1nalogue/g
s/\b([Cc])atalog\b/\1atalogue/g
s/\b([Dd])ialog\b/\1ialogue/g
s/\b([Ee])pilog\b/\1pilogue/g
s/\b([Pp])rolog\b/\1rologue/g

# Other common US/UK differences
s/\b([Aa])ging\b/\1geing/g
s/\b([Cc])heck\b/\1heck/g
s/\b([Gg])ray\b/\1rey/g
s/\b([Mm])old\b/\1ould/g
s/\b([Pp])low\b/\1lough/g
s/\b([Tt])ire\b/\1yre/g

# Compound words and derivatives
s/\borganisation's\b/organisation's/g
s/\borganised\b/organised/g
s/\borganising\b/organising/g
s/\borganiser\b/organiser/g
s/\borganisers\b/organisers/g
s/\brealised\b/realised/g
s/\brealising\b/realising/g
s/\brecognised\b/recognised/g
s/\brecognising\b/recognising/g
s/\bspecialised\b/specialised/g
s/\bspecialising\b/specialising/g
s/\bstandardised\b/standardised/g
s/\bstandardising\b/standardising/g
s/\bvisualised\b/visualised/g
s/\bvisualising\b/visualising/g
s/\boptimised\b/optimised/g
s/\boptimising\b/optimising/g
s/\bminimised\b/minimised/g
s/\bminimising\b/minimising/g
s/\bmaximised\b/maximised/g
s/\bmaximising\b/maximising/g
s/\bcustomised\b/customised/g
s/\bcustomising\b/customising/g

# Adjectives
s/\bcolored\b/coloured/g
s/\bflavored\b/flavoured/g
s/\bhonored\b/honoured/g
s/\blabored\b/laboured/g
s/\bneighbored\b/neighboured/g

# Past tense and gerunds for -ise verbs
s/\bemphasised\b/emphasised/g
s/\bemphasising\b/emphasising/g
s/\bnormalised\b/normalised/g
s/\bnormalising\b/normalising/g
s/\bprioritised\b/prioritised/g
s/\bprioritising\b/prioritising/g
s/\bsynchronised\b/synchronised/g
s/\bsynchronising\b/synchronising/g
s/\bcentralised\b/centralised/g
s/\bcentralising\b/centralising/g