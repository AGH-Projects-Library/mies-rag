import json
import os
# tylko dla kontenera
if os.path.isfile('/app/queries.json'):
    with open('/app/queries.json', 'r') as f:
        QUERIES = json.load(f)
else:
    QUERIES = [
        {
            "topic": "What is the type of the review?",
            "possible_options": "1 = rapid review,2 = umbrella review / overview of reviews,3 = scoping review,4 = narrative review,5 = slr [quantitative / with MA],6 = slr [qualitative / descriptive / without MA],7 = realist review,8 = evidence map,9 = other"
        },
        {
            "topic": "Is reporting checklist considering AI present / is there information that (we followed...) present?",
            "possible_options": "1 = No,2 = PRISMA 2020,3 = TRIPOD-SRMA,4 = MDRLES,5 = Other"
        },
        {
            "topic": "Do the authors report on artificial intelligence in the checklist?",
            "possible_options": "1 = No mentions aboiut AI,2 = No access to the checklist,3 = No (Not reported in checklist),4 = Yes (Reported in checklist)"
        },
        {
            "topic": "Do authors report on artificial intelligence in a flowchart?",
            "possible_options": "1 = No flowchart,2 = No,3 = Yes (n>0),4 = Yes(n=0, without auto*)"
        },
        {
            "topic": "Do authors report on artificial intelligence in methods/results?",
            "possible_options": "1 = No,2 = Yes (used),3 = Yes (not used)"
        },
    ]
